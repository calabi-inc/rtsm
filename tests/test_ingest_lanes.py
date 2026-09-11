"""Ingest lanes (Gate 4.5 plan, P1 task 3): per-lane admission under policy
`latest`, the producer-paced FIFO under `lossless`, the untouched legacy
queue, the policy resolution, the runner's drop handler, the watchdog's
sustained `backlogged` rule, and the receivers' admit-before-decode call into
the lanes. CPU-only; every timing uses an injected clock.
"""
from __future__ import annotations

import json
import struct
import threading
import time

import cv2
import numpy as np
import pytest

from rtsm.core.datamodel import FramePacket, IngestMeta, TimeBundle
from rtsm.core.watchdog import DEGRADED_STATES, FrameFlowMonitor, PipelineHeartbeat
from rtsm.io.ingest_lanes import (
    DROP_AGE, DROP_CLOSED, DROP_KF_DROPPED, DROP_KF_LANE_FULL, DROP_SUPERSEDED,
    KF_MINTED, KF_SOURCE, LANE_FIFO, LANE_KEYFRAME, LANE_LATEST,
    IngestLanes, LaneConfig, lane_drop_handler, make_ingest_queue, resolve_policy,
)
from rtsm.io.ingest_queue import IngestQueue


# ── helpers ──────────────────────────────────────────────────────────────────

class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def tick(self, dt: float) -> float:
        self.t += dt
        return self.t


def _pkt(seq: int, ts_ns: int | None = None, *, kf: bool = False, origin: str | None = None) -> FramePacket:
    return FramePacket(
        time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=(ts_ns if ts_ns is not None else seq * 1000), seq=seq),
        rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth_m=None, pose=None, intr=None, is_keyframe=kf,
        ingest=IngestMeta(keyframe_origin=(origin if origin is not None else (KF_MINTED if kf else None)), rx_seq=seq),
    )


def _seqs(pkts):
    return [p.time.seq for p in pkts]


# ── policy resolution + config ───────────────────────────────────────────────

class TestPolicyResolution:
    def test_auto_is_lossless_under_replay_and_latest_live(self):
        assert resolve_policy("auto", replay=True) == "lossless"
        assert resolve_policy("auto", replay=False) == "latest"
        assert resolve_policy(None, replay=False) == "latest"

    def test_explicit_values_pass_through(self):
        assert resolve_policy("latest", replay=True) == "latest"
        assert resolve_policy("legacy", replay=False) == "legacy"
        assert resolve_policy("LOSSLESS", replay=True) == "lossless"

    def test_lossless_is_refused_for_live_receivers(self):
        with pytest.raises(ValueError, match="replay/eval-only"):
            resolve_policy("lossless", replay=False)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="ingest.policy must be one of"):
            resolve_policy("newest", replay=True)

    def test_lane_config_validates_every_key(self):
        ok = LaneConfig.from_cfg({"ingest": {"policy": "latest", "keyframe_lane_depth": 5, "max_frame_age_s": None}}, replay=False)
        assert (ok.policy, ok.keyframe_lane_depth, ok.max_frame_age_s, ok.lossless_depth) == ("latest", 5, None, 32)
        with pytest.raises(ValueError, match="keyframe_lane_overflow"):
            LaneConfig.from_cfg({"ingest": {"keyframe_lane_overflow": "foo"}}, replay=True)
        with pytest.raises(ValueError, match="keyframe_lane_depth"):
            LaneConfig.from_cfg({"ingest": {"keyframe_lane_depth": 0}}, replay=True)
        with pytest.raises(ValueError, match="max_frame_age_s"):
            LaneConfig.from_cfg({"ingest": {"max_frame_age_s": -1}}, replay=True)
        with pytest.raises(ValueError, match="whole number"):
            LaneConfig.from_cfg({"ingest": {"keyframe_lane_depth": 2.9}}, replay=True)
        with pytest.raises(ValueError, match="positive integer"):
            LaneConfig.from_cfg({"ingest": {"lossless_depth": True}}, replay=True)
        with pytest.raises(ValueError, match="max_frame_age_s must be > 0"):
            IngestLanes("latest", max_frame_age_s=0)                       # one validation path, not a silent off
        assert LaneConfig.from_cfg({}, replay=True).policy == "lossless"   # empty cfg -> defaults

    def test_lane_config_carries_the_receiver_timing_and_the_pairing_window(self):
        """P1 task 6: keyframe_every_n / nonkf_min_interval_s moved here from
        io.websocket.*; pair_window_s / pair_window_fps size the ZeroMQ window."""
        dflt = LaneConfig.from_cfg({}, replay=True)
        assert (dflt.keyframe_every_n, dflt.nonkf_min_interval_s, dflt.pair_window_s, dflt.pair_window_fps,
                dflt.pair_window_frames) == (30, 0.5, 2.0, 30.0, 90)
        assert (dflt.non_kf_grace_s, dflt.dup_window_ns) == (0.0, 200_000_000)
        timed = LaneConfig.from_cfg({"ingest": {"keyframe_every_n": 5, "nonkf_min_interval_s": 0,
                                                 "pair_window_s": 4.48, "pair_window_fps": 25}}, replay=True)
        assert (timed.keyframe_every_n, timed.nonkf_min_interval_s, timed.pair_window_frames) == (5, 0.0, 168)   # not 169
        assert LaneConfig(policy="latest", pair_window_s=1.0, pair_window_fps=10).pair_window_frames == 15   # a property
        for bad, key in (({"keyframe_every_n": 0}, "keyframe_every_n"),
                         ({"keyframe_every_n": 2.5}, "whole number"),
                         ({"nonkf_min_interval_s": -1}, "nonkf_min_interval_s"),
                         ({"nonkf_min_interval_s": True}, "nonkf_min_interval_s"),
                         ({"pair_window_s": 0}, "pair_window_s"),
                         ({"pair_window_fps": 0}, "pair_window_fps"),
                         ({"pair_window_fps": "fast"}, "pair_window_fps"),
                         ({"pair_window_s": "2"}, "pair_window_s"),          # strings are not numbers (YAML 1e3 rule)
                         ({"non_kf_grace_s": -1}, "non_kf_grace_s"),       # negative is rejected, not "off"
                         ({"non_kf_grace_s": True}, "non_kf_grace_s"),     # a YAML true would have been a 1 s grace
                         ({"dup_window_ns": -5}, "dup_window_ns"),
                         ({"dup_window_ns": 1.5}, "dup_window_ns")):
            with pytest.raises(ValueError, match=key):
                LaneConfig.from_cfg({"ingest": bad}, replay=True)
        with pytest.raises(ValueError, match="ingest: must be a mapping"):
            LaneConfig.from_cfg({"ingest": "fast"}, replay=True)

    def test_packaged_bases_resolve_through_lane_config(self):
        from rtsm.cfg import load_config
        main_cfg = LaneConfig.from_cfg(load_config("rtsm.yaml"), replay=True)
        demo_cfg = LaneConfig.from_cfg(load_config("demo_config.yaml"), replay=True)
        assert (main_cfg.keyframe_every_n, main_cfg.nonkf_min_interval_s) == (30, 0.5)
        assert (demo_cfg.keyframe_every_n, demo_cfg.nonkf_min_interval_s) == (5, 0.3)   # the demo's 5 / 0.3 live in the yaml now

    def test_zeromq_takes_the_throttle_and_window_from_kwargs(self):
        from rtsm.io.zeromq import ZeroMQSubscriber
        sub = ZeroMQSubscriber(ingest_queue=IngestQueue(4), nonkf_min_interval_s=1.25,
                               frame_window_ttl_s=3.0, frame_window_max_items=135)
        assert sub._nonkf_min_interval_s == 1.25
        assert (sub.fw.max, sub.fw.ttl_ns) == (135, 3_000_000_000)
        assert ZeroMQSubscriber(ingest_queue=IngestQueue(4))._nonkf_min_interval_s == 0.5   # default unchanged
        # ...and the kwarg is what _nonkf_due enforces (sensor mode: pose stamps; wall mode: process time)
        sensor = ZeroMQSubscriber(ingest_queue=IngestQueue(4), nonkf_min_interval_s=1.25, throttle_clock="sensor")
        sensor._stamp_nonkf(10_000_000_000)
        assert sensor._nonkf_due(10_900_000_000) is False and sensor._nonkf_due(11_300_000_000) is True
        sub._stamp_nonkf(None)
        assert sub._nonkf_due(None) is False                              # 1.25 s have not passed on the wall clock

    def test_factory_builds_the_right_object(self):
        legacy = make_ingest_queue(LaneConfig.from_cfg({"ingest": {"policy": "legacy"}}, replay=False))
        assert isinstance(legacy, IngestQueue) and legacy.policy == "legacy" and legacy.maxsize == 512
        latest = make_ingest_queue(LaneConfig.from_cfg({}, replay=False))
        assert isinstance(latest, IngestLanes) and latest.policy == "latest" and latest.maxsize == 4
        lossless = make_ingest_queue(LaneConfig.from_cfg({"ingest": {"lossless_depth": 7}}, replay=True))
        assert lossless.policy == "lossless" and lossless.maxsize == 7


# ── policy latest ────────────────────────────────────────────────────────────

class TestLatest:
    def _lanes(self, clock: Clock, **kw):
        drops = []
        lanes = IngestLanes("latest", on_drop=lambda p, r: drops.append((p.time.seq, r)), now_fn=clock, **kw)
        return lanes, drops

    def test_keyframes_first_then_the_slot_and_supersession(self):
        clock = Clock(); lanes, drops = self._lanes(clock)
        assert lanes.put(_pkt(1)) and lanes.put(_pkt(2, kf=True)) and lanes.put(_pkt(3))
        assert lanes.qsize() == 2 and lanes.depth() == {LANE_KEYFRAME: 1, LANE_LATEST: 1}
        assert drops == [(1, DROP_SUPERSEDED)]                    # frame 1 was waiting; frame 3 replaced it
        assert _seqs([lanes.get(), lanes.get()]) == [2, 3] and lanes.get() is None
        s = lanes.stats()
        assert (s["admitted_kf"], s["admitted_nonkf"], s["nonkf_superseded"], s["max_depth_seen"]) == (1, 2, 1, 2)

    def test_ingest_meta_is_filled_at_admission(self):
        clock = Clock(1234.5); lanes, _ = self._lanes(clock)
        k1, k2, n1, n2 = _pkt(1, kf=True), _pkt(2, kf=True), _pkt(3), _pkt(4)
        for p in (k1, k2, n1, n2):
            lanes.put(p)
        assert (k1.ingest.lane, k1.ingest.lane_depth_at_admit, k1.ingest.admitted_mono, k1.ingest.admitted_sensor_ns) == (LANE_KEYFRAME, 0, 1234.5, 1000)
        assert (k2.ingest.lane, k2.ingest.lane_depth_at_admit) == (LANE_KEYFRAME, 1)
        assert (n1.ingest.lane, n1.ingest.lane_depth_at_admit, n1.ingest.drop_reason) == (LANE_LATEST, 0, DROP_SUPERSEDED)
        assert (n2.ingest.lane, n2.ingest.lane_depth_at_admit, n2.ingest.drop_reason) == (LANE_LATEST, 1, None)
        assert k1.ingest.keyframe_origin == KF_MINTED and k1.ingest.rx_seq == 1

    def test_refusal_only_for_a_source_keyframe_under_reject(self):
        clock = Clock()
        lanes_do, _ = self._lanes(clock)                                   # drop_oldest (default)
        lanes_rj, _ = self._lanes(clock, keyframe_lane_overflow="reject")
        for lanes in (lanes_do, lanes_rj):
            for i in range(3):
                assert lanes.put(_pkt(i, kf=True, origin=KF_SOURCE))
            assert lanes.refusal(False, None) is None                     # a non-keyframe always has the slot
            assert lanes.refusal(True, KF_MINTED) is None                 # minted keyframes drop the oldest
        assert lanes_do.refusal(True, KF_SOURCE) is None
        assert lanes_rj.refusal(True, KF_SOURCE) == DROP_KF_LANE_FULL
        assert lanes_do.full() is False and lanes_rj.full() is False

    def test_keyframe_overflow_drops_the_oldest_so_the_freshest_wins(self):
        clock = Clock(); lanes, drops = self._lanes(clock)
        for i in range(1, 5):
            assert lanes.put(_pkt(i, kf=True, origin=(KF_SOURCE if i % 2 else KF_MINTED)))
        assert drops == [(1, DROP_KF_DROPPED)]
        assert lanes.depth()[LANE_KEYFRAME] == 3
        got = [lanes.get(), lanes.get(), lanes.get()]
        assert _seqs(got) == [2, 3, 4]
        assert all(p.is_keyframe for p in got)                            # never demoted, whatever the origin
        assert lanes.stats()["kf_dropped"] == 1

    def test_reject_refuses_a_source_keyframe_but_minted_still_drops_oldest(self):
        clock = Clock(); lanes, drops = self._lanes(clock, keyframe_lane_overflow="reject")
        for i in range(1, 4):
            lanes.put(_pkt(i, kf=True, origin=KF_SOURCE))
        p4 = _pkt(4, kf=True, origin=KF_SOURCE)
        assert lanes.put(p4) is False and p4.ingest.drop_reason == DROP_KF_LANE_FULL
        assert drops == [] and lanes.stats()["kf_lane_full"] == 1          # refusals are the caller's to trace
        assert lanes.put(_pkt(5, kf=True, origin=KF_MINTED)) is True
        assert drops == [(1, DROP_KF_DROPPED)]
        assert _seqs([lanes.get(), lanes.get(), lanes.get()]) == [2, 3, 5]

    def test_age_drop_applies_to_the_slot_only(self):
        clock = Clock(); lanes, drops = self._lanes(clock, max_frame_age_s=2.0)
        lanes.put(_pkt(1, kf=True)); lanes.put(_pkt(2))
        clock.tick(2.5)
        assert lanes.get().time.seq == 1                                  # a 2.5 s-old keyframe is still delivered
        assert lanes.get() is None                                        # the 2.5 s-old non-keyframe is discarded
        assert drops == [(2, DROP_AGE)] and lanes.stats()["age_dropped"] == 1
        lanes.put(_pkt(3)); clock.tick(1.9)
        assert lanes.get().time.seq == 3                                  # inside the bound -> delivered
        no_age, _ = self._lanes(clock, max_frame_age_s=None)
        no_age.put(_pkt(4)); clock.tick(60.0)
        assert no_age.get().time.seq == 4                                 # null disables the age check

    def test_wedge_shape_30hz_producer_vs_1hz_consumer_is_bounded_and_kf_first(self):
        """The E1 wedge shape: a 30 Hz phone whose receiver mints a keyframe
        every 30 tracking-normal frames (1 Hz) and admits ~2 non-keyframes/s
        after the 0.5 s throttle, against a 1 s consumer. Total depth never
        exceeds keyframe_lane_depth + 1, keyframes are consumed before the
        slot, no keyframe is dropped or demoted, and the slot supersedes
        instead of queueing (documented consequence: nearly every processed
        frame is a keyframe)."""
        clock = Clock(); lanes, drops = self._lanes(clock)
        got, max_depth, seq = [], 0, 0
        for step in range(0, 6000):                                        # 60 s at 100 Hz simulation
            clock.tick(0.01)
            if step % 100 == 0:                                           # 1 Hz minted keyframe (frame_count % 30 at 30 Hz)
                seq += 1; lanes.put(_pkt(seq, kf=True))
            elif step % 50 == 25:                                         # ~2 Hz admitted non-keyframes (0.5 s throttle)
                seq += 1; lanes.put(_pkt(seq))
            max_depth = max(max_depth, lanes.qsize())
            if step % 100 == 99:                                          # 1 Hz consumer, the step before the next keyframe
                p = lanes.get()
                if p is not None:
                    got.append(p)
        while True:                                                       # drain what the last second left waiting
            p = lanes.get()
            if p is None:
                break
            got.append(p)
        assert max_depth <= 4
        assert all(p.is_keyframe for p in got if p.ingest.lane == LANE_KEYFRAME)
        assert lanes.stats()["kf_dropped"] == 0                           # 1 KF/s in, 1 frame/s out: lane never overflows
        assert len([p for p in got if p.is_keyframe]) == lanes.stats()["admitted_kf"]   # every keyframe processed
        assert sum(1 for p in got if not p.is_keyframe) <= 1              # the slot starves under a 1 Hz KF cadence (1 = the final drain)
        assert lanes.stats()["nonkf_superseded"] > 50 and lanes.stats()["age_dropped"] == 0

    def test_source_keyframe_flood_bounds_the_lane_and_never_demotes(self):
        """Source keyframes at 2 Hz plus non-keyframes at 30 Hz against a 1 Hz
        consumer: the keyframe lane stays bounded (oldest dropped and counted),
        every dequeued frame is a keyframe (the slot is best-effort under a
        keyframe flood -- documented, not a fault) and nothing is demoted."""
        clock = Clock(); lanes, drops = self._lanes(clock)
        got, seq, next_kf, next_consume = [], 0, 0.0, 1.0
        for step in range(3000):                                          # 30 s at 100 Hz
            t = clock.tick(0.01)
            if step % 3 == 0:
                seq += 1; lanes.put(_pkt(seq))                            # 33 Hz non-KF
            if t >= next_kf:
                seq += 1; lanes.put(_pkt(seq, kf=True, origin=KF_SOURCE)); next_kf += 0.5
            assert lanes.qsize() <= 4
            if t >= next_consume:
                p = lanes.get()
                if p is not None:
                    got.append(p)
                next_consume += 1.0
        assert got and all(p.is_keyframe for p in got)
        assert lanes.stats()["kf_dropped"] > 0 and lanes.stats()["kf_lane_full"] == 0
        assert all(r in (DROP_KF_DROPPED, DROP_SUPERSEDED) for _, r in drops)

    def test_backlog_signal_ignores_supersession(self):
        clock = Clock(); lanes, _ = self._lanes(clock)
        for i in range(5):
            lanes.put(_pkt(i))                                            # 4 supersessions
        assert lanes.backlog_signal()["lane_full"] is False
        for i in range(3):
            lanes.put(_pkt(10 + i, kf=True))
        sig = lanes.backlog_signal()
        assert sig["lane_full"] is True and sig["age_dropped"] == 0 and sig["depth"] == {LANE_KEYFRAME: 3, LANE_LATEST: 1}

    def test_non_framepacket_sentinel_is_tolerated(self):
        lanes = IngestLanes("latest")
        assert lanes.put(object()) is True and lanes.qsize() == 1          # treated as a non-keyframe; no meta


# ── policy lossless ──────────────────────────────────────────────────────────

class TestLossless:
    def test_put_blocks_the_producer_even_with_block_false_and_keeps_order(self):
        lanes = IngestLanes("lossless", lossless_depth=2)
        assert lanes.put(_pkt(1), block=False) and lanes.put(_pkt(2), block=False)
        assert lanes.full() is True and lanes.refusal(True, KF_SOURCE) is None
        result = {}

        def producer():
            result["ok"] = lanes.put(_pkt(3), block=False)               # block=False is ignored: it waits
        th = threading.Thread(target=producer, daemon=True); th.start()
        th.join(0.3)
        assert th.is_alive() and "ok" not in result                       # still waiting
        assert lanes.get().time.seq == 1                                   # frees one slot
        th.join(2.0)
        assert result.get("ok") is True and not th.is_alive()
        assert _seqs([lanes.get(), lanes.get()]) == [2, 3] and lanes.get() is None
        s = lanes.stats()
        assert s["blocked_puts"] == 1 and s["blocked_s"] > 0 and s["lossless_depth"] == 2

    def test_close_wakes_a_blocked_producer_and_get_still_drains(self):
        lanes = IngestLanes("lossless", lossless_depth=1)
        lanes.put(_pkt(1))
        p2 = _pkt(2); result = {}
        th = threading.Thread(target=lambda: result.__setitem__("ok", lanes.put(p2)), daemon=True); th.start()
        th.join(0.2); assert th.is_alive()
        t0 = time.monotonic(); lanes.close(); th.join(2.0)
        assert not th.is_alive() and (time.monotonic() - t0) < 1.0
        assert result["ok"] is False and p2.ingest.drop_reason == DROP_CLOSED
        assert lanes.refusal(False, None) == DROP_CLOSED
        assert lanes.stats()["blocked_puts"] == 1 and lanes.stats()["blocked_s"] > 0   # the aborted wait is accounted
        assert lanes.get().time.seq == 1                                   # draining continues after close
        assert lanes.stats()["closed"] is True and lanes.stats()["closed_puts"] == 1

    def test_timed_get_returns_with_a_frozen_injected_clock(self):
        lanes = IngestLanes("lossless", lossless_depth=2, now_fn=Clock())   # clock never advances
        t0 = time.monotonic()
        assert lanes.get(timeout=0.2) is None
        assert 0.15 < time.monotonic() - t0 < 2.0                          # real-time deadline, no spin

    def test_no_age_drop_and_fifo_meta(self):
        clock = Clock(); lanes = IngestLanes("lossless", lossless_depth=8, now_fn=clock)
        n = _pkt(1); lanes.put(n); clock.tick(100.0)
        assert lanes.get() is n and n.ingest.lane == LANE_FIFO and n.ingest.lane_depth_at_admit == 0
        assert lanes.backlog_signal() == {"lane_full": False, "age_dropped": 0, "depth": {LANE_FIFO: 0}}


# ── legacy queue: untouched behaviour, added surface ─────────────────────────

class TestLegacyQueue:
    def test_surface_and_unchanged_semantics(self):
        q = IngestQueue(maxsize=2)
        assert q.policy == "legacy" and q.refusal(True, KF_SOURCE) is None
        assert q.put(object()) and q.put(object()) and q.put(object()) is False   # tail-drop, sentinels fine
        assert q.refusal(False, None) == "queue_full" and q.full() and q.qsize() == 2
        assert q.backlog_signal()["lane_full"] is True and q.stats()["policy"] == "legacy"
        q.set_on_drop(lambda *a: None); q.close()                          # no-ops
        assert q.get() is not None and q.get() is not None and q.get() is None


# ── the runner's drop handler ────────────────────────────────────────────────

class _Analytics:
    def __init__(self):
        self.queue_drops = 0; self.age_drops = 0; self.superseded = 0

    def record_queue_drop(self):
        self.queue_drops += 1

    def record_age_drop(self):
        self.age_drops += 1

    def record_superseded(self):
        self.superseded += 1


class TestLaneDropHandler:
    def test_writes_a_lanes_line_and_counts(self):
        clock = Clock(); events, an = [], _Analytics()
        lanes = IngestLanes("latest", now_fn=clock, max_frame_age_s=1.0)
        lanes.set_on_drop(lane_drop_handler(events.append, an, lanes))
        lanes.put(_pkt(7, 7_000)); lanes.put(_pkt(8, 8_000))              # 7 superseded
        e = events[-1]
        assert (e.kind, e.source, e.decision, e.reason) == ("receiver", "lanes", "dropped", DROP_SUPERSEDED)
        assert (e.frame_seq, e.t_sensor_ns, e.is_keyframe, e.lane, e.rx_seq, e.queue_depth) == (7, 7_000, False, LANE_LATEST, 7, 1)
        clock.tick(5.0); assert lanes.get() is None                        # 8 aged out
        assert events[-1].reason == DROP_AGE
        assert (an.queue_drops, an.superseded, an.age_drops) == (0, 1, 1)   # supersession is not a queue drop
        for i in range(4):
            lanes.put(_pkt(20 + i, kf=True))                                # 4th keyframe drops the oldest
        assert (an.queue_drops, an.superseded, an.age_drops) == (1, 1, 1)

    def test_without_a_sink_it_still_counts(self):
        an = _Analytics(); lanes = IngestLanes("latest")
        lanes.set_on_drop(lane_drop_handler(None, an, lanes))
        lanes.put(_pkt(1)); lanes.put(_pkt(2))
        assert (an.superseded, an.queue_drops) == (1, 0)

    def test_real_analytics_counters_and_reset(self):
        from rtsm.analytics.latency_analytics import PipelineLatencyBuffer
        an = PipelineLatencyBuffer()
        lanes = IngestLanes("latest", max_frame_age_s=0.5, now_fn=(clock := Clock()))
        lanes.set_on_drop(lane_drop_handler(None, an, lanes))
        lanes.put(_pkt(1)); lanes.put(_pkt(2)); clock.tick(1.0); assert lanes.get() is None
        for i in range(4):
            lanes.put(_pkt(10 + i, kf=True))
        c = an.aggregate()["counters"]
        assert (c["queue_drops"], c["superseded"], c["age_drops"]) == (1, 1, 1)
        an.clear()
        c = an.aggregate()["counters"]
        assert (c["queue_drops"], c["superseded"], c["age_drops"]) == (0, 0, 0)   # /reset zeroes the new counters too


# ── watchdog: sustained backlogged ───────────────────────────────────────────

class _Recv:
    def liveness(self):
        return {"alive": True, "last_rx_mono": 999.9, "last_enqueue_mono": 999.9, "tracking_drops": 0}


def _monitor(clock, hb, signal):
    return FrameFlowMonitor(heartbeat=hb, queue_size=lambda: 3, receiver_liveness=_Recv().liveness,
                            now_fn=clock, backlog=lambda: signal, backlog_polls=3)


class TestWatchdogBacklogged:
    def test_needs_three_consecutive_full_polls_after_warmup(self):
        clock = Clock(); hb = PipelineHeartbeat(); hb.last_step_mono = 999.9
        sig = {"lane_full": True, "age_dropped": 0}
        m = _monitor(clock, hb, sig)
        assert [m.evaluate()["state"] for _ in range(4)] == ["ok"] * 4      # no frame processed yet: warm-up
        hb.last_frame_mono = 999.9
        states = [m.evaluate()["state"] for _ in range(3)]
        assert states[-1] == "backlogged" and "backlogged" in DEGRADED_STATES
        sig["lane_full"] = False
        assert m.evaluate()["state"] == "ok"                               # streak resets

    def test_age_drop_since_last_poll_flags_it_once(self):
        clock = Clock(); hb = PipelineHeartbeat(); hb.last_step_mono = 999.9; hb.last_frame_mono = 999.9
        sig = {"lane_full": False, "age_dropped": 0}
        m = _monitor(clock, hb, sig)
        assert m.evaluate()["state"] == "ok"
        sig["age_dropped"] = 2
        r = m.evaluate()
        assert r["state"] == "backlogged" and r["degraded"] is True and r["backlog"]["age_dropped"] == 2
        assert m.evaluate()["state"] == "ok"                               # no new drops -> clears

    def test_supersession_alone_is_never_a_backlog(self):
        clock = Clock(); hb = PipelineHeartbeat(); hb.last_step_mono = 999.9; hb.last_frame_mono = 999.9
        m = _monitor(clock, hb, {"lane_full": False, "age_dropped": 0, "nonkf_superseded": 500})
        assert all(m.evaluate()["state"] == "ok" for _ in range(5))

    def test_no_backlog_source_keeps_the_old_behaviour(self):
        clock = Clock(); hb = PipelineHeartbeat(); hb.last_step_mono = 999.9; hb.last_frame_mono = 999.9
        m = FrameFlowMonitor(heartbeat=hb, queue_size=lambda: 0, receiver_liveness=_Recv().liveness, now_fn=clock)
        r = m.evaluate()
        assert r["state"] == "ok" and r["backlog"] is None


# ── shutdown wiring + config echo ────────────────────────────────────────────

class TestShutdownWiring:
    def test_pipeline_shutdown_closes_the_queue(self):
        from rtsm.core.pipeline import Pipeline

        class _Sweep:
            def cell_and_vbin_from_pose(self, **kw):
                return (0, 0, 0), 0, np.array([0.0, 0.0, 1.0], dtype=np.float32)

        lanes = IngestLanes("lossless", lossless_depth=1)
        pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                        associator=None, ingest_gate=None, ingest_q=lanes, sweep_cache=_Sweep())
        pipe.shutdown()
        assert lanes.closed is True
        p = _pkt(1)
        assert lanes.put(p) is False and p.ingest.drop_reason == DROP_CLOSED

    def test_replay_stop_wakes_a_blocked_lossless_producer(self, tmp_path):
        from rtsm.io.recorder import SessionRecorder
        from rtsm.io.replayer import ReplayReceiver
        rec_dir = tmp_path / "rec"; rec = SessionRecorder(output_dir=str(rec_dir))
        for i in range(1, 5):
            rec.on_message("binary", _ws_frame(i, i * 1000)); time.sleep(0.01)
        rec.on_handshake({"type": "hello", "session_id": "s1"}, {"type": "hello_ack", "status": "ok"}); rec.close()
        events = []
        lanes = IngestLanes("lossless", lossless_depth=1)                  # nobody drains: the 2nd put blocks
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=lanes, keyframe_every_n=30,
                            nonkf_min_interval_s=0.0, replay_speed=10.0, event_sink=events.append)
        rr.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and lanes.stats()["blocked_puts"] == 0 and not rr.wait(0.0):
            time.sleep(0.02)
        assert lanes.qsize() == 1 and not rr.wait(0.0)                    # blocked with one frame waiting
        t0 = time.monotonic(); rr.stop()
        assert rr.wait(2.0) and (time.monotonic() - t0) < 1.5              # woke up and finished promptly
        assert lanes.closed and ("dropped", DROP_CLOSED) in [(e.decision, e.reason) for e in events]

    def test_stop_leaves_a_non_blocking_queue_open(self, tmp_path):
        from rtsm.io.recorder import SessionRecorder
        from rtsm.io.replayer import ReplayReceiver
        rec_dir = tmp_path / "rec"; rec = SessionRecorder(output_dir=str(rec_dir))
        rec.on_message("binary", _ws_frame(1, 1000)); rec.close()
        lanes = IngestLanes("latest")
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=lanes, keyframe_every_n=30,
                            nonkf_min_interval_s=0.0)
        rr.stop()
        assert lanes.closed is False                                       # only lossless has a producer to wake


class TestConfigEcho:
    def _echo(self, q):
        from rtsm.visualization.server import VisualizationServer
        vs = VisualizationServer(cfg={}, working_memory=None, ingest_queue=q)
        return vs._extract_analytics_config()["receiver"]

    def test_queue_capacity_and_policy_come_from_the_object(self):
        rx = self._echo(IngestLanes("latest", keyframe_lane_depth=3))
        assert (rx["queue_maxsize"], rx["ingest_policy"]) == (4, "latest")
        rx = self._echo(IngestLanes("lossless", lossless_depth=7))
        assert (rx["queue_maxsize"], rx["ingest_policy"]) == (7, "lossless")
        rx = self._echo(None)
        assert (rx["queue_maxsize"], rx["ingest_policy"]) == (None, "legacy")   # nothing wired: no invented 512
        rx = self._echo(IngestQueue(maxsize=512))
        assert (rx["queue_maxsize"], rx["ingest_policy"]) == (512, "legacy")

    def test_receiver_timing_is_echoed_from_the_ingest_block(self):
        from rtsm.visualization.server import VisualizationServer
        cfg = {"ingest": {"keyframe_every_n": 7, "nonkf_min_interval_s": 0.25},
               "io": {"websocket": {"keyframe_every_n": 99, "nonkf_min_interval_s": 9.9}}}   # a stale old block is ignored
        rx = VisualizationServer(cfg=cfg, working_memory=None, ingest_queue=None)._extract_analytics_config()["receiver"]
        assert (rx["keyframe_every_n"], rx["nonkf_min_interval_s"]) == (7, 0.25)


# ── receivers call the lanes before decoding ─────────────────────────────────

def _ws_frame(frame_id: int, timestamp_ns: int) -> bytes:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8); _, jpeg = cv2.imencode(".jpg", rgb)
    depth = np.ones((8, 8), dtype=np.uint16) * 1500
    header = {"frame_id": frame_id, "timestamp_ns": timestamp_ns, "unix_timestamp": 1700000000.0,
              "rgb_format": "jpeg", "rgb_width": 8, "rgb_height": 8,
              "depth_format": "uint16_mm", "depth_width": 8, "depth_height": 8, "depth_scale": 0.001,
              "fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0,
              "pose_format": "quat_translation", "T_wc": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "tracking_state": "normal"}
    hj = json.dumps(header).encode(); rj = jpeg.tobytes(); dj = depth.tobytes()
    return struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj + struct.pack("<I", len(dj)) + dj


class TestWebSocketWithLanes:
    def test_enqueued_lines_carry_lane_and_rx_seq_and_lane_drops_follow(self):
        from rtsm.io.websocket import WebSocketReceiver
        events = []
        lanes = IngestLanes("latest")
        lanes.set_on_drop(lane_drop_handler(events.append, None, lanes))
        recv = WebSocketReceiver(ingest_queue=lanes, keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 event_sink=events.append)
        for i in (1, 2, 3):
            pkt = recv._parse_binary_message(_ws_frame(i, i * 1_000))
            assert pkt is not None and pkt.ingest.rx_seq == i and pkt.ingest.keyframe_origin == (KF_MINTED if i == 1 else None)
            assert lanes.put(pkt) is True
            recv._trace_rx("enqueued", "", pkt=pkt)                        # what the stream loop does after put()
        rx = [(e.source, e.decision, e.reason, e.frame_seq, e.lane, e.rx_seq) for e in events]
        assert rx == [("websocket", "enqueued", "", 1, LANE_KEYFRAME, 1),
                      ("websocket", "enqueued", "", 2, LANE_LATEST, 2),
                      ("lanes", "dropped", DROP_SUPERSEDED, 2, LANE_LATEST, 2),
                      ("websocket", "enqueued", "", 3, LANE_LATEST, 3)]
        assert all(e.depth_valid_frac == pytest.approx(1.0) for e in events)   # one statistic, incl. the lanes line

    def test_legacy_full_queue_still_refuses_before_decode(self, monkeypatch):
        from rtsm.io import websocket as ws_mod
        from rtsm.io.websocket import WebSocketReceiver
        calls = {"n": 0}; real = ws_mod.decode_rgb
        monkeypatch.setattr(ws_mod, "decode_rgb", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1])
        q = IngestQueue(maxsize=1); q.put(object())
        events = []
        recv = WebSocketReceiver(ingest_queue=q, keyframe_every_n=1000, nonkf_min_interval_s=0.0, event_sink=events.append)
        assert recv._parse_binary_message(_ws_frame(1, 1_000)) is None
        assert calls["n"] == 0 and (events[-1].reason, events[-1].rx_seq) == ("queue_full", 1)


class TestZeroMQWithLanes:
    def _camera_msg(self, ts_ns: int):
        rgb = np.full((8, 8, 3), 90, dtype=np.uint8); _, jpg = cv2.imencode(".jpg", rgb)
        _, png = cv2.imencode(".png", np.ones((8, 8), dtype=np.uint16) * 1234)
        meta = {"ts_ns": ts_ns, "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0, "width": 8, "height": 8},
                "depth_units_m": 0.001}
        return [b"camera.rgbd", json.dumps(meta).encode(), jpg.tobytes(), png.tobytes()]

    def test_source_keyframe_refused_before_decode_under_reject(self, monkeypatch):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber
        lanes = IngestLanes("latest", keyframe_lane_overflow="reject")
        events = []
        sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                               ingest_queue=lanes, event_sink=events.append)
        try:
            t = np.zeros(3, dtype=np.float32); qq = np.array([0, 0, 0, 1], dtype=np.float32)
            for i in range(1, 4):
                ts = 10_000_000_000 + i * 100_000_000
                sub._handle_camera_rgbd(self._camera_msg(ts)); sub._try_enqueue_frame(ts, t, qq, is_keyframe=True)
            assert lanes.depth()[LANE_KEYFRAME] == 3
            pkt = lanes.get(); assert pkt.ingest.keyframe_origin == KF_SOURCE and pkt.ingest.rx_seq == 1
            lanes.put(pkt)                                                 # lane full again (3), nobody draining
            ts4 = 10_000_000_000 + 4 * 100_000_000
            sub._handle_camera_rgbd(self._camera_msg(ts4))
            calls = {"n": 0}
            monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or None)
            sub._try_enqueue_frame(ts4, t, qq, is_keyframe=True)
            assert calls["n"] == 0                                         # refused before JPEG/PNG decode
            assert (events[-1].decision, events[-1].reason, events[-1].rx_seq) == ("dropped", DROP_KF_LANE_FULL, 4)
            assert [e.lane for e in events if e.decision == "enqueued"] == [LANE_KEYFRAME] * 3
        finally:
            sub.close()
