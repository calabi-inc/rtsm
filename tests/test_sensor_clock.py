"""Ingest clock injection (Gate 4.5 plan, P1 task 1): SensorClock semantics, the
working-memory and pipeline injection points, and the receivers' sensor-time
non-keyframe throttle with stamp-on-admit. CPU-only, no models.
"""
from __future__ import annotations

import json
import struct
import time

import cv2
import numpy as np
import pytest

from rtsm.core.clock import SensorClock, WallClock, make_clock, resolve_clock_mode
from rtsm.core.datamodel import FramePacket, PoseStamped, TimeBundle
from rtsm.core.ingest_gate import IngestDecision
from rtsm.core.pipeline import Pipeline
from rtsm.evaluation.event_log import EventLogWriter
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver
from rtsm.stores.working_memory import WorkingMemory


class FakeClock:
    """Manual clock for the injection tests."""

    def __init__(self, t: float = 1000.0):
        self.t = t
        self.advanced = []

    def now_mono(self) -> float:
        return self.t

    def advance(self, t_sensor_ns, frame_epoch=None):
        self.advanced.append((t_sensor_ns, frame_epoch))
        return self.t


# ───────────────────────────── SensorClock ─────────────────────────────

class TestSensorClock:
    def test_first_frame_anchors_to_wall_magnitude(self):
        c = SensorClock(fallback=lambda: 500.0)
        assert c.now_mono() == 500.0                      # before any frame: wall fallback
        assert c.advance(10_000_000_000) == 500.0         # anchored: sensor 10 s == mono 500 s
        assert c.advance(12_500_000_000) == pytest.approx(502.5)
        assert c.now_mono() == pytest.approx(502.5)

    # In the tests below the fallback equals the first frame's sensor seconds, so
    # the anchor maps sensor time onto itself and the numbers read directly.

    def test_backwards_stamps_clamp_within_an_epoch(self):
        c = SensorClock(fallback=lambda: 5.0)
        assert c.advance(5_000_000_000, frame_epoch=0) == pytest.approx(5.0)
        assert c.advance(4_000_000_000, frame_epoch=0) == pytest.approx(5.0)   # clamped, not moved back
        assert c.clamped == 1
        assert c.advance(6_000_000_000, frame_epoch=0) == pytest.approx(6.0)

    def test_epoch_change_rebases_continuously(self):
        c = SensorClock(fallback=lambda: 100.0)
        c.advance(100_000_000_000, frame_epoch=0)         # sensor t=100 s -> now 100
        c.advance(101_000_000_000, frame_epoch=0)         # now 101
        # ARKit restart: sensor time jumps back to 3 s under a new epoch
        assert c.advance(3_000_000_000, frame_epoch=1) == pytest.approx(101.0)   # continuous
        assert c.rebases == 1 and c.epoch == 1
        assert c.advance(4_500_000_000, frame_epoch=1) == pytest.approx(102.5)   # runs on from there
        # neither frozen (clamp semantics) nor jumped back by 98 s

    def test_missing_stamp_leaves_clock_in_place(self):
        c = SensorClock(fallback=lambda: 2.0)
        c.advance(2_000_000_000)
        assert c.advance(None) == pytest.approx(2.0)
        assert c.advance(0) == pytest.approx(2.0)

    def test_none_epoch_small_reversal_clamps(self):
        """Without epoch information a small reversal (jitter, below
        rebase_after_s) clamps; a large one re-bases (see TestSensorClockRecovery)."""
        c = SensorClock(fallback=lambda: 50.0)
        c.advance(50_000_000_000, frame_epoch=None)
        assert c.advance(48_000_000_000, frame_epoch=None) == pytest.approx(50.0)  # clamp path
        assert c.rebases == 0 and c.clamped == 1

    def test_wall_clock_advance_is_a_noop(self):
        w = WallClock()
        t0 = w.now_mono()
        w.advance(123)
        assert w.now_mono() >= t0

    def test_factories(self):
        assert isinstance(make_clock("wall"), WallClock)
        assert isinstance(make_clock("sensor"), SensorClock)
        assert resolve_clock_mode("auto", replay=True) == "sensor"
        assert resolve_clock_mode("auto", replay=False) == "wall"
        assert resolve_clock_mode("wall", replay=True) == "wall"
        assert resolve_clock_mode(None, replay=False) == "wall"
        with pytest.raises(ValueError):
            resolve_clock_mode("bogus", replay=True)


# ───────────────────────────── WorkingMemory ─────────────────────────────

def _obs_args():
    return dict(p_world=np.array([1.0, 0.5, 2.0], dtype=np.float32),
                emb_vis=np.ones(8, dtype=np.float32) / np.sqrt(8.0),
                label_topk=[("mug", 0.9)])


class TestWorkingMemoryClock:
    def test_default_is_wall_and_explicit_clock_is_used_for_stamps(self):
        assert isinstance(WorkingMemory(cfg={})._clock, WallClock)
        clk = FakeClock(t=1000.0)
        wm = WorkingMemory(cfg={"object": {"proto_ttl_s": 10.0}}, clock=clk)
        oid = wm.create_object(**_obs_args())
        assert oid is not None
        o = wm._map[oid]
        assert o.last_seen_mono == pytest.approx(1000.0)

    def test_proto_expiry_follows_the_injected_clock_not_wall_time(self):
        clk = FakeClock(t=1000.0)
        wm = WorkingMemory(cfg={"object": {"proto_ttl_s": 10.0}}, clock=clk)
        oid = wm.create_object(**_obs_args())
        assert wm.expire_timeouts() == []                  # no time passed on the ingest clock
        clk.t = 1009.0
        assert wm.expire_timeouts() == []                  # 9 s < 10 s TTL
        clk.t = 1010.5
        assert wm.expire_timeouts() == [oid]               # expired purely by the injected clock
        assert oid not in wm._map

    def test_update_dt_uses_the_injected_clock(self):
        clk = FakeClock(t=2000.0)
        wm = WorkingMemory(cfg={}, clock=clk)
        oid = wm.create_object(**_obs_args())
        clk.t = 2003.0

        class _Obs:
            p_world = np.array([1.0, 0.5, 2.0], dtype=np.float32)
            xyz_world = p_world
            emb_vis = np.ones(8, dtype=np.float32) / np.sqrt(8.0)
            label_topk = [("mug", 0.9)]
            depth_valid = 1.0
            quality = 1.0
            is_keyframe = False
            view_dir_cam = None
            centroid_px = None
            crop = None
            frame_id = None
            stats = None

        wm.update_object(oid, _Obs())
        assert wm._map[oid].last_seen_mono == pytest.approx(2003.0)


# ───────────────────────────── Pipeline dispatcher ─────────────────────────────

class _GateSpy:
    def __init__(self):
        self.now_seen = []

    def should_accept(self, **kw) -> IngestDecision:
        self.now_seen.append(kw.get("now_mono"))
        return IngestDecision(False, "spy_reject")       # stop before segmentation


class _SweepCacheStub:
    def cell_and_vbin_from_pose(self, *, twc_xyz, q_wc_xyzw):
        return (0, 0, 0), 0, np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _packet(ts_ns: int, epoch=None) -> FramePacket:
    return FramePacket(rgb=np.zeros((4, 4, 3), dtype=np.uint8), depth_m=np.ones((4, 4), dtype=np.float32),
                       pose=PoseStamped(stamp_ns=ts_ns, frame_id="arkit",
                                        t_wc=np.zeros(3, dtype=np.float32),
                                        q_wc_xyzw=np.array([0, 0, 0, 1], dtype=np.float32)),
                       intr=None, is_keyframe=False,
                       time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=ts_ns, seq=1),
                       frame_epoch=epoch)


class TestPipelineClock:
    def test_default_wall_clock(self):
        pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                        associator=None, ingest_gate=None, ingest_q=IngestQueue())
        assert isinstance(pipe.clock, WallClock)

    def test_dispatcher_advances_sensor_clock_and_gate_reads_it(self, tmp_path):
        clock = SensorClock(fallback=lambda: 7.0)     # anchor == first frame's sensor seconds
        gate = _GateSpy()
        q = IngestQueue()
        log = EventLogWriter(enabled=True, configured_path=str(tmp_path / "e.jsonl"))
        pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                        associator=None, ingest_gate=gate, ingest_q=q, sweep_cache=_SweepCacheStub(),
                        event_log=log, clock=clock)
        q.put(_packet(7_000_000_000, epoch=0))
        pipe.run_one_step()
        q.put(_packet(9_500_000_000, epoch=0))
        pipe.run_one_step()
        assert gate.now_seen == [pytest.approx(7.0), pytest.approx(9.5)]     # gate saw sensor time
        log.close()
        rows = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
        deq = [r for r in rows if r["kind"] == "dequeue"]
        assert [r["clock_s"] for r in deq] == [pytest.approx(7.0), pytest.approx(9.5)]

    def test_fake_clock_receives_frame_epoch(self):
        clk = FakeClock()
        gate = _GateSpy()
        q = IngestQueue()
        pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                        associator=None, ingest_gate=gate, ingest_q=q, sweep_cache=_SweepCacheStub(), clock=clk)
        q.put(_packet(1_000, epoch=3))
        pipe.run_one_step()
        assert clk.advanced == [(1_000, 3)]
        assert gate.now_seen == [1000.0]


# ───────────────────────────── receiver throttle ─────────────────────────────

def _ws_frame(frame_id: int, timestamp_ns: int, **overrides) -> bytes:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", rgb)
    depth = (np.ones((16, 16), dtype=np.uint16) * 1500).tobytes()
    header = {
        "frame_id": frame_id, "timestamp_ns": timestamp_ns, "unix_timestamp": 1700000000.0,
        "rgb_format": "jpeg", "rgb_width": 16, "rgb_height": 16,
        "depth_format": "uint16_mm", "depth_width": 16, "depth_height": 16, "depth_scale": 0.001,
        "fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0,
        "pose_format": "quat_translation", "T_wc": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "tracking_state": "normal",
    }
    header.update(overrides)
    hj = json.dumps(header).encode("utf-8"); rj = jpeg.tobytes()
    return (struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj
            + struct.pack("<I", len(depth)) + depth)


def _recv(mode: str, interval: float = 0.5) -> WebSocketReceiver:
    return WebSocketReceiver(ingest_queue=IngestQueue(maxsize=16), keyframe_every_n=1000,
                             nonkf_min_interval_s=interval, throttle_clock=mode)


class TestSensorThrottle:
    def test_sensor_mode_admits_by_header_time_regardless_of_wall_time(self):
        rx = _recv("sensor")
        s = int(1e9)
        assert rx._parse_binary_message(_ws_frame(1, 0)) is not None        # frame 1 = keyframe
        assert rx._parse_binary_message(_ws_frame(2, int(0.2 * s))) is not None   # first non-KF: admitted
        assert rx._parse_binary_message(_ws_frame(3, int(0.4 * s))) is None       # 0.2 s after -> throttled
        assert rx._parse_binary_message(_ws_frame(4, int(0.8 * s))) is not None   # 0.6 s after last ADMIT
        # all four parsed within milliseconds of wall time: wall mode would have throttled 3 AND 4
        assert rx._last_nonkf_admit_sensor_ns == int(0.8 * s)

    def test_wall_mode_unchanged(self):
        rx = _recv("wall")
        assert rx._parse_binary_message(_ws_frame(1, 0)) is not None
        assert rx._parse_binary_message(_ws_frame(2, int(1e9))) is not None      # first non-KF admitted
        assert rx._parse_binary_message(_ws_frame(3, int(5e9))) is None          # ~0 s wall later -> throttled
        rx._last_nonkf_enq_mono = time.monotonic() - 1.0                       # pretend 1 s passed
        assert rx._parse_binary_message(_ws_frame(4, int(9e9))) is not None

    def test_stamp_advances_on_admit_not_on_enqueue(self):
        """The receiver never enqueued anything here (we only parse), yet the
        throttle still thins: that is the E1 dead-throttle fix."""
        rx = _recv("sensor")
        s = int(1e9)
        rx._parse_binary_message(_ws_frame(1, 0))
        assert rx._parse_binary_message(_ws_frame(2, s)) is not None
        assert rx._last_nonkf_admit_sensor_ns == s                             # stamped at the decision
        assert rx._parse_binary_message(_ws_frame(3, s + int(0.1 * s))) is None

    def test_negative_delta_restarts_the_window(self):
        rx = _recv("sensor")
        s = int(1e9)
        rx._parse_binary_message(_ws_frame(1, 0))
        assert rx._parse_binary_message(_ws_frame(2, 10 * s)) is not None
        assert rx._parse_binary_message(_ws_frame(3, 2 * s)) is not None         # clock went backwards: admit + re-stamp
        assert rx._last_nonkf_admit_sensor_ns == 2 * s
        assert rx._parse_binary_message(_ws_frame(4, 2 * s + int(0.1 * s))) is None

    def test_sensor_mode_falls_back_to_wall_without_a_timestamp(self):
        rx = _recv("sensor")
        rx._parse_binary_message(_ws_frame(1, 0))
        assert rx._parse_binary_message(_ws_frame(2, 0)) is not None            # ts 0 -> wall path, first admit
        assert rx._parse_binary_message(_ws_frame(3, 0)) is None                # wall: ~0 s later -> throttled

    def test_session_reset_clears_sensor_stamp(self):
        rx = _recv("sensor")
        rx._parse_binary_message(_ws_frame(1, 0))
        rx._parse_binary_message(_ws_frame(2, int(1e9)))
        assert rx._last_nonkf_admit_sensor_ns is not None
        rx._frame_count = 0
        rx._last_nonkf_admit_sensor_ns = None                                   # what the handshake reset does
        assert rx._parse_binary_message(_ws_frame(1, 0)) is not None

    def test_frame_packet_carries_the_epoch(self):
        rx = _recv("sensor")
        pkt = rx._parse_binary_message(_ws_frame(1, 0))
        assert pkt is not None and pkt.frame_epoch == rx._frame_epoch


class TestZeroMQSensorThrottle:
    def test_sensor_mode_throttles_by_pose_timestamp(self):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber

        q = IngestQueue(maxsize=16)
        sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                               ingest_queue=q, throttle_clock="sensor")
        try:
            rgb = np.zeros((4, 4, 3), dtype=np.uint8); depth = np.ones((4, 4), dtype=np.float32)

            class _FW:
                def assemble_pair(self, ts_ns):
                    return rgb, depth, None
            sub.fw = _FW()
            t = np.zeros(3, dtype=np.float32); qq = np.array([0, 0, 0, 1], dtype=np.float32)
            s = int(1e9)
            sub._try_enqueue_frame(1 * s, t, qq, is_keyframe=False)      # admitted
            sub._try_enqueue_frame(1 * s + int(0.3 * s), t, qq, is_keyframe=False)   # throttled (sensor delta 0.3 s)
            sub._try_enqueue_frame(2 * s, t, qq, is_keyframe=False)      # admitted (1.0 s after last admit)
            assert q.qsize() == 2
            assert sub._last_nonkf_admit_sensor_ns == 2 * s
        finally:
            sub.close()


# ───────────────────────────── review follow-ups ─────────────────────────────

class TestReplaySensorThrottle:
    def test_replayer_thins_by_recorded_time_at_any_speed(self, tmp_path):
        """Three recorded frames: KF at t=0, non-KF at +0.3 s (recorded), non-KF
        at +1.0 s (recorded). At replay_speed=10 the wall gaps are ~0.03 s and
        ~0.07 s: wall mode would throttle BOTH non-KFs after the first admit;
        sensor mode admits by recorded time -> frame 2 admitted, frame 3
        admitted (1.0 - 0.3 = 0.7 s >= 0.5 s)."""
        from rtsm.io.recorder import SessionRecorder
        from rtsm.io.replayer import ReplayReceiver

        s = int(1e9)
        rec_dir = tmp_path / "rec"
        rec = SessionRecorder(output_dir=str(rec_dir))
        for fid, ts in ((1, 0), (2, int(0.3 * s)), (3, int(1.0 * s))):
            rec.on_message("binary", _ws_frame(fid, ts))
            time.sleep(0.02)
        rec.on_handshake({"type": "hello", "session_id": "s1"}, {"type": "hello_ack", "status": "ok"})
        rec.close()

        events = []
        q = IngestQueue(maxsize=8)
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=q, keyframe_every_n=30,
                            nonkf_min_interval_s=0.5, event_sink=events.append, replay_speed=10.0,
                            throttle_clock="sensor")
        rr.start()
        assert rr._done.wait(10.0)
        got = [(e.decision, e.reason, e.frame_seq) for e in events]
        assert got == [("enqueued", "", 1), ("enqueued", "", 2), ("enqueued", "", 3)]
        assert q.qsize() == 3


class TestZeroMQPairingDoesNotBurnTheWindow:
    def test_unpaired_pose_then_paired_pose_is_admitted(self):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber

        q = IngestQueue(maxsize=16)
        sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                               ingest_queue=q, throttle_clock="sensor")
        try:
            rgb = np.zeros((4, 4, 3), dtype=np.uint8); depth = np.ones((4, 4), dtype=np.float32)

            class _FW:
                def __init__(self):
                    self.has = False

                def assemble_pair(self, ts_ns):
                    return (rgb, depth, None) if self.has else (None, None, None)
            sub.fw = _FW()
            t = np.zeros(3, dtype=np.float32); qq = np.array([0, 0, 0, 1], dtype=np.float32)
            s = int(1e9)
            sub._try_enqueue_frame(1 * s, t, qq, is_keyframe=False)            # no camera frame yet -> not an admission
            assert q.qsize() == 0 and sub._last_nonkf_admit_sensor_ns is None
            sub.fw.has = True
            sub._try_enqueue_frame(1 * s + 33_000_000, t, qq, is_keyframe=False)   # 33 ms later, frame present
            assert q.qsize() == 1                                              # admitted: the window was not burned
            assert sub._last_nonkf_admit_sensor_ns == 1 * s + 33_000_000
        finally:
            sub.close()


class TestUpsertSchedulingOnTheClock:
    def test_mark_upsert_failed_requeues_at_the_injected_clock(self):
        clk = FakeClock(t=5000.0)
        wm = WorkingMemory(cfg={}, clock=clk)
        oid = wm.create_object(**_obs_args())
        clk.t = 5042.0
        assert wm.mark_upsert_failed([oid]) == 1
        due, heap_oid = min(wm._ltm_heap)
        assert (due, heap_oid) == (5042.0, oid)                                # heap due time = clock, not wall


class TestEventLogMeta:
    def test_meta_line_carries_the_ingest_clock(self, tmp_path):
        w = EventLogWriter(enabled=True, configured_path=str(tmp_path / "m.jsonl"), extra_meta={"ingest_clock": "sensor"})
        w.close()
        rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
        assert rows[0]["kind"] == "meta" and rows[0]["ingest_clock"] == "sensor"


class TestSensorClockRecovery:
    def test_reset_reanchors_on_the_next_frame(self):
        c = SensorClock(fallback=lambda: 100.0)
        c.advance(60_000_000_000)                                   # sensor 60 s -> now 100
        c.advance(61_000_000_000)                                   # now 101
        c.reset()
        assert c.now_mono() == 100.0                                # wall fallback until re-anchored
        assert c.advance(3_000_000_000) == 100.0                    # re-anchored: not clamped at 101
        assert c.advance(4_000_000_000) == pytest.approx(101.0)     # and it moves again

    def test_large_backwards_jump_in_one_epoch_rebases_instead_of_freezing(self):
        """A recording replayed twice in one process (or a source restart that
        did not bump frame_epoch): sensor time jumps back by the whole
        recording. The clock must keep running, not clamp forever."""
        c = SensorClock(fallback=lambda: 0.0, rebase_after_s=5.0)
        c.advance(1_000_000_000, frame_epoch=0)                     # now 0
        c.advance(45_000_000_000, frame_epoch=0)                    # now 44
        assert c.advance(1_000_000_000, frame_epoch=0) == pytest.approx(44.0)   # continuous, re-based
        assert c.rebases == 1 and c.clamped == 0
        assert c.advance(2_000_000_000, frame_epoch=0) == pytest.approx(45.0)   # running again

    def test_small_backwards_jitter_still_clamps(self):
        c = SensorClock(fallback=lambda: 0.0, rebase_after_s=5.0)
        c.advance(10_000_000_000, frame_epoch=0)                    # now 0
        assert c.advance(9_500_000_000, frame_epoch=0) == pytest.approx(0.0)     # 0.5 s back: clamp
        assert c.clamped == 1 and c.rebases == 0

    def test_first_epoch_after_epochless_start_is_adopted_not_rebased(self):
        c = SensorClock(fallback=lambda: 0.0)
        c.advance(1_000_000_000, frame_epoch=None)
        assert c.advance(2_000_000_000, frame_epoch=0) == pytest.approx(1.0)     # normal advance
        assert c.rebases == 0 and c.epoch == 0
