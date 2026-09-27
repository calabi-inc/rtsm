"""
P3 task 1 -- the bag source on the one ingest front-end, and the G3-1 parity
predicate: recordings/session1_bag through BagSource reproduces the replay
path's receiver decisions (the B1 record) and its packets, bit for bit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")

from rtsm.evaluation.event_log import RX_DROPPED, RX_ENQUEUED, RX_THROTTLE
from rtsm.io.contracts import Source, SourceContext
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, IngestFrontEnd
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.sources import make_source

REPO = Path(__file__).resolve().parents[1]
SESSION1 = REPO / "recordings" / "session1"
SESSION1_BAG = REPO / "recordings" / "session1_bag"
B1 = REPO / "eval" / "baselines" / "2026-09-sensor-clock" / "B1.events.jsonl"


def _ctx(q, events=None, **kw):
    base = dict(ingest_queue=q, clock_mode="sensor", keyframe_every_n=30, nonkf_min_interval_s=0.5,
                require_tracking_normal=True, confidence_threshold=2, event_sink=(events.append if events is not None else None))
    base.update(kw)
    return SourceContext(**base)


def test_bag_source_is_a_source_and_reports_refusals(tmp_path):
    from test_bag_reader import simple_bag_messages, write_bag
    bag = write_bag(tmp_path / "b", simple_bag_messages(8, seq_topic=True, tracking=["normal", "limited"]))
    events = []
    q = IngestQueue(64)
    src = make_source("bag", {"io": {"bag": {}}}, _ctx(q, events, keyframe_every_n=3, nonkf_min_interval_s=0.15), path=str(bag))
    assert isinstance(src, Source) and src.name == "bag"
    src.start()
    assert src.wait(30)
    st = src.stats()
    assert st["error"] is None and st["yielded"] == 8 and st["paired"] == 8
    # the tracking topic exists -> the filter is on: "limited" frames dropped before the keyframe rule
    assert src.frontend.require_tracking_normal is True
    decisions = [(e.decision, e.reason, e.frame_seq, e.is_keyframe) for e in events]
    limited = [d for d in decisions if d[1] == "tracking_state"]
    assert len(limited) == 4 and all(d[2] % 2 == 1 for d in limited)
    enq = [d for d in decisions if d[0] == RX_ENQUEUED]
    assert enq[0][3] is True and enq[0][2] == 100                                  # first admitted frame = keyframe, source seq
    assert q.qsize() == st["enqueued"] == len(enq)
    assert src.liveness()["alive"] is False and src.frontend.frame_epoch == 1
    # a refused bag surfaces through stats()/error, not an exception in the runner thread
    nod = write_bag(tmp_path / "nodepth", [m for m in simple_bag_messages(3) if "depth" not in m[0]])
    src2 = make_source("bag", {"io": {"bag": {}}}, _ctx(IngestQueue(8)), path=str(nod))
    src2.start(); assert src2.wait(30)
    assert src2.stats()["refusal_codes"] == ["no_depth_topic"] and src2.error is not None


def test_bag_without_tracking_topic_turns_the_filter_off(tmp_path):
    from test_bag_reader import simple_bag_messages, write_bag
    bag = write_bag(tmp_path / "b", simple_bag_messages(4))
    q = IngestQueue(64)
    src = make_source("bag", {}, _ctx(q, keyframe_every_n=2, nonkf_min_interval_s=0.0), path=str(bag))
    src.start(); assert src.wait(30)
    assert src.frontend.require_tracking_normal is False and q.qsize() == 4


def test_speed_paces_by_header_stamps(tmp_path):
    import time
    from test_bag_reader import simple_bag_messages, write_bag
    bag = write_bag(tmp_path / "b", simple_bag_messages(4))                       # stamps 100 ms apart
    src = make_source("bag", {"io": {"bag": {"speed": 2.0}}}, _ctx(IngestQueue(64), nonkf_min_interval_s=0.0), path=str(bag))
    t0 = time.monotonic(); src.start(); assert src.wait(30); dt = time.monotonic() - t0
    assert 0.12 <= dt < 5.0                                                           # 3 gaps x 100 ms / 2x


def test_registry_config_defaults_and_options(tmp_path):
    from test_bag_reader import simple_bag_messages, write_bag
    bag = write_bag(tmp_path / "b", simple_bag_messages(3))
    cfg = {"io": {"bag": {"path": str(bag), "pair_tolerance_s": 0.5, "typestore": "humble", "topics": {"rgb": "/camera/color/image_raw"}}}}
    src = make_source("bag", cfg, _ctx(IngestQueue(8)))
    assert src.path == str(bag)
    with pytest.raises(ValueError, match="needs a path"):
        make_source("bag", {}, _ctx(IngestQueue(8)))


# ───────────────────────────── G3-1 parity ─────────────────────────────

def _replay_packets_and_lines():
    """The replay path: recordings/session1 through parse_lens_message on a front-end with the replay settings."""
    from rtsm.evaluation.recording_mcap import iter_recording
    from rtsm.io.websocket import parse_lens_message
    events = []
    q = IngestQueue(4096)
    fe = IngestFrontEnd(source="replay", policy=WEBSOCKET_POLICY, ingest_queue=q, throttle_clock="sensor", keyframe_every_n=30,
                        nonkf_min_interval_s=0.5, require_tracking_normal=True, confidence_threshold=2, event_sink=events.append)
    fe.new_session("4F4E55A3-9132-49DD-B11A-37DC06FD1881")
    pkts = []
    for _e, data in iter_recording(SESSION1):
        pkt = parse_lens_message(fe, data, apply_camera_flip=True)
        if pkt is not None and fe.enqueue(pkt):
            pkts.append(pkt)
    return pkts, events


def _drain(q):
    out = []
    while True:
        p = q.get(timeout=0)
        if p is None:
            return out
        out.append(p)


@pytest.mark.skipif(not ((SESSION1 / "messages.bin").is_file() and (SESSION1_BAG / "metadata.yaml").is_file() and B1.is_file()),
                    reason="session1 recording, its bag and the B1 record are needed")
def test_session1_bag_reproduces_the_replay_path():
    events = []
    q = IngestQueue(4096)
    src = make_source("bag", {"io": {"bag": {}}}, _ctx(q, events), path=str(SESSION1_BAG))
    src.start(); assert src.wait(600)
    st = src.stats()
    assert st["error"] is None and st["yielded"] == 240 and st["paired"] == 240 and st["pose_missing"] == 0
    bag_pkts = _drain(q)
    # 1. receiver decisions == the B1 record (P1 task 1 anchor run of the replay receiver)
    b1 = [json.loads(l) for l in B1.read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = [(r["decision"], r["reason"], r["frame_seq"], r["t_sensor_ns"], r["is_keyframe"], r["frame_count"]) for r in b1 if r.get("kind") == "receiver"]
    got = [(e.decision, e.reason, e.frame_seq, e.t_sensor_ns, e.is_keyframe, e.frame_count) for e in events]
    assert len(got) == len(ref) == 240
    assert got == ref
    assert sum(1 for g in got if g[0] == RX_ENQUEUED) == 86 and sum(1 for g in got if g[1] == RX_THROTTLE) == 154
    # 2. packets == the replay path's packets, bit for bit
    rep_pkts, rep_events = _replay_packets_and_lines()
    assert [(e.decision, e.reason, e.frame_seq, e.t_sensor_ns, e.is_keyframe, e.frame_count) for e in rep_events] == ref
    assert len(bag_pkts) == len(rep_pkts) == 86
    for b, r in zip(bag_pkts, rep_pkts):
        assert (b.time.seq, b.time.t_sensor_ns, b.is_keyframe, b.ingest.keyframe_origin, b.ingest.rx_seq) == \
               (r.time.seq, r.time.t_sensor_ns, r.is_keyframe, r.ingest.keyframe_origin, r.ingest.rx_seq)
        assert abs(b.time.t_wall_utc_s - r.time.t_wall_utc_s) < 1e-6
        assert np.array_equal(b.rgb, r.rgb)
        assert np.array_equal(np.isnan(b.depth_m), np.isnan(r.depth_m)) and np.array_equal(np.nan_to_num(b.depth_m), np.nan_to_num(r.depth_m))
        assert np.array_equal(b.confidence, r.confidence)
        assert (b.intr.width, b.intr.height, b.intr.fx, b.intr.fy, b.intr.cx, b.intr.cy) == (r.intr.width, r.intr.height, r.intr.fx, r.intr.fy, r.intr.cx, r.intr.cy)
        assert np.array_equal(b.pose.t_wc, r.pose.t_wc) and np.array_equal(b.pose.q_wc_xyzw, r.pose.q_wc_xyzw)
        assert b.ingest.depth_valid_frac == r.ingest.depth_valid_frac
        assert b.rgb_jpeg is None                                                     # the bag path never carries JPEG bytes
