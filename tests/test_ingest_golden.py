"""Golden ingest traces (Gate 4.5 plan, P3 task 0.5 -- the ingest front-end extraction).

Two deterministic synthetic streams, one per receiver, recorded from the code
BEFORE the extraction into `tests/fixtures/ingest_golden_*.json`: every receiver
decision line, every pose-sink call, every pose-ledger line, every packet's
summary and every callback. After the extraction the same streams must
reproduce the fixtures exactly -- this is the CPU regression net for a
behaviour-preserving refactor, beside the session1 anchor on the GPU.

Regenerate ONLY when a behaviour change is intended and gated:
    RTSM_WRITE_GOLDEN=1 python -m pytest tests/test_ingest_golden.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from rtsm.evaluation.event_log import RX_ENQUEUED
from rtsm.io.ingest_queue import IngestQueue

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WRITE = os.environ.get("RTSM_WRITE_GOLDEN") == "1"


# ───────────────────────────── helpers ─────────────────────────────

def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def _r(v, nd=4):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, nd) for x in v]
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    if isinstance(v, (int, np.integer, bool, np.bool_)):
        return int(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
    return v


def _rx_line(e) -> dict:
    d = {k: v for k, v in e.__dict__.items() if k != "timestamp"}
    return _r(d)


def _pose_line(p) -> dict:
    d = {k: v for k, v in p.__dict__.items() if k != "timestamp"}
    return _r(d)


def _pkt_summary(pkt) -> dict:
    m = pkt.ingest
    return {
        "seq": pkt.time.seq, "t_sensor_ns": pkt.time.t_sensor_ns, "is_keyframe": bool(pkt.is_keyframe),
        "frame_epoch": pkt.frame_epoch,
        "keyframe_origin": getattr(m, "keyframe_origin", None), "rx_seq": getattr(m, "rx_seq", None),
        "depth_valid_frac": _r(getattr(m, "depth_valid_frac", None), 6),
        "intr": (_r([pkt.intr.fx, pkt.intr.fy, pkt.intr.cx, pkt.intr.cy, pkt.intr.width, pkt.intr.height]) if pkt.intr else None),
        "rgb": [list(pkt.rgb.shape), _sha(pkt.rgb)],
        "depth": ([list(pkt.depth_m.shape), int(np.isnan(pkt.depth_m).sum()), _sha(np.nan_to_num(pkt.depth_m, nan=-1.0))]
                  if pkt.depth_m is not None else None),
        "pose": ([_r(pkt.pose.t_wc.tolist()), _r(pkt.pose.q_wc_xyzw.tolist())] if pkt.pose is not None else None),
        "confidence": (list(pkt.confidence.shape) if pkt.confidence is not None else None),
        "rgb_jpeg": (len(pkt.rgb_jpeg) if pkt.rgb_jpeg is not None else None),
    }


def _compare_or_write(name: str, got: dict) -> None:
    path = FIXTURES / f"ingest_golden_{name}.json"
    if WRITE:
        FIXTURES.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True), encoding="utf-8")
        pytest.skip(f"golden fixture written: {path}")
    assert path.is_file(), f"missing fixture {path}; generate it from the reference code with RTSM_WRITE_GOLDEN=1"
    want = json.loads(path.read_text(encoding="utf-8"))
    for key in want:
        assert got.get(key) == want[key], f"{name}: section {key!r} differs\n got: {json.dumps(got.get(key))[:1500]}\nwant: {json.dumps(want[key])[:1500]}"
    assert set(got) == set(want), (sorted(got), sorted(want))


# ───────────────────────────── websocket stream ─────────────────────────────

def _ws_frame(*, frame_id, timestamp_ns, tracking_state="normal", T_wc=None, unix_ts=1700000000.0,
              conf=None, rgb_seed=0, truncate=False) -> bytes:
    rng = np.random.default_rng(rgb_seed)
    rgb = rng.integers(0, 255, size=(16, 16, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", rgb)
    depth = (np.ones((8, 8), dtype=np.uint16) * (1500 + 10 * frame_id))
    depth[0, :3] = 0                                                       # a few invalid pixels
    header = {
        "frame_id": frame_id, "timestamp_ns": timestamp_ns, "unix_timestamp": unix_ts,
        "rgb_format": "jpeg", "rgb_width": 16, "rgb_height": 16,
        "depth_format": "uint16_mm", "depth_width": 8, "depth_height": 8, "depth_scale": 0.001,
        "fx": 20.0, "fy": 20.0, "cx": 16.0, "cy": 16.0, "intrinsics_width": 32, "intrinsics_height": 32,
        "pose_format": "quat_translation", "T_wc": T_wc if T_wc is not None else [0.0, 0.7071068, 0.0, 0.7071068, 1.0, 2.0, 3.0],
        "tracking_state": tracking_state,
    }
    parts = b""
    if conf is not None:
        header.update({"confidence_format": "uint8", "confidence_width": int(conf.shape[1]), "confidence_height": int(conf.shape[0])})
        cb = np.ascontiguousarray(conf, dtype=np.uint8).tobytes()
        parts = struct.pack("<I", len(cb)) + cb
    hj = json.dumps(header).encode("utf-8"); rj = jpeg.tobytes(); dj = depth.tobytes()
    msg = struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj + struct.pack("<I", len(dj)) + dj + parts
    return msg[: len(msg) // 2] if truncate else msg


def run_websocket_stream() -> dict:
    """The stream loop's contract, driven by hand: parse -> put -> trace enqueued
    (or the receiver's own refusal path), plus a mid-stream new session and a
    pose_corrections text message."""
    from rtsm.io.websocket import WebSocketReceiver

    events, poses, sink_calls, cam, kfs, corr = [], [], [], [], [], []
    q = IngestQueue(maxsize=2)                                          # never drained -> refusals after 2
    recv = WebSocketReceiver(
        ingest_queue=q, keyframe_every_n=4, nonkf_min_interval_s=0.5, confidence_threshold=2,
        apply_camera_flip=True, throttle_clock="sensor",
        pose_sink=lambda t, qq, ts, ep, **kw: sink_calls.append([_r(t.tolist()), _r(qq.tolist()), _r(ts, 3), ep, _r(kw)]),
        event_sink=events.append, ledger_sink=poses.append,
        on_camera_frame=lambda p: cam.append(p.time.seq), on_keyframe=lambda p: kfs.append(p.time.seq),
        on_pose_corrections_batch=lambda b: corr.append(sorted(b)),
    )
    recv._note_session("s1")
    conf = np.zeros((4, 4), dtype=np.uint8); conf.flat[:6] = 2; conf.flat[6:9] = 1
    stream = [
        dict(frame_id=1, timestamp_ns=1_000_000_000, conf=conf, rgb_seed=1),           # KF (first)
        dict(frame_id=2, timestamp_ns=1_100_000_000, rgb_seed=2),                      # non-KF admitted (first)
        dict(frame_id=3, timestamp_ns=1_200_000_000, rgb_seed=3),                      # throttled (0.1 s)
        dict(frame_id=4, timestamp_ns=1_300_000_000, tracking_state="limited"),        # tracking drop (pose parses)
        dict(frame_id=5, timestamp_ns=1_350_000_000, tracking_state="limited", T_wc=[1.0, 2.0]),   # tracking drop, bad pose
        dict(frame_id=6, timestamp_ns=1_700_000_000, rgb_seed=6),                      # admitted -> queue full? (q holds 2) -> refused
        dict(frame_id=7, timestamp_ns=1_750_000_000, truncate=True),                   # malformed
        dict(frame_id=8, timestamp_ns=1_800_000_000, T_wc=[1.0, 2.0]),                 # parse_error (raises)
        dict(frame_id=9, timestamp_ns=2_400_000_000, rgb_seed=9, unix_ts=0),           # server clock; KF? (count 4)
        "session:s2",
        dict(frame_id=10, timestamp_ns=500_000_000, rgb_seed=10),                      # new session: count 1 -> KF, stamps reset
        dict(frame_id=11, timestamp_ns=600_000_000, rgb_seed=11),                      # non-KF (first after reset)
        "corrections",
        dict(frame_id=12, timestamp_ns=1_200_000_000, rgb_seed=12, conf=conf),         # admitted (0.6 s)
    ]
    packets, errors = [], []
    for item in stream:
        if item == "session:s2":
            recv._note_session("s2"); continue
        if item == "corrections":
            recv._handle_text_message(json.dumps({"type": "pose_corrections", "corrections": {
                "ws_1": [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3], "ws_2": list(np.eye(4, dtype=np.float32).flatten(order="F"))}}))
            continue
        try:
            pkt = recv._parse_binary_message(_ws_frame(**item))
        except Exception as e:                                             # the stream loop logs and continues
            errors.append(type(e).__name__); continue
        if pkt is None:
            continue
        packets.append(_pkt_summary(pkt))
        if q.put(pkt, block=False):
            recv._trace_rx(RX_ENQUEUED, "", pkt=pkt)
            recv._last_enq_ts_ns = pkt.time.t_sensor_ns
            cam.append(pkt.time.seq)
            if pkt.is_keyframe:
                kfs.append(pkt.time.seq)
        else:
            reason = getattr(getattr(pkt, "ingest", None), "drop_reason", None) or "queue_full"
            recv._trace_rx("dropped", reason, pkt=pkt)
    return {
        "receiver": [_rx_line(e) for e in events],
        "pose_sink": sink_calls,
        "pose_ledger": [_pose_line(p) for p in poses],
        "packets": packets,
        "callbacks": {"camera": cam, "keyframe": kfs, "corrections": corr, "errors": errors},
        "state": {"frame_count": recv._frame_count, "frame_epoch": recv._frame_epoch, "tracking_drops": recv.tracking_drops,
                  "queue": q.qsize()},
    }


# ───────────────────────────── zeromq stream ─────────────────────────────

def _zmq_msg(topic: str, **fields) -> list:
    return [topic.encode(), json.dumps(fields).encode()]


def _zmq_camera(ts_ns: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8); _, jpg = cv2.imencode(".jpg", rgb)
    d = (np.ones((8, 8), dtype=np.uint16) * (1000 + seed)); d[0, 0] = 0
    _, png = cv2.imencode(".png", d)
    meta = {"ts_ns": ts_ns, "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0, "width": 8, "height": 8}, "depth_units_m": 0.001}
    return [b"camera.rgbd", json.dumps(meta).encode(), jpg.tobytes(), png.tobytes()]


def run_zeromq_stream() -> dict:
    pytest.importorskip("zmq")
    from rtsm.io.zeromq import ZeroMQSubscriber

    events, poses, sink_calls, cam = [], [], [], []
    q = IngestQueue(maxsize=2)
    sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1", ingest_queue=q,
                           pose_sink=lambda t, qq, ts, ep, **kw: sink_calls.append([_r(t.tolist()), _r(qq.tolist()), ep, _r(kw)]),
                           event_sink=events.append, ledger_sink=poses.append, throttle_clock="sensor",
                           nonkf_min_interval_s=0.5)
    try:
        base = 10_000                                                       # stamp_ms
        script = []
        for i in range(6):                                                  # poses every 33 ms, camera for most
            ms = base + i * 33
            if i != 2:
                script.append(("cam", ms, i))
            script.append(("pose", ms, [0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0]))
        script.append(("pose", base + 5 * 33, [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))        # duplicate stamp
        script.append(("cam", base + 600, 6)); script.append(("pose", base + 600, [0.6, 0.0, 0.0, 0.0, 0.0, 0.0]))   # due again
        script.append(("kf", base + 600, 4)); script.append(("kf_nostamp", None, 5))    # stamped kf + stampless kf (inherits)
        script.append(("cam", base + 1200, 7)); script.append(("pose", base + 1200, [1.2, 0.0, 0.0, 0.0, 0.0, 0.0]))  # refusal (q full)
        script.append(("bad", None, None))                                              # malformed (3 parts)
        script.append(("cam", 1_000, 8)); script.append(("pose", 1_000, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))            # 9 s back -> epoch 1
        script.append(("pose", 1_033, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))                  # no camera frame
        for kind, ms, arg in script:
            if kind == "cam":
                sub._handle_camera_rgbd(_zmq_camera(ms * 1_000_000, arg))
            elif kind == "pose":
                sub._handle_tracking_pose(_zmq_msg("rtabmap.tracking_pose", stamp_ms=ms, T_wc=arg))
            elif kind == "kf":
                sub._handle_kf_pose(_zmq_msg("rtabmap.kf_pose", kf_id=arg, stamp_ms=ms, T_wc=[0.6, 0.0, 0.0, 0.0, 0.0, 0.0]))
            elif kind == "kf_nostamp":
                sub._handle_kf_pose(_zmq_msg("rtabmap.kf_pose", kf_id=arg, T_wc=[0.6, 0.0, 0.0, 0.0, 0.0, 0.0]))
            elif kind == "bad":
                sub._handle_tracking_pose([b"rtabmap.tracking_pose", b"{}", b"extra"])
        packets = []
        while True:
            pkt = q.get(timeout=0.0)
            if pkt is None:
                break
            packets.append(_pkt_summary(pkt))
        return {
            "receiver": [_rx_line(e) for e in events],
            "pose_sink": sink_calls,
            "pose_ledger": [_pose_line(p) for p in poses],
            "packets": packets,
            "state": {"frame_epoch": sub._frame_epoch, "kf_stamps_inherited": sub._kf_stamps_inherited,
                      "last_pose_ts_ns": sub._last_pose_ts_ns, "window": sub.fw.stats() if hasattr(sub.fw, "stats") else None},
        }
    finally:
        sub.close()


# ───────────────────────────── tests ─────────────────────────────

def test_websocket_golden():
    _compare_or_write("websocket", run_websocket_stream())


def test_zeromq_golden():
    _compare_or_write("zeromq", run_zeromq_stream())
