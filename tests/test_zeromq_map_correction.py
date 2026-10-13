"""Stage 6 of the session registry: the ZeroMQ lane's two correction topics.

rtabmap.map_correction (the bridge's map->odom) composed onto tracking poses at
receive time, the map id as the session id, the burst-end flush; the RTAB-Map
node id on the keyframe packet. The subscriber is driven through its handlers
with JSON parts, no sockets (tests/evaluation/test_ledgers.py pattern).
"""
from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest

from rtsm.io.ingest_queue import IngestQueue

S = math.sqrt(0.5)
YAW90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
TS = 5_000_000_000


def _sub(poses=None, **kw):
    pytest.importorskip("zmq")
    from rtsm.io.zeromq import ZeroMQSubscriber
    sink = (lambda t, q, wall, epoch, **k: poses.append((np.asarray(t, np.float32), np.asarray(q, np.float32), epoch))) if poses is not None else None
    return ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                            ingest_queue=IngestQueue(maxsize=8), pose_sink=sink, **kw)


def _msg(topic: str, **fields) -> list:
    return [topic.encode(), json.dumps(fields).encode()]


def _map_corr(map_id=1, ts_ns=TS - 1_000_000_000, t=(1.0, 0.0, 0.0), q=(0.0, 0.0, S, S)):
    return _msg("rtabmap.map_correction", T_w_odom={"t": list(t), "q": list(q)}, map_id=map_id, ts_ns=ts_ns)


def _tracking(ts_ns=TS, xyz=(0.0, 0.0, 0.0)):
    return _msg("rtabmap.tracking_pose", stamp_ms=ts_ns / 1e6, T_wc=[*xyz, 0.0, 0.0, 0.0])


def _camera_msg(ts_ns: int):
    rgb = np.full((8, 8, 3), 90, dtype=np.uint8)
    _, jpg = cv2.imencode(".jpg", rgb)
    depth = np.ones((8, 8), dtype=np.uint16) * 1234
    _, png = cv2.imencode(".png", depth)
    meta = {"ts_ns": ts_ns, "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0, "width": 8, "height": 8},
            "depth_units_m": 0.001}
    return [b"camera.rgbd", json.dumps(meta).encode("utf-8"), jpg.tobytes(), png.tobytes()]


def _R(q):
    x, y, z, w = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=np.float32)


def test_tracking_pose_composed_with_map_correction():
    poses = []
    sub = _sub(poses)
    try:
        sub._handle_map_correction(_map_corr())
        sub._handle_tracking_pose(_tracking(TS, (0.0, 0.0, 0.0)))
        sub._handle_tracking_pose(_tracking(TS + 33_000_000, (1.0, 0.0, 0.0)))
        assert len(poses) == 2
        t0, q0, _ = poses[0]
        assert np.allclose(t0, [1.0, 0.0, 0.0], atol=1e-5) and np.allclose(_R(q0), YAW90, atol=1e-5)
        t1, q1, _ = poses[1]
        assert np.allclose(t1, [1.0, 1.0, 0.0], atol=1e-5)          # T_w_odom @ T_odom_c, not the other order
        assert np.allclose(_R(q1), YAW90, atol=1e-5)
        st = sub.stats()["map_correction"]
        assert st["received"] == 1 and st["applied"] == 2 and st["map_id"] == 1 and st["enabled"] is True
        assert st["age_s"] == pytest.approx((TS + 33_000_000 - (TS - 1_000_000_000)) / 1e9)
    finally:
        sub.close()


def test_composed_pose_reaches_the_packet():
    sub = _sub()
    try:
        sub._handle_camera_rgbd(_camera_msg(TS))
        sub._handle_map_correction(_map_corr())
        sub._handle_tracking_pose(_tracking(TS, (0.0, 0.0, 0.0)))
        pkt = sub.ingest_q.get(timeout=0)
        assert pkt is not None and not pkt.is_keyframe and pkt.kf_id is None
        assert np.allclose(pkt.pose.t_wc, [1.0, 0.0, 0.0], atol=1e-5)
        assert pkt.session_id == "rtabmap-map-1" and pkt.frame_epoch == 1
    finally:
        sub.close()


def test_no_map_correction_is_identity():
    poses = []
    sub = _sub(poses)
    try:
        sub._handle_tracking_pose(_tracking(TS, (2.0, 0.0, 0.0)))
        t, q, _ = poses[0]
        assert np.allclose(t, [2.0, 0.0, 0.0]) and np.allclose(_R(q), np.eye(3), atol=1e-6)
        assert sub.stats()["map_correction"]["applied"] == 0 and sub.stats()["map_correction"]["map_id"] is None
    finally:
        sub.close()


def test_apply_map_correction_false_keeps_odometry_frame():
    poses = []
    sub = _sub(poses, apply_map_correction=False)
    try:
        sub._handle_map_correction(_map_corr())
        sub._handle_tracking_pose(_tracking(TS, (2.0, 0.0, 0.0)))
        t, q, _ = poses[0]
        assert np.allclose(t, [2.0, 0.0, 0.0]) and np.allclose(_R(q), np.eye(3), atol=1e-6)
        st = sub.stats()["map_correction"]
        assert st["enabled"] is False and st["received"] == 1 and st["applied"] == 0
        assert st["map_id"] == 1                                      # the session id still follows the map id
    finally:
        sub.close()


def test_map_id_change_bumps_epoch_once():
    seen = []
    sub = _sub(on_session_change=lambda sid, epoch: seen.append((sid, epoch)))
    try:
        for mid in (1, 1, 2, 0, 2):
            sub._handle_map_correction(_map_corr(map_id=mid))
        assert sub._fe.frame_epoch == 2 and sub._fe._epoch_session_id == "rtabmap-map-2"
        assert seen == [("rtabmap-map-1", 1), ("rtabmap-map-2", 2)]
        assert sub.stats()["map_correction"]["map_id"] == 2
    finally:
        sub.close()


def test_kf_id_on_header_and_packet():
    sub = _sub()
    try:
        sub._handle_camera_rgbd(_camera_msg(TS))
        sub._handle_kf_pose(_msg("rtabmap.kf_pose", kf_id=42, stamp_ms=TS / 1e6, T_wc=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        pkt = sub.ingest_q.get(timeout=0)
        assert pkt is not None and pkt.is_keyframe and pkt.kf_id == "42"
        sub._handle_kf_pose(_msg("rtabmap.kf_pose", stamp_ms=TS / 1e6, T_wc=[0.0] * 6))      # no kf_id: None
        pkt2 = sub.ingest_q.get(timeout=0)
        assert pkt2 is not None and pkt2.kf_id is None
    finally:
        sub.close()


def test_map_correction_calls_flush_and_counts_malformed():
    flushes = []
    sub = _sub(on_corrections_flush=lambda: flushes.append(1))
    try:
        sub._handle_map_correction(_map_corr(map_id=1))
        sub._handle_map_correction(_map_corr(map_id=1))
        assert len(flushes) == 2
        sub._handle_map_correction([b"rtabmap.map_correction", b"{not json"])
        sub._handle_map_correction(_msg("rtabmap.map_correction", T_w_odom={"t": [0, 0, 0]}, map_id=3))
        sub._handle_map_correction([b"rtabmap.map_correction"])
        st = sub.stats()["map_correction"]
        assert st["received"] == 5 and st["malformed"] == 3 and st["map_id"] == 1 and len(flushes) == 2
        assert sub._T_w_odom is not None                              # the last good correction stays
    finally:
        sub.close()


def test_kf_pose_is_not_composed():
    """The node pose from Statistics::poses() is already map-frame."""
    poses = []
    sub = _sub(poses)
    try:
        sub._handle_camera_rgbd(_camera_msg(TS))
        sub._handle_map_correction(_map_corr())
        sub._handle_kf_pose(_msg("rtabmap.kf_pose", kf_id=7, stamp_ms=TS / 1e6, T_wc=[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        pkt = sub.ingest_q.get(timeout=0)
        assert np.allclose(pkt.pose.t_wc, [3.0, 0.0, 0.0])
        assert sub.stats()["map_correction"]["applied"] == 0
    finally:
        sub.close()
