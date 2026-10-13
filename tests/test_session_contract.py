"""Stage 1 of the session registry: the packet carries the session identity and the
source's keyframe id, and the front-end tells the registry about session changes.

Pure front-end tests on the fake source of tests/test_ingest_frontend.py: no sockets,
no models.
"""
from __future__ import annotations

from dataclasses import fields

import numpy as np

from rtsm.core.datamodel import FramePacket, PinholeIntrinsics
from rtsm.io.contracts import (
    CONVENTION_OPENCV, POSE_FMT_PREPARED, EncodedImage, FrameHeader, PoseSample, RawFrame, SourceContext,
)
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, ZEROMQ_POLICY, IngestFrontEnd
from rtsm.io.ingest_queue import IngestQueue

INTR = PinholeIntrinsics(width=8, height=6, fx=5.0, fy=5.0, cx=4.0, cy=3.0)
IDENT_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _raw(seq: int, ts_ns: int, *, extra=None) -> RawFrame:
    rgb = np.full((6, 8, 3), seq % 255, dtype=np.uint8)
    depth = np.ones((6, 8), dtype=np.float32)
    return RawFrame(header=FrameHeader(
        source="fake", seq=seq, t_sensor_ns=ts_ns, t_wall_utc_s=1.7e9 + seq, tracking_state="normal",
        keyframe_hint=None, rgb=EncodedImage(rgb, "raw_bgr", 8, 6), depth=EncodedImage(depth, "raw_depth_m", 8, 6),
        intrinsics=INTR, pose_raw=(np.array([float(seq), 0.0, 0.0], dtype=np.float32), IDENT_Q),
        pose_format=POSE_FMT_PREPARED, pose_convention=CONVENTION_OPENCV, extra=dict(extra or {}),
    ))


def _fe(policy=WEBSOCKET_POLICY, **kw) -> IngestFrontEnd:
    return IngestFrontEnd(source="fake", policy=policy, ingest_queue=IngestQueue(8), throttle_clock="sensor",
                          keyframe_every_n=3, nonkf_min_interval_s=0.5, **kw)


def test_packet_and_context_gain_optional_fields():
    names = {f.name: f for f in fields(FramePacket)}
    assert names["session_id"].default is None
    assert names["kf_id"].default is None
    ctx = {f.name: f for f in fields(SourceContext)}
    assert ctx["on_session_change"].default is None
    assert ctx["on_corrections_flush"].default is None


def test_new_session_notifies_on_every_call_and_bumps_only_on_a_new_id():
    seen = []
    fe = _fe(on_session_change=lambda sid, epoch: seen.append((sid, epoch)))
    assert fe.new_session("A") == 1
    assert fe.new_session("A") == 1            # a same-id reconnect: no bump, still notified
    assert fe.new_session("B") == 2
    assert seen == [("A", 1), ("A", 1), ("B", 2)]


def test_stamp_regression_epoch_is_notified_with_no_session_id():
    seen = []
    fe = _fe(ZEROMQ_POLICY, on_session_change=lambda sid, epoch: seen.append((sid, epoch)))
    t = np.zeros(3, dtype=np.float32)
    fe.pose_event(PoseSample(t_sensor_ns=100_000_000_000, t_wall_utc_s=None, t_wc=t, q_wc_xyzw=IDENT_Q))
    fe.pose_event(PoseSample(t_sensor_ns=10_000_000_000, t_wall_utc_s=None, t_wc=t, q_wc_xyzw=IDENT_Q))
    assert fe.frame_epoch == 1
    assert seen == [(None, 1)]


def test_packet_carries_session_id_and_kf_id():
    fe = _fe()
    fe.new_session("sess-A")
    pkt = fe.admit(_raw(1, 1_000_000_000, extra={"kf_id": 42}))
    assert pkt is not None and pkt.is_keyframe
    assert pkt.session_id == "sess-A"
    assert pkt.kf_id == "42"
    pkt2 = fe.admit(_raw(2, 2_000_000_000))
    assert pkt2 is not None
    assert pkt2.session_id == "sess-A" and pkt2.kf_id is None


def test_session_change_callback_error_is_swallowed():
    def boom(sid, epoch):
        raise RuntimeError("registry down")
    fe = _fe(on_session_change=boom)
    assert fe.new_session("A") == 1
    assert fe.frame_epoch == 1
