"""Tests for the pose-conversion frame-drop fix.

A frame whose pose is PRESENT but fails matrix conversion (raises, or
produces non-finite values) must be dropped in _get_snapshot_via_queue —
never handed downstream where the associator would substitute an identity
world transform and write camera-frame coordinates into the world-frame map.

A frame with NO pose at all (pose-less sources, e.g. the webcam demo) keeps
its existing behavior: snapshot proceeds with pose_cam_T_world=None.
"""
from __future__ import annotations

import numpy as np

from rtsm.core.datamodel import FramePacket, PoseStamped, TimeBundle
from rtsm.core.pipeline import Pipeline
from rtsm.io.ingest_queue import IngestQueue


def _make_pipeline(q: IngestQueue) -> Pipeline:
    return Pipeline(
        cfg={},
        segmenter=None,
        clip=None,
        working_mem=None,
        proximity_index=None,
        associator=None,
        ingest_gate=None,
        ingest_q=q,
    )


def _packet(pose) -> FramePacket:
    return FramePacket(
        rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        depth_m=None,
        pose=pose,
        intr=None,
        is_keyframe=True,
        time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=0),
    )


def _valid_pose() -> PoseStamped:
    return PoseStamped(
        stamp_ns=0,
        frame_id="arkit",
        t_wc=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        q_wc_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),  # identity rotation
    )


class _RaisingPose:
    """Duck-typed pose whose matrix conversion raises (corrupt input)."""

    t_wc = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def T_wc(self):
        raise ValueError("corrupt pose payload")


def test_valid_pose_converts_to_inverse_matrix():
    q = IngestQueue()
    pipe = _make_pipeline(q)
    q.put(_packet(_valid_pose()))
    snap, pkt = pipe._get_snapshot_via_queue()
    assert snap is not None and pkt is not None
    T = snap.pose_cam_T_world
    assert T is not None and np.isfinite(T).all()
    # Identity rotation: T_cam_world translation must be -t_wc
    np.testing.assert_allclose(T[:3, 3], [-1.0, -2.0, -3.0], atol=1e-6)
    assert pipe.pose_conversion_failures == 0


def test_no_pose_keeps_cameraframe_behavior():
    """pkt.pose=None (pose-less source) is NOT a failure: snapshot proceeds
    with pose_cam_T_world=None, nothing is counted."""
    q = IngestQueue()
    pipe = _make_pipeline(q)
    q.put(_packet(None))
    snap, pkt = pipe._get_snapshot_via_queue()
    assert snap is not None
    assert snap.pose_cam_T_world is None
    assert pipe.pose_conversion_failures == 0


def test_raising_pose_drops_frame_and_counts():
    q = IngestQueue()
    pipe = _make_pipeline(q)
    q.put(_packet(_RaisingPose()))
    snap, pkt = pipe._get_snapshot_via_queue()
    assert snap is None and pkt is None  # frame dropped, nothing downstream
    assert pipe.pose_conversion_failures == 1


def test_nonfinite_pose_drops_frame_and_counts():
    """NaN-poisoned poses don't raise in T_wc() — they produce a NaN matrix.
    The finiteness check must catch them."""
    q = IngestQueue()
    pipe = _make_pipeline(q)
    bad = _valid_pose()
    bad.t_wc = np.array([np.nan, 0.0, 0.0], dtype=np.float32)
    q.put(_packet(bad))
    snap, pkt = pipe._get_snapshot_via_queue()
    assert snap is None and pkt is None
    assert pipe.pose_conversion_failures == 1


def test_counter_accumulates_and_good_frames_still_flow():
    q = IngestQueue()
    pipe = _make_pipeline(q)
    for _ in range(3):
        q.put(_packet(_RaisingPose()))
    q.put(_packet(_valid_pose()))
    results = [pipe._get_snapshot_via_queue() for _ in range(4)]
    assert [snap is None for snap, _ in results] == [True, True, True, False]
    assert pipe.pose_conversion_failures == 3


def test_dropped_frame_never_reaches_memory(monkeypatch):
    """End-to-end through run_one_step: a corrupt-pose frame must not touch
    segmentation, association, or working memory."""
    q = IngestQueue()
    pipe = _make_pipeline(q)
    touched = []
    monkeypatch.setattr(
        pipe, "segmenter",
        type("Seg", (), {"segment": lambda self, *a, **k: touched.append("segment")})(),
    )
    q.put(_packet(_RaisingPose()))
    pipe.run_one_step()
    assert touched == []
    assert pipe.pose_conversion_failures == 1
