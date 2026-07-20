"""
Tests for receive-time robot pose passthrough.

Covers:
- WorkingMemory.update_robot_pose monotonic-timestamp guard (a slow pipeline
  write must not overwrite a fresher receive-time pose) and clear() reset.
- WebSocketReceiver pose_sink firing at input rate — including frames the
  keyframe/interval throttle skips — with the same post-flip pose the
  FramePacket carries.
- Frame epoch: bumps when a hello carries a new session_id (sender re-created
  its streaming session → world origin may have moved), rides the pose sink,
  and is preserved by epoch-less writers (pipeline dual-write).
"""

from __future__ import annotations
import json
import struct
import time

import numpy as np
import cv2
import pytest

from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver
from rtsm.stores.working_memory import WorkingMemory


# ─────────────────── update_robot_pose guard ───────────────────


class TestUpdateRobotPoseGuard:
    def _wm(self) -> WorkingMemory:
        return WorkingMemory(cfg={})

    def test_first_write_stored(self):
        wm = self._wm()
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0)
        pose = wm.get_robot_pose()
        assert pose is not None
        assert pose["xyz"] == [1.0, 2.0, 3.0]
        assert pose["timestamp"] == 100.0

    def test_newer_overwrites(self):
        wm = self._wm()
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0)
        wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 101.0)
        assert wm.get_robot_pose()["xyz"] == [4.0, 5.0, 6.0]

    def test_older_rejected(self):
        """The pipeline writes the pose of an already-processed (older) frame;
        it must not stomp a fresher receive-time pose."""
        wm = self._wm()
        wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 101.0)
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0)
        pose = wm.get_robot_pose()
        assert pose["xyz"] == [4.0, 5.0, 6.0]
        assert pose["timestamp"] == 101.0

    def test_equal_timestamp_accepted(self):
        """Same frame written twice (receiver then pipeline) is harmless."""
        wm = self._wm()
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0)
        wm.update_robot_pose(np.array([1.5, 2.5, 3.5]), np.array([0, 0, 0, 1.0]), 100.0)
        assert wm.get_robot_pose()["xyz"] == [1.5, 2.5, 3.5]

    def test_older_accepted_after_staleness_window(self):
        """A sender clock stepping backward (new device / NTP) must not
        freeze the pose forever: once the stored pose is older than the
        guard window, an older-timestamped update is accepted."""
        wm = self._wm()
        wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 1000.0)
        # Simulate the stored pose aging past the guard window.
        wm._latest_pose_arrival_mono -= wm._POSE_GUARD_WINDOW_S + 1.0
        wm.update_robot_pose(np.array([7.0, 8.0, 9.0]), np.array([0, 0, 0, 1.0]), 10.0)
        pose = wm.get_robot_pose()
        assert pose["xyz"] == [7.0, 8.0, 9.0]
        assert pose["timestamp"] == 10.0

    def test_clear_resets_guard(self):
        """After clear() (e.g. /reset between replay runs or demo trials),
        older re-fed timestamps must be accepted again."""
        wm = self._wm()
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 1000.0)
        wm.clear()
        assert wm.get_robot_pose() is None
        wm.update_robot_pose(np.array([7.0, 8.0, 9.0]), np.array([0, 0, 0, 1.0]), 5.0)
        assert wm.get_robot_pose()["xyz"] == [7.0, 8.0, 9.0]


# ─────────────────── WebSocketReceiver pose_sink ───────────────────


def _identity_T_wc_col_major(t=(1.0, 2.0, 3.0)) -> list:
    """4x4 identity-rotation pose with translation t, flattened column-major
    (ARKit wire format)."""
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = t
    return list(m.flatten(order="F"))


def _binary_message(header: dict, rgb_bytes: bytes = b"", depth_bytes: bytes = b"") -> bytes:
    hj = json.dumps(header).encode("utf-8")
    return (
        struct.pack("<I", len(hj)) + hj
        + struct.pack("<I", len(rgb_bytes)) + rgb_bytes
        + struct.pack("<I", len(depth_bytes)) + depth_bytes
    )


def _make_receiver(pose_sink, apply_camera_flip: bool = False) -> WebSocketReceiver:
    return WebSocketReceiver(
        ingest_queue=IngestQueue(maxsize=4),
        require_tracking_normal=True,
        keyframe_every_n=30,
        nonkf_min_interval_s=0.5,
        apply_camera_flip=apply_camera_flip,
        pose_sink=pose_sink,
    )


def _header(unix_ts: float = 1234.5) -> dict:
    return {
        "T_wc": _identity_T_wc_col_major(),
        "pose_format": "matrix4x4_col_major",
        "tracking_state": "normal",
        "unix_timestamp": unix_ts,
        "timestamp_ns": 42,
        "frame_id": 7,
        "rgb_format": "jpeg",
        "rgb_width": 2,
        "rgb_height": 2,
        "fx": 1.0, "fy": 1.0, "cx": 1.0, "cy": 1.0,
    }


class TestPoseSinkReceiveRate:
    def test_sink_fires_on_throttled_frame(self):
        """The key behavior: a frame the non-KF interval throttle skips
        (parse returns None, no image decode) still delivers its pose."""
        calls = []
        rx = _make_receiver(lambda t, q, ts, ep: calls.append((t.copy(), np.asarray(q).copy(), ts)))
        # Arm the throttle: pretend frame 1 was just enqueued.
        rx._frame_count = 1
        rx._last_nonkf_enq_mono = time.monotonic()

        pkt = rx._parse_binary_message(_binary_message(_header(unix_ts=999.25)))

        assert pkt is None, "frame should be throttled (no FramePacket)"
        assert len(calls) == 1, "pose sink must fire even for throttled frames"
        t, q, ts = calls[0]
        np.testing.assert_allclose(t, [1.0, 2.0, 3.0], atol=1e-6)
        assert ts == 999.25

    def test_sink_not_called_on_bad_tracking(self):
        calls = []
        rx = _make_receiver(lambda t, q, ts, ep: calls.append(ts))
        hdr = _header()
        hdr["tracking_state"] = "limited"
        pkt = rx._parse_binary_message(_binary_message(hdr))
        assert pkt is None
        assert calls == [], "garbage-tracking poses must not reach the sink"

    def test_sink_pose_matches_framepacket_pose(self):
        """The sink must deliver the same (post-convention) pose the
        FramePacket carries — one convention for all consumers."""
        calls = []
        rx = _make_receiver(lambda t, q, ts, ep: calls.append((np.asarray(t).copy(), np.asarray(q).copy())))
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", img)
        pkt = rx._parse_binary_message(
            _binary_message(_header(), rgb_bytes=jpeg.tobytes())
        )  # frame 1 → keyframe → fully parsed
        assert pkt is not None
        assert len(calls) == 1
        t, q = calls[0]
        np.testing.assert_allclose(t, pkt.pose.t_wc, atol=1e-6)
        np.testing.assert_allclose(q, pkt.pose.q_wc_xyzw, atol=1e-6)

    def test_flip_applied_to_sink_pose(self):
        """With apply_camera_flip the sink sees the flipped (OpenCV) pose:
        translation unchanged, rotation changed from identity."""
        calls_flip, calls_noflip = [], []
        for flip, calls in ((True, calls_flip), (False, calls_noflip)):
            rx = _make_receiver(
                lambda t, q, ts, ep, c=calls: c.append((np.asarray(t).copy(), np.asarray(q).copy())),
                apply_camera_flip=flip,
            )
            rx._frame_count = 1
            rx._last_nonkf_enq_mono = time.monotonic()
            assert rx._parse_binary_message(_binary_message(_header())) is None

        (t_f, q_f), (t_n, q_n) = calls_flip[0], calls_noflip[0]
        np.testing.assert_allclose(t_f, t_n, atol=1e-6)  # translation unaffected
        assert not np.allclose(np.abs(q_f), np.abs(q_n), atol=1e-6), (
            "flip must change the rotation quaternion"
        )

    def test_no_sink_is_safe(self):
        rx = _make_receiver(pose_sink=None)
        rx._frame_count = 1
        rx._last_nonkf_enq_mono = time.monotonic()
        assert rx._parse_binary_message(_binary_message(_header())) is None

    def test_sink_error_does_not_break_parse(self):
        def boom(t, q, ts, ep):
            raise RuntimeError("sink failure")

        rx = _make_receiver(pose_sink=boom)
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", img)
        pkt = rx._parse_binary_message(_binary_message(_header(), rgb_bytes=jpeg.tobytes()))
        assert pkt is not None, "a failing sink must not break frame parsing"


# ─────────────────── frame epoch ───────────────────


class TestFrameEpoch:
    def test_epoch_bumps_on_new_session_only(self):
        rx = _make_receiver(pose_sink=None)
        assert rx._note_session("A") == 1
        assert rx._note_session("A") == 1, "same-id reconnect keeps the epoch"
        assert rx._note_session("B") == 2
        assert rx._note_session("B") == 2
        assert rx._note_session("A") == 3, "returning to an old id is still a new session"

    def test_sink_receives_current_epoch(self):
        calls = []
        rx = _make_receiver(lambda t, q, ts, ep: calls.append(ep))
        rx._note_session("A")
        rx._frame_count = 1
        rx._last_nonkf_enq_mono = time.monotonic()
        assert rx._parse_binary_message(_binary_message(_header())) is None  # throttled
        rx._note_session("B")
        assert rx._parse_binary_message(_binary_message(_header(unix_ts=1235.5))) is None
        assert calls == [1, 2]

    def test_wm_stores_epoch_and_none_preserves(self):
        """The receiver writes with an epoch; the pipeline dual-writer calls
        without one (None) and must NOT clear the stored epoch."""
        wm = WorkingMemory(cfg={})
        wm.update_robot_pose(
            np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0, frame_epoch=3
        )
        assert wm.get_robot_pose()["frame_epoch"] == 3
        # pipeline-style 3-arg write, newer timestamp → pose updates, epoch kept
        wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 101.0)
        pose = wm.get_robot_pose()
        assert pose["xyz"] == [4.0, 5.0, 6.0]
        assert pose["frame_epoch"] == 3
        # next receiver write with a new epoch replaces it
        wm.update_robot_pose(
            np.array([7.0, 8.0, 9.0]), np.array([0, 0, 0, 1.0]), 102.0, frame_epoch=4
        )
        assert wm.get_robot_pose()["frame_epoch"] == 4

    def test_wm_epoch_none_when_never_provided(self):
        """ZMQ/replay paths have no epochs — the field exists but is None."""
        wm = WorkingMemory(cfg={})
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0)
        assert wm.get_robot_pose()["frame_epoch"] is None

    def test_epoch_survives_wm_clear(self):
        """clear() (demo /reset) drops the pose; the next receive-time write
        re-carries the receiver's current epoch."""
        wm = WorkingMemory(cfg={})
        wm.update_robot_pose(
            np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0, frame_epoch=2
        )
        wm.clear()
        assert wm.get_robot_pose() is None
        wm.update_robot_pose(
            np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 101.0, frame_epoch=2
        )
        assert wm.get_robot_pose()["frame_epoch"] == 2

    def test_stats_surfaces_epoch(self):
        wm = WorkingMemory(cfg={})
        wm.update_robot_pose(
            np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0, frame_epoch=5
        )
        assert wm.stats()["robot_pose"]["frame_epoch"] == 5
