"""Robot-pose receive-time path: the websocket pose sink fires for every
tracking-normal frame (throttled ones included) with the post-convention pose,
the frame epoch bumps on a new session, and WorkingMemory.update_robot_pose is
the single-writer MAILBOX (P1 task 4): a same-epoch older stamp is accepted and
counted as a regression; only an older-epoch or epoch-less write is rejected
(second-writer safety net). See tests/test_pose_mailbox.py for the full table.
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


class TestUpdateRobotPoseMailbox:
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

    def test_older_accepted_and_counted_as_a_regression(self):
        """P1 task 4: there is ONE writer (the receiver, at receive time; the
        pipeline no longer writes the pose), so an older stamp is the sender
        restarting or stepping its clock. It is accepted -- the pose follows
        the sender -- and counted as a regression for diagnostics."""
        wm = self._wm()
        wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 101.0)
        assert wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0) is True
        pose = wm.get_robot_pose()
        assert pose["xyz"] == [1.0, 2.0, 3.0]
        assert pose["timestamp"] == 100.0
        assert pose["sensor_ts_regressions"] == 1 and pose["rejected_writes"] == 0

    def test_equal_timestamp_accepted(self):
        """The same frame written twice by the one writer is harmless."""
        wm = self._wm()
        wm.update_robot_pose(np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0)
        wm.update_robot_pose(np.array([1.5, 2.5, 3.5]), np.array([0, 0, 0, 1.0]), 100.0)
        assert wm.get_robot_pose()["xyz"] == [1.5, 2.5, 3.5]

    def test_no_time_window_governs_acceptance(self):
        """P1 task 4: the 2 s fail-open window is gone in both directions --
        acceptance never depends on how long ago the stored pose arrived."""
        wm = self._wm()
        wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 1000.0)
        wm._latest_pose_arrival_mono -= 60.0
        assert wm.update_robot_pose(np.array([7.0, 8.0, 9.0]), np.array([0, 0, 0, 1.0]), 10.0) is True
        pose = wm.get_robot_pose()
        assert pose["xyz"] == [7.0, 8.0, 9.0] and pose["sensor_ts_regressions"] == 1
        assert wm.update_robot_pose(np.array([1.0, 1.0, 1.0]), np.array([0, 0, 0, 1.0]), 11.0) is True   # immediately, too
        assert wm.get_robot_pose()["sensor_ts_regressions"] == 1

    def test_clear_resets_mailbox(self):
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
        rx = _make_receiver(lambda t, q, ts, ep, **kw: calls.append((t.copy(), np.asarray(q).copy(), ts)))
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
        rx = _make_receiver(lambda t, q, ts, ep, **kw: calls.append(ts))
        hdr = _header()
        hdr["tracking_state"] = "limited"
        pkt = rx._parse_binary_message(_binary_message(hdr))
        assert pkt is None
        assert calls == [], "garbage-tracking poses must not reach the sink"

    def test_sink_pose_matches_framepacket_pose(self):
        """The sink must deliver the same (post-convention) pose the
        FramePacket carries — one convention for all consumers."""
        calls = []
        rx = _make_receiver(lambda t, q, ts, ep, **kw: calls.append((np.asarray(t).copy(), np.asarray(q).copy())))
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
                lambda t, q, ts, ep, c=calls, **kw: c.append((np.asarray(t).copy(), np.asarray(q).copy())),
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
        rx = _make_receiver(lambda t, q, ts, ep, **kw: calls.append(ep))
        rx._note_session("A")
        rx._frame_count = 1
        rx._last_nonkf_enq_mono = time.monotonic()
        assert rx._parse_binary_message(_binary_message(_header())) is None  # throttled
        rx._note_session("B")
        assert rx._parse_binary_message(_binary_message(_header(unix_ts=1235.5))) is None
        assert calls == [1, 2]

    def test_wm_stores_epoch_and_epochless_write_is_rejected(self):
        """P1 task 4: the receiver writes with an epoch; a writer that knows
        no epoch (legacy 3-arg call) is REJECTED against an epoch-bearing
        pose -- it can neither clear nor bypass the epoch. (The pipeline no
        longer writes the pose at all; this is the second-writer safety net.)"""
        wm = WorkingMemory(cfg={})
        wm.update_robot_pose(
            np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0, 1.0]), 100.0, frame_epoch=3
        )
        assert wm.get_robot_pose()["frame_epoch"] == 3
        assert wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), np.array([0, 0, 0, 1.0]), 101.0) is False
        pose = wm.get_robot_pose()
        assert pose["xyz"] == [1.0, 2.0, 3.0]
        assert pose["frame_epoch"] == 3
        assert pose["rejected_by_reason"]["epochless_into_epoch"] == 1
        # next receiver write with a new epoch replaces it
        wm.update_robot_pose(
            np.array([7.0, 8.0, 9.0]), np.array([0, 0, 0, 1.0]), 102.0, frame_epoch=4
        )
        assert wm.get_robot_pose()["frame_epoch"] == 4

    def test_wm_epoch_none_when_never_provided(self):
        """The ZMQ path has no epochs — the field exists but is None (replay
        carries the recording's epoch since P1 task 2 wired its pose sink)."""
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
