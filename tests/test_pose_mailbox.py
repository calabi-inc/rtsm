"""Robot-pose mailbox (Gate 4.5 plan, P1 task 4): one receive-time writer per
process (the pipeline no longer writes the pose); writes are keyed on
(frame_epoch, sensor_ts_ns) as a REGRESSION DETECTOR (a stamp going backwards
within an epoch is accepted and counted, never used to freeze the pose);
only a second writer can be rejected (older epoch, epoch-less into an epoch);
the payload carries age_s / stale / pose_clock / counters; the websocket sink
passes the sensor stamp and a sender|server clock tag; ZeroMQ writes the pose
at receive time for tracking poses with a receiver-minted epoch, and a
stamp-less kf_pose inherits the last tracking stamp. CPU-only.
"""
from __future__ import annotations

import inspect
import json
import logging
import struct

import cv2
import numpy as np
import pytest

from rtsm.io.ingest_queue import IngestQueue
from rtsm.stores.working_memory import WorkingMemory, resolve_pose_stale_after_s

T = np.array([1.0, 2.0, 3.0])
Q = np.array([0.0, 0.0, 0.0, 1.0])
LOG = "rtsm.stores.working_memory"


def _wm(**cfg) -> WorkingMemory:
    return WorkingMemory(cfg=cfg)


def _xyz(wm):
    return wm.get_robot_pose()["xyz"]


# ── the rule ─────────────────────────────────────────────────────────────────

class TestSingleWriterRule:
    def test_older_stamp_is_accepted_and_counted_as_a_regression(self):
        """No time window and no rejection: the single writer says this is the
        pose now (looped replay, bag loop, stepped clock)."""
        wm = _wm()
        assert wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), Q, 1000.0, frame_epoch=0, sensor_ts_ns=1000)
        wm._latest_pose_arrival_mono -= 60.0
        assert wm.update_robot_pose(T, Q, 10.0, frame_epoch=0, sensor_ts_ns=10) is True
        p = wm.get_robot_pose()
        assert (p["xyz"], p["sensor_ts_ns"]) == ([1.0, 2.0, 3.0], 10)
        assert (p["sensor_ts_regressions"], p["rejected_writes"], p["writes_accepted"]) == (1, 0, 2)

    def test_new_epoch_with_a_smaller_stamp_is_not_a_regression(self):
        wm = _wm()
        assert wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), Q, 1000.0, frame_epoch=1, sensor_ts_ns=1000)
        assert wm.update_robot_pose(T, Q, 10.0, frame_epoch=2, sensor_ts_ns=10)
        p = wm.get_robot_pose()
        assert (p["xyz"], p["frame_epoch"], p["sensor_ts_ns"], p["sensor_ts_regressions"]) == ([1.0, 2.0, 3.0], 2, 10, 0)

    def test_older_epoch_is_rejected_even_with_a_larger_stamp(self):
        """Only a second writer can do this (a drained previous-session frame)."""
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 10.0, frame_epoch=2, sensor_ts_ns=10)
        assert wm.update_robot_pose(np.array([9.0, 9.0, 9.0]), Q, 5000.0, frame_epoch=1, sensor_ts_ns=5000) is False
        p = wm.get_robot_pose()
        assert p["frame_epoch"] == 2 and p["rejected_by_reason"]["older_epoch"] == 1 and p["rejected_writes"] == 1

    def test_epochless_write_into_an_epoch_mailbox_is_rejected_and_epoch_unchanged(self):
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 100.0, frame_epoch=2, sensor_ts_ns=100)
        assert wm.update_robot_pose(np.array([7.0, 8.0, 9.0]), Q, 1000.0, sensor_ts_ns=1000) is False
        p = wm.get_robot_pose()
        assert p["xyz"] == [1.0, 2.0, 3.0] and p["frame_epoch"] == 2
        assert p["rejected_by_reason"]["epochless_into_epoch"] == 1

    def test_first_epoch_bearing_write_adopts_the_epoch(self):
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 100.0, sensor_ts_ns=100)                       # epoch-less first
        assert wm.update_robot_pose(np.array([4.0, 5.0, 6.0]), Q, 50.0, frame_epoch=1, sensor_ts_ns=50)
        p = wm.get_robot_pose()
        assert p["frame_epoch"] == 1 and p["sensor_ts_regressions"] == 0               # adopted, not a regression

    def test_equal_key_is_accepted_and_keeps_the_stored_clock_tag(self):
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 100.0, frame_epoch=0, sensor_ts_ns=100, pose_clock="sender")
        assert wm.update_robot_pose(np.array([1.5, 2.5, 3.5]), Q, 100.0, frame_epoch=0, sensor_ts_ns=100)
        p = wm.get_robot_pose()
        assert p["xyz"] == [1.5, 2.5, 3.5] and p["pose_clock"] == "sender" and p["sensor_ts_regressions"] == 0

    def test_regressions_are_detected_on_the_sensor_stamp_not_the_wall_timestamp(self):
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 5.0, frame_epoch=0, sensor_ts_ns=200)
        assert wm.update_robot_pose(T, Q, 999.0, frame_epoch=0, sensor_ts_ns=100)      # stamp back: regression
        assert wm.update_robot_pose(T, Q, 1.0, frame_epoch=0, sensor_ts_ns=300)        # wall back, stamp forward: fine
        assert wm.get_robot_pose()["sensor_ts_regressions"] == 1

    def test_wall_timestamp_is_the_key_without_sensor_stamps_and_zero_counts_as_none(self):
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 100.0) and wm.update_robot_pose(T, Q, 99.0)  # timestamp regression
        assert wm.get_robot_pose()["sensor_ts_regressions"] == 1
        assert wm.update_robot_pose(T, Q, 101.0, sensor_ts_ns=0)                        # 0 = no stamp
        assert wm.get_robot_pose()["sensor_ts_ns"] is None
        assert wm.update_robot_pose(T, Q, 100.5, sensor_ts_ns=0)                        # judged on the timestamp
        assert wm.get_robot_pose()["sensor_ts_regressions"] == 2

    def test_looped_sender_keeps_the_pose_advancing_and_warns_once_per_minute(self, caplog):
        """The G1-C shape: session1 streamed in a loop behind one handshake.
        Every write is accepted, one regression per wrap, one WARNING."""
        wm = _wm()
        with caplog.at_level(logging.WARNING, logger=LOG):
            for loop in range(3):
                for i in range(5):
                    assert wm.update_robot_pose(np.array([float(loop), float(i), 0.0]), Q, 1000.0 + loop * 5 + i,
                                                frame_epoch=0, sensor_ts_ns=1_000 + i * 100)
        p = wm.get_robot_pose()
        assert (p["writes_accepted"], p["sensor_ts_regressions"], p["rejected_writes"]) == (15, 2, 0)
        assert p["xyz"] == [2.0, 4.0, 0.0]                                              # follows the sender
        assert sum(1 for r in caplog.records if "pose key went backwards" in r.getMessage()) == 1

    def test_second_writer_rejections_warn_with_backoff(self, caplog):
        wm = _wm()
        assert wm.update_robot_pose(T, Q, 100.0, frame_epoch=2, sensor_ts_ns=100)
        with caplog.at_level(logging.WARNING, logger=LOG):
            for i in range(1000):
                assert wm.update_robot_pose(T, Q, 200.0 + i, sensor_ts_ns=200 + i) is False
        assert wm.get_robot_pose()["rejected_by_reason"]["epochless_into_epoch"] == 1000
        assert sum(1 for r in caplog.records if "write rejected" in r.getMessage()) == 3       # at 1, 100, 1000


# ── payload, config, reset ───────────────────────────────────────────────────

class TestPayload:
    def test_age_stale_and_counters(self):
        wm = _wm(robot_pose={"stale_after_s": 0.5})
        assert wm.get_robot_pose() is None and wm.stats()["robot_pose"] is None
        wm.update_robot_pose(T, Q, 100.0, frame_epoch=0, sensor_ts_ns=100, pose_clock="sender")
        p = wm.get_robot_pose()
        assert p["stale"] is False and p["age_s"] < 0.5 and p["stale_after_s"] == 0.5 and p["pose_clock"] == "sender"
        assert (p["writes_accepted"], p["rejected_writes"], p["sensor_ts_regressions"]) == (1, 0, 0)
        wm._latest_pose_arrival_mono -= 2.0
        p = wm.get_robot_pose()
        assert p["stale"] is True and 1.9 < p["age_s"] < 3.0
        assert wm.stats()["robot_pose"]["stale"] is True                        # same builder behind /stats

    def test_payload_is_a_copy(self):
        wm = _wm()
        wm.update_robot_pose(T, Q, 100.0, sensor_ts_ns=100)
        wm.get_robot_pose()["xyz"].append(99.0)
        assert _xyz(wm) == [1.0, 2.0, 3.0]

    def test_payload_is_json_serialisable_with_numpy_inputs(self):
        wm = _wm()
        wm.update_robot_pose(np.float32([1, 2, 3]), np.float32([0, 0, 0, 1]), np.float64(100.0),
                             frame_epoch=np.int64(2), sensor_ts_ns=np.int64(5))
        p = wm.get_robot_pose()
        json.dumps(p)
        assert (p["frame_epoch"], p["sensor_ts_ns"]) == (2, 5) and type(p["frame_epoch"]) is int

    def test_stale_threshold_is_validated_not_coerced(self):
        assert resolve_pose_stale_after_s({}) == 0.5
        assert resolve_pose_stale_after_s({"robot_pose": {"stale_after_s": None}}) == 0.5
        assert resolve_pose_stale_after_s({"robot_pose": {"stale_after_s": "1.5"}}) == 1.5
        for bad in (0, -1, "x", True, float("inf")):
            with pytest.raises(ValueError, match="robot_pose.stale_after_s"):
                resolve_pose_stale_after_s({"robot_pose": {"stale_after_s": bad}})
        with pytest.raises(ValueError, match="robot_pose must be a mapping"):
            resolve_pose_stale_after_s({"robot_pose": 0.5})                      # the obvious one-key typo
        with pytest.raises(ValueError):
            WorkingMemory(cfg={"robot_pose": {"stale_after_s": 0}})

    def test_clear_resets_mailbox_and_counters(self):
        wm = _wm()
        wm.update_robot_pose(T, Q, 1000.0, frame_epoch=5, sensor_ts_ns=1000)
        wm.update_robot_pose(T, Q, 1.0, frame_epoch=5, sensor_ts_ns=1)                 # regression
        assert wm.update_robot_pose(T, Q, 1.0, sensor_ts_ns=1) is False                # epoch-less: rejected
        wm.clear()
        assert wm.get_robot_pose() is None
        assert wm.update_robot_pose(np.array([7.0, 8.0, 9.0]), Q, 5.0, sensor_ts_ns=5)
        p = wm.get_robot_pose()
        assert (p["writes_accepted"], p["rejected_writes"], p["sensor_ts_regressions"], p["frame_epoch"]) == (1, 0, 0, None)


# ── websocket sink + no pipeline write ───────────────────────────────────────

def _ws_frame(frame_id: int, timestamp_ns: int, unix_ts=1700000000.0, with_unix=True, T_wc=(0.0, 0.0, 0.0)) -> bytes:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8); _, jpeg = cv2.imencode(".jpg", rgb)
    depth = np.ones((8, 8), dtype=np.uint16) * 1500
    header = {"frame_id": frame_id, "timestamp_ns": timestamp_ns,
              "rgb_format": "jpeg", "rgb_width": 8, "rgb_height": 8,
              "depth_format": "uint16_mm", "depth_width": 8, "depth_height": 8, "depth_scale": 0.001,
              "fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0,
              "pose_format": "quat_translation", "T_wc": [0.0, 0.0, 0.0, 1.0, *T_wc], "tracking_state": "normal"}
    if with_unix:
        header["unix_timestamp"] = unix_ts
    hj = json.dumps(header).encode(); rj = jpeg.tobytes(); dj = depth.tobytes()
    return struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj + struct.pack("<I", len(dj)) + dj


class TestWebSocketSink:
    def test_sink_gets_sensor_stamp_and_clock_tag(self):
        from rtsm.io.websocket import WebSocketReceiver
        calls = []
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 pose_sink=lambda t, q, ts, ep, **kw: calls.append((ts, ep, kw)))
        recv._note_session("A")
        recv._parse_binary_message(_ws_frame(1, 5_000, unix_ts=123.5))
        recv._parse_binary_message(_ws_frame(2, 6_000, with_unix=False))
        recv._parse_binary_message(_ws_frame(3, 0))                                      # zero stamp -> None
        (ts1, ep1, kw1), (ts2, ep2, kw2), (_, _, kw3) = calls
        assert (ts1, ep1, kw1) == (123.5, 1, {"sensor_ts_ns": 5_000, "pose_clock": "sender"})
        assert ep2 == 1 and kw2 == {"sensor_ts_ns": 6_000, "pose_clock": "server"} and ts2 > 1e9   # time.time() substituted
        assert kw3["sensor_ts_ns"] is None

    def test_receive_time_writes_end_to_end_and_the_pipeline_never_writes(self):
        from rtsm.core import pipeline as pipeline_mod
        from rtsm.io.websocket import WebSocketReceiver
        assert ".update_robot_pose(" not in inspect.getsource(pipeline_mod), \
            "the pipeline must not write the robot pose: the receivers do, at receive time (P1 task 4)"
        wm = _wm()
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 pose_sink=wm.update_robot_pose)
        recv._note_session("A")
        recv._parse_binary_message(_ws_frame(1, 1_000, unix_ts=100.0, T_wc=(1.0, 0.0, 0.0)))
        recv._parse_binary_message(_ws_frame(2, 2_000, unix_ts=101.0, T_wc=(2.0, 0.0, 0.0)))
        p = wm.get_robot_pose()
        assert p["xyz"][0] == pytest.approx(2.0) and (p["frame_epoch"], p["sensor_ts_ns"], p["pose_clock"]) == (1, 2_000, "sender")
        assert (p["writes_accepted"], p["sensor_ts_regressions"], p["rejected_writes"]) == (2, 0, 0)


# ── ZeroMQ: receive-time sink, receiver-minted epoch, stamp inheritance ──────

class TestZeroMQPose:
    def _sub(self, sink):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber
        return ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                                ingest_queue=IngestQueue(maxsize=8), pose_sink=sink)

    @staticmethod
    def _msg(topic: str, **fields) -> list:
        return [topic.encode(), json.dumps(fields).encode()]

    @staticmethod
    def _camera_msg(ts_ns: int) -> list:
        rgb = np.full((8, 8, 3), 90, dtype=np.uint8); _, jpg = cv2.imencode(".jpg", rgb)
        _, png = cv2.imencode(".png", np.ones((8, 8), dtype=np.uint16) * 1234)
        meta = {"ts_ns": ts_ns, "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0, "width": 8, "height": 8},
                "depth_units_m": 0.001}
        return [b"camera.rgbd", json.dumps(meta).encode(), jpg.tobytes(), png.tobytes()]

    def test_tracking_poses_write_at_receive_time_and_keyframes_do_not(self):
        wm = _wm(); calls = []

        def sink(t, q, ts, ep, **kw):
            calls.append((ep, kw)); return wm.update_robot_pose(t, q, ts, ep, **kw)
        sub = self._sub(sink)
        try:
            base_ms = 1_700_000_000_000
            for i in range(30):
                sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=base_ms + i * 33,
                                                    T_wc=[0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0]))
            enq = []
            real = sub._try_enqueue_frame
            sub._try_enqueue_frame = lambda ts, t, q, is_keyframe: (enq.append((ts, is_keyframe)), real(ts, t, q, is_keyframe))[1]
            sub._handle_kf_pose(self._msg("rtabmap.kf_pose", kf_id=4, T_wc=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            assert len(calls) == 30                                                      # kf_pose: no pose write
            assert all(ep == 0 and kw["pose_clock"] == "server" for ep, kw in calls)
            assert [kw["sensor_ts_ns"] for _, kw in calls] == [(base_ms + i * 33) * 1_000_000 for i in range(30)]
            assert enq == [((base_ms + 29 * 33) * 1_000_000, True)]                       # inherited the last tracking stamp
            assert sub._kf_stamps_inherited == 1 and sub.liveness()["kf_stamps_inherited"] == 1
            p = wm.get_robot_pose()
            assert (p["writes_accepted"], p["sensor_ts_regressions"], p["rejected_writes"], p["frame_epoch"]) == (30, 0, 0, 0)
            assert p["xyz"][0] == pytest.approx(2.9)
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", T_wc=[0, 0, 0, 0, 0, 0]))   # stamp-less TRACKING pose
            assert sub._kf_stamps_inherited == 1                                            # ...is not a keyframe stamp
        finally:
            sub.close()

    def test_stamp_jumping_back_over_five_seconds_mints_a_new_epoch(self):
        """A bag loop / bridge restart: every write accepted, epoch 0 -> 1, no
        regression counted (a new epoch is a restart), FramePacket carries it."""
        wm = _wm(); epochs = []
        sub = self._sub(lambda t, q, ts, ep, **kw: (epochs.append(ep), wm.update_robot_pose(t, q, ts, ep, **kw))[1])
        try:
            sub._nonkf_min_interval_s = 0.0
            for i in range(300):
                sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=10_000 + i * 33, T_wc=[0, 0, 0, 0, 0, 0]))
            for i in range(300):                                                          # wrap: ~9 s back (a 0 stamp would be "no stamp")
                sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=1_000 + i * 33, T_wc=[0, 0, 0, 0, 0, 0]))
            assert epochs == [0] * 300 + [1] * 300 and sub.liveness()["frame_epoch"] == 1
            p = wm.get_robot_pose()
            assert (p["writes_accepted"], p["sensor_ts_regressions"], p["rejected_writes"], p["frame_epoch"]) == (600, 0, 0, 1)
            ts = (1_000 + 300 * 33) * 1_000_000
            sub._handle_camera_rgbd(self._camera_msg(ts))
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=1_000 + 300 * 33, T_wc=[0, 0, 0, 0, 0, 0]))
            pkt = sub.ingest_q.get()
            assert pkt is not None and pkt.frame_epoch == 1                              # the packet carries the epoch
        finally:
            sub.close()

    def test_zero_stamp_neither_bumps_the_epoch_nor_becomes_the_reference(self):
        wm = _wm()
        sub = self._sub(wm.update_robot_pose)
        try:
            for ms in (5_000, 0, 5_033):                                              # an uninitialised ROS header in between
                sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=ms, T_wc=[0, 0, 0, 0, 0, 0]))
            p = wm.get_robot_pose()
            assert (sub.liveness()["frame_epoch"], p["frame_epoch"], p["sensor_ts_regressions"], p["writes_accepted"]) == (0, 0, 0, 3)
            assert sub._last_pose_ts_ns == 5_033_000_000
        finally:
            sub.close()

    def test_small_backwards_step_is_a_regression_not_an_epoch(self):
        wm = _wm()
        sub = self._sub(wm.update_robot_pose)
        try:
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=5_000, T_wc=[0, 0, 0, 0, 0, 0]))
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=4_000, T_wc=[0, 0, 0, 0, 0, 0]))   # 1 s back
            p = wm.get_robot_pose()
            assert (p["frame_epoch"], p["sensor_ts_regressions"], p["writes_accepted"]) == (0, 1, 2)
        finally:
            sub.close()

    def test_stampless_keyframe_fallback_chain_and_duplicate_image_avoidance(self, monkeypatch):
        sub = self._sub(None)
        try:
            import rtsm.io.zeromq as zmod
            monkeypatch.setattr(zmod.time, "time_ns", lambda: 777)
            ts, *_ = sub._parse_rtabmap_pose({"T_wc": [0, 0, 0, 0, 0, 0]})
            assert ts == 777                                                              # nothing on the sensor clock yet
            sub.fw.watermark = 4_000_000_000
            ts, *_ = sub._parse_rtabmap_pose({"T_wc": [0, 0, 0, 0, 0, 0]})
            assert ts == 4_000_000_000                                                    # newest camera frame
            sub._last_pose_ts_ns = 3_000_000_000
            ts, *_ = sub._parse_rtabmap_pose({"T_wc": [0, 0, 0, 0, 0, 0]})
            assert ts == 3_000_000_000                                                    # last tracking stamp wins
            # ...unless the image that tracking pose was PAIRED with (nearest camera
            # stamp, here 2.99 s for a 3.0 s pose) was already enqueued and a newer frame exists
            sub._handle_camera_rgbd(self._camera_msg(2_990_000_000)); sub._handle_camera_rgbd(self._camera_msg(4_000_000_000))
            sub._last_enq_cam_ts = 2_990_000_000
            ts, *_ = sub._parse_rtabmap_pose({"T_wc": [0, 0, 0, 0, 0, 0]})
            assert ts == 4_000_000_000                                                    # then the newest frame
            sub._last_enq_cam_ts = None
            ts, *_ = sub._parse_rtabmap_pose({"T_wc": [0, 0, 0, 0, 0, 0]})
            assert ts == 3_000_000_000                                                    # not enqueued -> tracking stamp
            ts, *_ = sub._parse_rtabmap_pose({"stamp_ms": 6_000, "T_wc": [0, 0, 0, 0, 0, 0]})
            assert ts == 6_000_000_000                                                    # own stamp always wins
        finally:
            sub.close()

    def test_no_sink_is_safe(self):
        sub = self._sub(None)
        try:
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=1, T_wc=[0, 0, 0, 0, 0, 0]))
        finally:
            sub.close()
