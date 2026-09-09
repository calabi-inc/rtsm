"""Admit before decode (Gate 4.5 plan, P1 task 2): the queue-capacity decision
and the non-keyframe throttle run before the RGB decode; depth decodes for
every tracking-normal frame; the viz broadcast happens only for admitted
frames; the replayer fires the pose sink; ZeroMQ buffers encoded bytes and
decodes once per stamp after admission. CPU-only.
"""
from __future__ import annotations

import json
import struct
import time

import cv2
import numpy as np
import pytest

from rtsm.io import websocket as ws_mod
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver


def _ws_frame(frame_id: int, timestamp_ns: int, confidence: np.ndarray | None = None, **overrides) -> bytes:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", rgb)
    depth = np.ones((16, 16), dtype=np.uint16) * 1500
    depth[0, :4] = 0                                    # 4 invalid pixels of 256 -> valid frac 0.984375
    header = {
        "frame_id": frame_id, "timestamp_ns": timestamp_ns, "unix_timestamp": 1700000000.0,
        "rgb_format": "jpeg", "rgb_width": 16, "rgb_height": 16,
        "depth_format": "uint16_mm", "depth_width": 16, "depth_height": 16, "depth_scale": 0.001,
        "fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0,
        "pose_format": "quat_translation", "T_wc": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "tracking_state": "normal",
    }
    if confidence is not None:
        header.update({"confidence_format": "uint8", "confidence_width": 16, "confidence_height": 16})
    header.update(overrides)
    hj = json.dumps(header).encode("utf-8"); rj = jpeg.tobytes(); dj = depth.tobytes()
    msg = (struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj + struct.pack("<I", len(dj)) + dj)
    if confidence is not None:
        cj = confidence.astype(np.uint8).tobytes()
        msg += struct.pack("<I", len(cj)) + cj
    return msg


@pytest.fixture
def count_rgb_decodes(monkeypatch):
    calls = {"n": 0}
    real = ws_mod.decode_rgb

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(ws_mod, "decode_rgb", counting)
    return calls


class TestWebSocketAdmitBeforeDecode:
    def test_full_queue_drops_before_rgb_decode(self, count_rgb_decodes):
        events = []
        q = IngestQueue(maxsize=1)
        recv = WebSocketReceiver(ingest_queue=q, keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 event_sink=events.append)
        pkt1 = recv._parse_binary_message(_ws_frame(1, 1_000))          # keyframe, decoded
        assert pkt1 is not None and count_rgb_decodes["n"] == 1
        assert q.put(pkt1) is True and q.full()
        assert recv._parse_binary_message(_ws_frame(2, 2_000)) is None  # queue full -> refused before decode
        assert count_rgb_decodes["n"] == 1                               # no second RGB decode
        e = events[-1]
        assert (e.decision, e.reason, e.frame_seq, e.is_keyframe) == ("dropped", "queue_full", 2, False)
        assert e.depth_valid_frac == pytest.approx(252 / 256)           # depth WAS decoded for the dropped frame

    def test_keyframes_are_subject_to_the_same_capacity_check(self, count_rgb_decodes):
        q = IngestQueue(maxsize=1)
        recv = WebSocketReceiver(ingest_queue=q, keyframe_every_n=1, nonkf_min_interval_s=0.0)
        q.put(recv._parse_binary_message(_ws_frame(1, 1_000)))
        assert recv._parse_binary_message(_ws_frame(2, 2_000)) is None
        assert count_rgb_decodes["n"] == 1

    def test_throttled_frame_carries_depth_stats_and_skips_rgb(self, count_rgb_decodes):
        events = []
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000,
                                 nonkf_min_interval_s=10.0, event_sink=events.append)
        assert recv._parse_binary_message(_ws_frame(1, 1_000)) is not None
        assert recv._parse_binary_message(_ws_frame(2, 2_000)) is not None      # first non-KF admitted
        assert recv._parse_binary_message(_ws_frame(3, 3_000)) is None          # throttled
        thr = [e for e in events if e.reason == "throttle"]
        assert len(thr) == 1 and thr[0].depth_valid_frac == pytest.approx(252 / 256)
        assert count_rgb_decodes["n"] == 2                                       # frames 1 and 2 only

    def test_no_trace_sink_means_no_depth_stat_work(self, monkeypatch):
        calls = {"n": 0}
        real = ws_mod.depth_valid_fraction

        def counting(d):
            calls["n"] += 1
            return real(d)
        monkeypatch.setattr(ws_mod, "depth_valid_fraction", counting)

        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.0)
        pkt = recv._parse_binary_message(_ws_frame(1, 1_000))
        assert pkt is not None and pkt.depth_m is not None                       # depth still decoded for the packet
        assert calls["n"] == 0                                                   # ...but no statistic without a sink
        events = []
        recv2 = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                  event_sink=events.append)
        assert recv2._parse_binary_message(_ws_frame(1, 1_000)) is not None
        assert calls["n"] >= 1

    def test_enqueued_and_throttled_lines_share_the_pre_filter_statistic(self):
        """depth_valid_frac is ONE statistic: the finite fraction before the
        confidence filter, on enqueued lines too (the packet's depth has been
        NaN-masked by then)."""
        conf = np.zeros((16, 16), dtype=np.uint8)
        conf[:8, :] = 2                                  # top half confident, bottom half below threshold 2
        events = []
        q = IngestQueue(maxsize=8)
        recv = WebSocketReceiver(ingest_queue=q, keyframe_every_n=1000, nonkf_min_interval_s=10.0,
                                 confidence_threshold=2, event_sink=events.append)
        pkt1 = recv._parse_binary_message(_ws_frame(1, 1_000, confidence=conf))
        pkt2 = recv._parse_binary_message(_ws_frame(2, 2_000, confidence=conf))          # admitted non-KF
        assert recv._parse_binary_message(_ws_frame(3, 3_000, confidence=conf)) is None  # throttled
        for pkt in (pkt1, pkt2):
            assert q.put(pkt) is True
            recv._trace_rx("enqueued", "", pkt=pkt)      # what the stream loop / replayer do after put()
        post_filter = float(np.isfinite(pkt2.depth_m).mean())
        assert post_filter < 0.6                         # the confidence filter really masked the bottom half
        by_dec = {(e.decision, e.reason): e.depth_valid_frac for e in events}
        assert by_dec[("enqueued", "")] == pytest.approx(252 / 256)
        assert by_dec[("dropped", "throttle")] == pytest.approx(252 / 256)
        assert all(e.depth_valid_frac == pytest.approx(252 / 256) for e in events)

    def test_admission_queue_overrides_the_dummy(self, count_rgb_decodes):
        real_q = IngestQueue(maxsize=1)
        real_q.put(object())                                                     # full
        recv = WebSocketReceiver(ingest_queue=IngestQueue(1), keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 admission_queue=real_q)
        assert recv._parse_binary_message(_ws_frame(1, 1_000)) is None
        assert count_rgb_decodes["n"] == 0


class TestReplayerAdmission:
    def _record(self, tmp_path, n: int):
        from rtsm.io.recorder import SessionRecorder
        rec_dir = tmp_path / "rec"
        rec = SessionRecorder(output_dir=str(rec_dir))
        for i in range(1, n + 1):
            rec.on_message("binary", _ws_frame(i, i * 1000))
            time.sleep(0.01)
        rec.on_handshake({"type": "hello", "session_id": "s1"}, {"type": "hello_ack", "status": "ok"})
        rec.close()
        return rec_dir

    def test_broadcast_only_for_admitted_frames_and_pose_sink_for_all(self, tmp_path, count_rgb_decodes):
        from rtsm.io.replayer import ReplayReceiver

        rec_dir = self._record(tmp_path, 3)
        broadcasts, poses, events = [], [], []
        q = IngestQueue(maxsize=1)                       # room for exactly one frame; nobody drains it
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=q, keyframe_every_n=30,
                            nonkf_min_interval_s=0.0, replay_speed=10.0,
                            on_camera_frame=broadcasts.append, pose_sink=lambda *a: poses.append(a),
                            event_sink=events.append)
        rr.start()
        assert rr.wait(10.0)
        assert [(e.decision, e.reason, e.frame_seq) for e in events] == [
            ("enqueued", "", 1), ("dropped", "queue_full", 2), ("dropped", "queue_full", 3)]
        assert len(broadcasts) == 1                      # viz saw the admitted frame only
        assert count_rgb_decodes["n"] == 1               # the two refused frames were never RGB-decoded
        assert len(poses) == 3                           # pose sink fired for every tracking-normal frame
        assert all(len(p) == 4 for p in poses)           # (t_wc, q_xyzw, unix_ts, frame_epoch)

    def test_bad_depth_frame_does_not_hang_the_replayer(self, tmp_path, monkeypatch):
        """The depth decode now runs before the throttle / admission for every
        tracking-normal frame; a frame whose depth payload raises must surface
        as a parse_error line and the replay must still finish (wait() returns)."""
        from rtsm.io.replayer import ReplayReceiver

        rec_dir = self._record(tmp_path, 3)
        calls = {"n": 0}
        real = ws_mod.decode_depth

        def flaky(*a, **kw):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("corrupt depth payload")
            return real(*a, **kw)
        monkeypatch.setattr(ws_mod, "decode_depth", flaky)

        events = []
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=30,
                            nonkf_min_interval_s=0.0, replay_speed=10.0, event_sink=events.append)
        rr.start()
        assert rr.wait(10.0)                             # did not hang on the exception
        assert [(e.decision, e.reason, e.frame_seq) for e in events] == [
            ("enqueued", "", 1), ("dropped", "parse_error", 2), ("enqueued", "", 3)]


class TestZeroMQEncodedWindow:
    def _sub(self, q):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber
        return ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                                ingest_queue=q)

    def _camera_msg(self, ts_ns: int):
        rgb = np.full((8, 8, 3), 90, dtype=np.uint8)
        _, jpg = cv2.imencode(".jpg", rgb)
        depth = (np.ones((8, 8), dtype=np.uint16) * 1234)
        _, png = cv2.imencode(".png", depth)
        meta = {"ts_ns": ts_ns, "intrinsics": {"fx": 5.0, "fy": 5.0, "cx": 4.0, "cy": 4.0, "width": 8, "height": 8},
                "depth_units_m": 0.001}
        return [b"camera.rgbd", json.dumps(meta).encode("utf-8"), jpg.tobytes(), png.tobytes()]

    def test_window_holds_bytes_and_decodes_once_after_admission(self, monkeypatch):
        q = IngestQueue(maxsize=8)
        sub = self._sub(q)
        try:
            sub._nonkf_min_interval_s = 0.0
            ts = 5_000_000_000
            sub._handle_camera_rgbd(self._camera_msg(ts))
            assert isinstance(sub.fw.rgb[ts], (bytes, bytearray))          # encoded, not an array
            png_bytes, units = sub.fw.depth[ts]
            assert isinstance(png_bytes, (bytes, bytearray)) and units == pytest.approx(0.001)

            imdecodes = {"n": 0}
            real = cv2.imdecode

            def counting(*a, **kw):
                imdecodes["n"] += 1
                return real(*a, **kw)
            monkeypatch.setattr(cv2, "imdecode", counting)

            t = np.zeros(3, dtype=np.float32); qq = np.array([0, 0, 0, 1], dtype=np.float32)
            sub._try_enqueue_frame(ts, t, qq, is_keyframe=False)             # JPEG + PNG decode
            assert imdecodes["n"] == 2
            sub._try_enqueue_frame(ts, t, qq, is_keyframe=True)              # same stamp, keyframe: memo hit
            assert imdecodes["n"] == 2                                       # no further imdecode
            assert q.qsize() == 2 and len(sub._decode_cache) == 1
            pkt = q.get()
            assert pkt.rgb.shape == (8, 8, 3) and pkt.depth_m.shape == (8, 8)
            assert pkt.depth_m[0, 0] == pytest.approx(1.234)
        finally:
            sub.close()

    def test_full_queue_skips_the_decode(self, monkeypatch):
        q = IngestQueue(maxsize=1)
        q.put(object())
        sub = self._sub(q)
        try:
            ts = 7_000_000_000
            sub._handle_camera_rgbd(self._camera_msg(ts))
            calls = {"n": 0}
            monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or None)
            events = []
            sub._event_sink = events.append
            sub._try_enqueue_frame(ts, np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32),
                                   is_keyframe=True)
            assert calls["n"] == 0                                           # no JPEG/PNG decode for a refused frame
            assert [(e.decision, e.reason) for e in events] == [("dropped", "queue_full")]
        finally:
            sub.close()

    def test_refused_nonkf_still_burns_the_throttle_window(self):
        """Attempt-based throttle semantics, as on the websocket path: with a
        full queue, 30 Hz poses inside one 0.5 s window produce ONE queue_full
        line, the rest are throttle lines (not 15 queue drops)."""
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber
        q = IngestQueue(maxsize=1)
        q.put(object())
        events = []
        sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                               ingest_queue=q, throttle_clock="sensor", event_sink=events.append)
        try:
            sub._nonkf_min_interval_s = 0.5
            t = np.zeros(3, dtype=np.float32); qq = np.array([0, 0, 0, 1], dtype=np.float32)
            base = 9_000_000_000
            for i in range(15):
                ts = base + i * 33_000_000
                sub._handle_camera_rgbd(self._camera_msg(ts))
                sub._try_enqueue_frame(ts, t, qq, is_keyframe=False)
            reasons = [e.reason for e in events]
            assert reasons[0] == "queue_full" and reasons.count("queue_full") == 1
            assert reasons.count("throttle") == 14
        finally:
            sub.close()

    def test_decode_memo_is_keyed_by_the_matched_camera_stamp(self):
        """A pose whose nearest camera frame changes between two calls (a
        newer frame arrived within slop) gets the NEW frame's pixels, and two
        poses within slop of one frame share one decode."""
        q = IngestQueue(maxsize=8)
        sub = self._sub(q)
        try:
            sub._nonkf_min_interval_s = 0.0
            t = np.zeros(3, dtype=np.float32); qq = np.array([0, 0, 0, 1], dtype=np.float32)
            P = 20_000_000_000
            c1, c2 = P - 17_000_000, P + 16_000_000
            msg1 = self._camera_msg(c1); msg2 = self._camera_msg(c2)
            # make frame 2 distinguishable: brighter JPEG
            _, jpg2 = cv2.imencode(".jpg", np.full((8, 8, 3), 200, dtype=np.uint8)); msg2[2] = jpg2.tobytes()
            sub._handle_camera_rgbd(msg1)
            sub._try_enqueue_frame(P, t, qq, is_keyframe=False)             # pairs with c1
            sub._handle_camera_rgbd(msg2)
            sub._try_enqueue_frame(P, t, qq, is_keyframe=True)              # now nearest is c2
            a, b = q.get(), q.get()
            assert int(a.rgb[0, 0, 0]) < 120 < int(b.rgb[0, 0, 0])          # different pixels, not a stale memo hit
            assert set(sub._decode_cache) == {c1, c2}
        finally:
            sub.close()

    def test_window_defaults_are_bounded(self):
        sub = self._sub(IngestQueue(maxsize=8))
        try:
            assert sub.fw.max == 90 and sub.fw.ttl_ns == 2 * 10**9          # TTL binds up to 45 fps
        finally:
            sub.close()
