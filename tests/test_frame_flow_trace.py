"""Frame-flow trace (Gate 4.5 plan, P1 task 0): every receiver decision and every
dequeued frame leaves a line in the diagnostic event log; gate_acceptance_rate is
computed, not hard-coded. Everything here runs CPU-only, no models, no sockets
beyond a ZeroMQ SUB connect to a port nobody listens on.
"""
from __future__ import annotations

import json
import struct
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from rtsm.analytics.latency_analytics import FrameTimingStats, PipelineLatencyBuffer
from rtsm.core.datamodel import FramePacket, PoseStamped, TimeBundle
from rtsm.core.ingest_gate import IngestDecision
from rtsm.core.pipeline import Pipeline
from rtsm.evaluation.event_log import (
    SCHEMA_VERSION, DequeueEvent, EventLogWriter, FrameEvent, ReceiverEvent,
)
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver


# ───────────────────────────── helpers ─────────────────────────────

def _lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _writer(tmp_path: Path) -> EventLogWriter:
    return EventLogWriter(enabled=True, configured_path=str(tmp_path / "events.jsonl"))


def _pose() -> PoseStamped:
    return PoseStamped(stamp_ns=0, frame_id="arkit",
                       t_wc=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                       q_wc_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))


def _packet(*, seq: int = 7, ts_ns: int = 123_000_000, is_kf: bool = False,
            t_mono: float = 0.0, pose=None) -> FramePacket:
    return FramePacket(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.ones((8, 8), dtype=np.float32),
        pose=_pose() if pose is None else pose,
        intr=None,
        is_keyframe=is_kf,
        time=TimeBundle(t_mono_s=t_mono, t_wall_utc_s=0.0, t_sensor_ns=ts_ns, seq=seq),
    )


class _GateStub:
    """IngestGate stand-in: returns a scripted decision, records the calls."""

    def __init__(self, accept: bool, reason: str):
        self._dec = IngestDecision(accept, reason)
        self.calls = 0

    def should_accept(self, **_kw) -> IngestDecision:
        self.calls += 1
        return self._dec


class _SweepCacheStub:
    def cell_and_vbin_from_pose(self, *, twc_xyz, q_wc_xyzw):
        return (0, 0, 0), 0, np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _accept_all_frames(pipe: Pipeline) -> None:
    """Neutralise the frame-quality gate (synthetic frames are black)."""
    class _Ok:
        accept, reason = True, ""
    pipe.frame_gate.check = lambda rgb, depth: _Ok()     # type: ignore[assignment]
    pipe.frame_gate.maybe_log = lambda d: None           # type: ignore[assignment]


def _pipeline(q: IngestQueue, gate, event_log: EventLogWriter) -> Pipeline:
    return Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                    associator=None, ingest_gate=gate, ingest_q=q,
                    sweep_cache=_SweepCacheStub(), event_log=event_log)


def _ws_frame(**overrides) -> bytes:
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", rgb)
    depth = (np.ones((16, 16), dtype=np.uint16) * 1500).tobytes()
    header = {
        "frame_id": 11, "timestamp_ns": 5_000_000_000, "unix_timestamp": 1700000000.0,
        "rgb_format": "jpeg", "rgb_width": 16, "rgb_height": 16,
        "depth_format": "uint16_mm", "depth_width": 16, "depth_height": 16, "depth_scale": 0.001,
        "fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0,
        "pose_format": "quat_translation", "T_wc": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "tracking_state": "normal",
    }
    header.update(overrides)
    hj = json.dumps(header).encode("utf-8")
    rj = jpeg.tobytes()
    return (struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj
            + struct.pack("<I", len(depth)) + depth)


# ───────────────────────────── writer ─────────────────────────────

class TestEventLogWriter:
    def test_disabled_is_a_noop_and_has_no_sink(self, tmp_path):
        w = EventLogWriter(enabled=False, configured_path=str(tmp_path / "never.jsonl"))
        assert w.sink() is None
        w.write({"kind": "x"})
        w.write(ReceiverEvent(timestamp=0.0, source="websocket", decision="dropped", reason="throttle"))
        w.close()
        assert not (tmp_path / "never.jsonl").exists()

    def test_meta_line_then_typed_lines(self, tmp_path):
        w = _writer(tmp_path)
        assert w.sink() is not None
        w.write(ReceiverEvent(timestamp=1.0, source="replay", decision="enqueued", reason="",
                              frame_seq=3, t_sensor_ns=30, is_keyframe=True, frame_count=1, queue_depth=1))
        w.write(DequeueEvent(timestamp=2.0, frame_seq=3, t_sensor_ns=30, is_keyframe=True, queue_wait_s=0.5,
                             queue_depth=0, outcome="processed", reason="keyframe"))
        w.write(FrameEvent(timestamp=3.0, frame_seq=3, is_keyframe=True, n_masks_raw=4, t_sensor_ns=30))
        w.close()
        rows = _lines(w.path)
        assert [r["kind"] for r in rows] == ["meta", "receiver", "dequeue", "frame"]
        assert rows[0]["schema_version"] == SCHEMA_VERSION
        assert rows[1]["decision"] == "enqueued" and rows[1]["source"] == "replay"
        assert rows[2]["outcome"] == "processed" and rows[2]["queue_wait_s"] == 0.5
        assert rows[3]["t_sensor_ns"] == 30

    def test_concurrent_writers_keep_lines_intact(self, tmp_path):
        w = _writer(tmp_path)

        def worker(tag: str):
            for i in range(300):
                w.write({"kind": "t", "tag": tag, "i": i, "pad": "x" * 200})

        ts = [threading.Thread(target=worker, args=(f"w{k}",)) for k in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        w.close()
        rows = _lines(w.path)          # json.loads on every line = no interleaving
        assert len(rows) == 1 + 4 * 300


# ───────────────────────────── pipeline: dequeue lines ─────────────────────────────

class TestDequeueTrace:
    def test_gate_rejection_writes_reason_and_age(self, tmp_path):
        w = _writer(tmp_path)
        q = IngestQueue()
        pipe = _pipeline(q, _GateStub(False, "ttl_not_expired"), w)
        q.put(_packet(seq=41, ts_ns=41_000, is_kf=False, t_mono=0.0))
        pipe.run_one_step()                       # returns at the ingest gate
        w.close()
        deq = [r for r in _lines(w.path) if r["kind"] == "dequeue"]
        assert len(deq) == 1
        r = deq[0]
        assert r["outcome"] == "gate_rejected" and r["reason"] == "ttl_not_expired"
        assert r["frame_seq"] == 41 and r["t_sensor_ns"] == 41_000 and r["is_keyframe"] is False
        assert r["queue_wait_s"] > 0.0                   # arrival t_mono 0.0 vs a live monotonic dequeue
        assert r["queue_depth"] == 0

    def test_frame_gate_rejection_writes_frame_rejected(self, tmp_path):
        w = _writer(tmp_path)
        q = IngestQueue()
        pipe = _pipeline(q, _GateStub(True, "keyframe"), w)

        class _Reject:
            accept, reason = False, "dark"

        pipe.frame_gate.check = lambda rgb, depth: _Reject()   # type: ignore[assignment]
        pipe.frame_gate.maybe_log = lambda d: None             # type: ignore[assignment]
        q.put(_packet(seq=5, is_kf=True))
        pipe.run_one_step()
        w.close()
        deq = [r for r in _lines(w.path) if r["kind"] == "dequeue"]
        assert [(r["outcome"], r["reason"]) for r in deq] == [("frame_rejected", "dark")]

    def test_accepted_frame_is_traced_before_processing(self, tmp_path):
        """The 'processed' line is written before segmentation starts, so a
        crash inside processing still leaves the dequeue record. With
        segmenter=None the step raises right after the trace."""
        w = _writer(tmp_path)
        q = IngestQueue()
        pipe = _pipeline(q, _GateStub(True, "keyframe"), w)
        _accept_all_frames(pipe)                  # the synthetic frame is black; keep the quality gate out of it
        q.put(_packet(seq=9, ts_ns=9_000, is_kf=True))
        with pytest.raises(AttributeError, match="segment"):
            pipe.run_one_step()
        w.close()
        deq = [r for r in _lines(w.path) if r["kind"] == "dequeue"]
        assert [(r["outcome"], r["reason"], r["frame_seq"]) for r in deq] == [("processed", "keyframe", 9)]

    def test_no_pose_frame_is_labelled_no_pose(self, tmp_path):
        w = _writer(tmp_path)
        q = IngestQueue()
        gate = _GateStub(False, "would_reject")
        pipe = _pipeline(q, gate, w)
        _accept_all_frames(pipe)
        pkt = FramePacket(rgb=np.zeros((8, 8, 3), dtype=np.uint8), depth_m=None, pose=None, intr=None,
                          is_keyframe=False, time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=1, seq=2))
        q.put(pkt)
        with pytest.raises(AttributeError, match="segment"):            # pose-less frames bypass the gate and go on to (absent) segmentation
            pipe.run_one_step()
        w.close()
        deq = [r for r in _lines(w.path) if r["kind"] == "dequeue"]
        assert deq and deq[0]["outcome"] == "processed" and deq[0]["reason"] == "no_pose"
        assert gate.calls == 0

    def test_disabled_log_costs_nothing_and_changes_nothing(self, tmp_path):
        w = EventLogWriter(enabled=False, configured_path=str(tmp_path / "off.jsonl"))
        q = IngestQueue()
        gate = _GateStub(False, "ttl_not_expired")
        pipe = _pipeline(q, gate, w)
        q.put(_packet())
        pipe.run_one_step()
        assert gate.calls == 1
        assert not (tmp_path / "off.jsonl").exists()

    def test_pose_conversion_failure_is_a_dropped_dequeue_line(self, tmp_path):
        """A present-but-corrupt pose is dropped inside _get_snapshot_via_queue
        (PR #23). It WAS dequeued, so the trace must say so, or the receiver
        and dequeue sequences drift apart."""
        class _RaisingPose:
            t_wc = np.array([0.0, 0.0, 0.0], dtype=np.float32)

            def T_wc(self):
                raise ValueError("corrupt pose payload")

        w = _writer(tmp_path)
        q = IngestQueue()
        gate = _GateStub(True, "keyframe")
        pipe = _pipeline(q, gate, w)
        q.put(_packet(seq=13, ts_ns=13_000, is_kf=True, pose=_RaisingPose()))
        pipe.run_one_step()                       # dropped before the gates; no exception
        w.close()
        deq = [r for r in _lines(w.path) if r["kind"] == "dequeue"]
        assert [(r["outcome"], r["reason"], r["frame_seq"], r["t_sensor_ns"]) for r in deq] == [
            ("dropped", "pose_conversion_failed", 13, 13_000)]
        assert pipe.pose_conversion_failures == 1 and gate.calls == 0

    def test_pipeline_builds_its_own_writer_when_none_is_passed(self):
        q = IngestQueue()
        pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=None, proximity_index=None,
                        associator=None, ingest_gate=None, ingest_q=q)
        assert pipe._event_log.enabled is False


# ───────────────────────────── receivers ─────────────────────────────

class TestReceiverTrace:
    def _recv(self, sink, **kw) -> WebSocketReceiver:
        return WebSocketReceiver(ingest_queue=IngestQueue(maxsize=4), keyframe_every_n=30,
                                 nonkf_min_interval_s=kw.pop("nonkf_min_interval_s", 0.5),
                                 event_sink=sink, **kw)

    def test_no_sink_means_no_events_and_same_parse_result(self):
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=4), keyframe_every_n=30)
        assert recv._event_sink is None
        assert recv._parse_binary_message(_ws_frame()) is not None
        assert recv._parse_binary_message(b"\x00\x00") is None

    def test_malformed_before_header_has_no_seq(self):
        events = []
        recv = self._recv(events.append)
        assert recv._parse_binary_message(b"\x00\x00") is None
        assert len(events) == 1
        e = events[0]
        assert (e.decision, e.reason, e.frame_seq, e.t_sensor_ns, e.source) == ("dropped", "malformed", None, None, "websocket")

    def test_tracking_state_drop_carries_header_ids(self):
        events = []
        recv = self._recv(events.append)
        assert recv._parse_binary_message(_ws_frame(tracking_state="limited", frame_id=77, timestamp_ns=777)) is None
        assert [(e.decision, e.reason, e.frame_seq, e.t_sensor_ns) for e in events] == [("dropped", "tracking_state", 77, 777)]

    def test_throttle_drop_is_traced_with_frame_count(self):
        events = []
        recv = self._recv(events.append, nonkf_min_interval_s=10.0)
        assert recv._parse_binary_message(_ws_frame(frame_id=1)) is not None     # frame 1 = keyframe, decoded
        import time as _t
        recv._last_nonkf_enq_mono = _t.monotonic()                              # as if the KF had just been enqueued
        assert recv._parse_binary_message(_ws_frame(frame_id=2, timestamp_ns=2)) is None
        thr = [e for e in events if e.reason == "throttle"]
        assert len(thr) == 1
        assert (thr[0].decision, thr[0].frame_seq, thr[0].is_keyframe, thr[0].frame_count) == ("dropped", 2, False, 2)

    def test_decoded_frames_emit_nothing_at_parse_time(self):
        """The enqueue decision belongs to the caller (stream loop / replayer),
        so a successful parse must not pre-empt it with an event."""
        events = []
        recv = self._recv(events.append)
        assert recv._parse_binary_message(_ws_frame()) is not None
        assert events == []

    def test_event_source_label_and_queue_depth_override(self):
        events = []
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=4), event_sink=events.append, event_source="replay")
        pkt = recv._parse_binary_message(_ws_frame(frame_id=3, timestamp_ns=33))
        recv._trace_rx("enqueued", "", pkt=pkt, queue_depth=17)
        e = events[-1]
        assert (e.source, e.decision, e.frame_seq, e.t_sensor_ns, e.is_keyframe, e.queue_depth) == ("replay", "enqueued", 3, 33, True, 17)

    def test_parse_error_after_header_carries_ids_and_reraises(self):
        """A malformed T_wc raises out of the parser exactly as before, but now
        leaves one 'parse_error' line with the header ids and frame_count."""
        events = []
        recv = self._recv(events.append)
        with pytest.raises(Exception):
            recv._parse_binary_message(_ws_frame(frame_id=21, timestamp_ns=2100, T_wc=[1.0, 2.0, 3.0]))
        assert len(events) == 1
        e = events[0]
        assert (e.decision, e.reason, e.frame_seq, e.t_sensor_ns, e.frame_count) == ("dropped", "parse_error", 21, 2100, 0)

    def test_non_object_header_is_malformed_not_a_crash(self):
        events = []
        recv = self._recv(events.append)
        hj = json.dumps([1, 2, 3]).encode("utf-8")
        msg = struct.pack("<I", len(hj)) + hj + b"\x00\x00"        # list header + truncated body
        assert recv._parse_binary_message(msg) is None
        assert [(e.decision, e.reason, e.frame_seq) for e in events] == [("dropped", "malformed", None)]

    def test_sink_errors_never_break_the_receive_path(self):
        def boom(_e):
            raise RuntimeError("sink down")
        recv = self._recv(boom)
        assert recv._parse_binary_message(b"\x00\x00") is None
        assert recv._parse_binary_message(_ws_frame()) is not None


class TestReplayTrace:
    def test_replay_lines_are_tagged_and_read_the_real_queue(self, tmp_path):
        """Three recorded frames: frame 1 is a keyframe (enqueued), frame 2 a
        non-KF (enqueued), frame 3 a non-KF inside the 0.5 s throttle window
        (dropped). Lines carry source='replay' and the REAL queue's depth."""
        from rtsm.io.recorder import SessionRecorder
        from rtsm.io.replayer import ReplayReceiver

        rec_dir = tmp_path / "rec"
        rec = SessionRecorder(output_dir=str(rec_dir))
        for i in (1, 2, 3):
            rec.on_message("binary", _ws_frame(frame_id=i, timestamp_ns=i * 1000))
        rec.on_handshake({"type": "hello", "session_id": "s1"}, {"type": "hello_ack", "status": "ok"})
        rec.close()

        events = []
        q = IngestQueue(maxsize=8)
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=q, keyframe_every_n=30,
                            nonkf_min_interval_s=0.5, event_sink=events.append, replay_speed=10.0)
        rr.start()
        assert rr._done.wait(10.0), "replay did not finish"
        got = [(e.source, e.decision, e.reason, e.frame_seq, e.is_keyframe, e.queue_depth) for e in events]
        assert got == [
            ("replay", "enqueued", "", 1, True, 1),
            ("replay", "enqueued", "", 2, False, 2),
            ("replay", "dropped", "throttle", 3, False, 2),
        ]
        assert q.qsize() == 2


class TestZeroMQTrace:
    def test_enqueue_decisions_are_traced(self):
        zmq = pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber

        events = []
        q = IngestQueue(maxsize=1)
        sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                               ingest_queue=q, event_sink=events.append)
        try:
            sub._nonkf_min_interval_s = 0.0
            rgb = np.zeros((4, 4, 3), dtype=np.uint8)
            depth = np.ones((4, 4), dtype=np.float32)

            class _FW:
                def __init__(self):
                    self.has = True

                def assemble_pair(self, ts_ns):
                    return (rgb, depth, None) if self.has else (None, None, None)

            sub.fw = _FW()
            t = np.zeros(3, dtype=np.float32)
            qq = np.array([0, 0, 0, 1], dtype=np.float32)
            sub._try_enqueue_frame(100, t, qq, is_keyframe=True)      # enqueued (queue now full)
            sub._try_enqueue_frame(101, t, qq, is_keyframe=True)      # queue_full
            sub._try_enqueue_frame(100, t, qq, is_keyframe=False)     # duplicate_ts (last enqueued ts)
            sub.fw.has = False
            sub._try_enqueue_frame(102, t, qq, is_keyframe=False)     # no_camera_frame
            got = [(e.source, e.decision, e.reason, e.t_sensor_ns, e.is_keyframe) for e in events]
            assert got == [
                ("zeromq", "enqueued", "", 100, True),
                ("zeromq", "dropped", "queue_full", 101, True),
                ("zeromq", "dropped", "duplicate_ts", 100, False),
                ("zeromq", "dropped", "no_camera_frame", 102, False),
            ]
        finally:
            sub.close()

    def test_malformed_pose_messages_are_traced(self):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber

        events = []
        sub = ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                               ingest_queue=IngestQueue(maxsize=4), event_sink=events.append)
        try:
            sub._handle_tracking_pose([b"rtabmap.tracking_pose"])             # wrong part count
            sub._handle_kf_pose([b"rtabmap.kf_pose", b"not json at all"])     # parse error
            assert [(e.decision, e.reason, e.is_keyframe, e.t_sensor_ns) for e in events] == [
                ("dropped", "malformed", False, None),
                ("dropped", "malformed", True, None),
            ]
        finally:
            sub.close()


# ───────────────────────────── analytics ─────────────────────────────

class TestGateAcceptanceRate:
    def test_rate_and_counters_are_computed_from_lifetime_counts(self):
        buf = PipelineLatencyBuffer(max_frames=50)
        for _ in range(3):
            buf.record_frame_received()
        for i in range(6):
            buf.append(FrameTimingStats(timestamp=float(i)))
        buf.record_gate_rejection(); buf.record_gate_rejection(); buf.record_gate_rejection()
        buf.record_frame_rejection()
        buf.record_throttle_skip(); buf.record_queue_drop(); buf.record_tracking_drop()
        agg = buf.aggregate()
        # 6 processed out of 6 + 3 + 1 dequeued
        assert agg["gate_acceptance_rate"] == pytest.approx(0.6)
        assert agg["counters"] == {
            "received": 3, "processed": 6, "gate_rejections": 3, "frame_rejections": 1,
            "queue_drops": 1, "throttle_skips": 1, "tracking_drops": 1,
            "age_drops": 0,
            "superseded": 0,
        }

    def test_rate_is_zero_when_nothing_was_dequeued(self):
        buf = PipelineLatencyBuffer(max_frames=50)
        agg = buf.aggregate()
        assert agg["frame_count"] == 0 and agg["gate_acceptance_rate"] == 0.0
        assert agg["counters"]["processed"] == 0

    def test_all_rejected_reads_zero_not_one(self):
        buf = PipelineLatencyBuffer(max_frames=50)
        buf.record_gate_rejection()
        assert buf.aggregate()["gate_acceptance_rate"] == 0.0
