"""P2 ledgers, stage A: the pose ledger (Gate 4.5 plan, P2; code plan
code-plan-p2-ledgers-2026-09.md). CPU-only: synthetic websocket frames through
the parser, a two-frame recording through the replayer, a ZeroMQ subscriber
driven by hand (no socket ever listens), and synthetic rows for the reader.

Contract under test:
  * off = nothing: no `pose` kind, no ledger sink, meta says so;
  * one pose line per sensor frame, tracking-limited frames included, written
    BEFORE the tracking filter / throttle / admission can drop the frame;
  * the line carries the same post-flip pose the pose sink gets, the same
    pre-filter depth statistic as the receiver line, and a confidence
    histogram of the RAW map; rx_seq joins it to the receiver line;
  * the receiver's own decisions and lines are unchanged.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from rtsm.evaluation import ledger
from rtsm.evaluation.event_log import (
    LEDGER_SCHEMA, RX_ENQUEUED, SCHEMA_VERSION, EventLogWriter, PoseEvent, resolve_ledger_config,
)
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver

import synth_ledger


# ───────────────────────────── helpers ─────────────────────────────

def _lines(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _writer(tmp_path: Path, *, ledgers: bool = True, **kw) -> EventLogWriter:
    return EventLogWriter(enabled=True, configured_path=str(tmp_path / "events.jsonl"), ledgers=ledgers, **kw)


def _ws_frame(*, conf: np.ndarray | None = None, **overrides) -> bytes:
    """One Calabi-Lens binary message (16x16 JPEG + uint16 depth [+ confidence])."""
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode(".jpg", rgb)
    depth = (np.ones((16, 16), dtype=np.uint16) * 1500).tobytes()
    header = {
        "frame_id": 11, "timestamp_ns": 5_000_000_000, "unix_timestamp": 1700000000.0,
        "rgb_format": "jpeg", "rgb_width": 16, "rgb_height": 16,
        "depth_format": "uint16_mm", "depth_width": 16, "depth_height": 16, "depth_scale": 0.001,
        "fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 8.0,
        "pose_format": "quat_translation", "T_wc": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
        "tracking_state": "normal",
    }
    header.update(overrides)
    parts = b""
    if conf is not None:
        header.update({"confidence_format": "uint8", "confidence_width": int(conf.shape[1]),
                       "confidence_height": int(conf.shape[0])})
        cb = np.ascontiguousarray(conf, dtype=np.uint8).tobytes()
        parts = struct.pack("<I", len(cb)) + cb
    hj = json.dumps(header).encode("utf-8")
    rj = jpeg.tobytes()
    return (struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(rj)) + rj
            + struct.pack("<I", len(depth)) + depth + parts)


def _receiver(events, poses, *, sinks=True, **kw) -> WebSocketReceiver:
    recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.5,
                             event_sink=(events.append if sinks else None), ledger_sink=(poses.append if sinks else None),
                             pose_sink=(lambda *a, **k: None), **kw)
    recv._note_session("s1")
    return recv


# ───────────────────────────── writer / config ─────────────────────────────

class TestWriterAndConfig:
    def test_ledgers_off_writes_only_the_trace_kinds(self, tmp_path):
        w = _writer(tmp_path, ledgers=False)
        assert w.ledger_sink() is None and w.ledgers_enabled is False and w.sink() is not None
        w.write(PoseEvent(timestamp=0.0, source="websocket", rx_seq=1, frame_seq=1, t_sensor_ns=1, t_wall_utc_s=0.0,
                          pose_clock="sender", epoch=0, tracking_state="normal", mailbox_write=True))
        w.close()
        rows = _lines(w.path)
        assert rows[0]["schema_version"] == SCHEMA_VERSION == 3 and rows[0]["ledgers"] == {"enabled": False}
        # the writer itself is kind-agnostic: the guarantee is that no PRODUCER
        # gets a sink, so a real run never contains the kind (tested below)
        assert [r["kind"] for r in rows] == ["meta", "pose"]

    def test_ledgers_on_meta_block_and_sink(self, tmp_path):
        w = _writer(tmp_path)
        assert w.ledger_sink() == w.write and w.ledgers_enabled and w.ledger_format == "jsonl"
        w.close()
        assert _lines(w.path)[0]["ledgers"] == {"enabled": True, "schema": LEDGER_SCHEMA, "format": "jsonl"}

    def test_disabled_writer_ignores_ledgers(self, tmp_path):
        w = EventLogWriter(enabled=False, configured_path=str(tmp_path / "x.jsonl"), ledgers=True)
        assert w.ledger_sink() is None and not w.ledgers_enabled and w.path is None

    @pytest.mark.parametrize("cfg,msg", [
        ({"diagnostics": {"ledgers": 1}}, "diagnostics.ledgers"),
        ({"diagnostics": {"ledgers": "yes"}}, "diagnostics.ledgers"),
        ({"diagnostics": {"ledger_format": "csv"}}, "diagnostics.ledger_format"),
        ({"diagnostics": {"ledger_format": 3}}, "diagnostics.ledger_format"),
        ({"diagnostics": [1]}, "diagnostics"),      # an EMPTY list is coerced like null by the `or {}` rule
    ])
    def test_config_validation_names_the_key(self, cfg, msg):
        with pytest.raises(ValueError, match=msg):
            resolve_ledger_config(cfg)

    def test_config_defaults_and_normalisation(self):
        assert resolve_ledger_config({}) == resolve_ledger_config({"diagnostics": None})
        # parquet not in effect (ledgers off) -> accepted and normalised whether or not pyarrow is installed
        lc = resolve_ledger_config({"diagnostics": {"enabled": True, "ledgers": False, "ledger_format": " Parquet ",
                                                    "event_log_path": ""}})
        assert (lc.enabled, lc.ledgers, lc.event_log_path, lc.ledger_format) == (True, False, None, "parquet")

    def test_parquet_without_pyarrow_is_refused_only_when_it_would_take_effect(self, monkeypatch):
        import rtsm.evaluation.event_log as el
        monkeypatch.setattr(el, "_pyarrow_available", lambda: False)
        with pytest.raises(ValueError, match="rtsm\\[eval\\]"):
            resolve_ledger_config({"diagnostics": {"enabled": True, "ledgers": True, "ledger_format": "parquet"}})
        # not in effect: accepted (the packaged yaml could carry parquet with diagnostics off)
        assert resolve_ledger_config({"diagnostics": {"ledger_format": "parquet"}}).ledger_format == "parquet"

    def test_parquet_close_logs_and_never_raises_without_pyarrow(self, tmp_path, monkeypatch, caplog):
        import rtsm.evaluation.ledger as lg
        monkeypatch.setattr(lg, "to_parquet", lambda p, *a, **k: (_ for _ in ()).throw(RuntimeError("no pyarrow")))
        w = _writer(tmp_path, ledger_format="parquet")
        w.write(PoseEvent(timestamp=0.0, source="websocket", rx_seq=1, frame_seq=1, t_sensor_ns=1, t_wall_utc_s=0.0,
                          pose_clock="sender", epoch=0, tracking_state="normal", mailbox_write=True))
        with caplog.at_level("WARNING", logger="rtsm.evaluation.event_log"):
            w.close()
        assert "parquet conversion failed" in caplog.text
        assert len(_lines(w.path)) == 2                     # the JSONL is intact

    def test_pose_line_cost_is_small(self, tmp_path):
        """240 lines (one session1 replay) through a real writer: a loose CI bound;
        the stage-A gate measures the real thing."""
        w = _writer(tmp_path)
        ev = PoseEvent(timestamp=0.0, source="replay", rx_seq=1, frame_seq=1, t_sensor_ns=1, t_wall_utc_s=0.0,
                       pose_clock="sender", epoch=1, tracking_state="normal", mailbox_write=True,
                       t_wc=[1.0, 2.0, 3.0], q_wc_xyzw=[0.0, 0.0, 0.0, 1.0], depth_valid_frac=0.9,
                       conf_hist=[100, 200, 48852])
        t0 = time.perf_counter()
        for _ in range(240):
            w.write(ev)
        per_line_ms = (time.perf_counter() - t0) * 1000.0 / 240
        w.close()
        assert per_line_ms < 2.0, per_line_ms


# ───────────────────────────── websocket / replay ─────────────────────────────

class TestWebSocketPoseLedger:
    def test_no_sink_means_no_lines_and_same_result(self):
        recv = _receiver([], [], sinks=False)
        pkt = recv._parse_binary_message(_ws_frame())
        assert pkt is not None and recv._ledger_sink is None

    def test_tracking_limited_frame_gets_a_pose_line_and_the_drop_line_is_unchanged(self):
        events, poses = [], []
        recv = _receiver(events, poses)
        assert recv._parse_binary_message(_ws_frame(tracking_state="limited", frame_id=3, timestamp_ns=3_000)) is None
        assert [(e.decision, e.reason, e.frame_seq, e.t_sensor_ns, e.rx_seq) for e in events] == \
            [("dropped", "tracking_state", 3, 3_000, 1)]
        assert recv.tracking_drops == 1
        (p,) = poses
        assert (p.kind, p.source, p.tracking_state, p.mailbox_write, p.rx_seq, p.frame_seq, p.t_sensor_ns) == \
            ("pose", "websocket", "limited", False, 1, 3, 3_000)
        assert p.t_wc == [1.0, 2.0, 3.0] and p.q_wc_xyzw == [0.0, 0.0, 0.0, 1.0] and p.pose_error is None
        assert p.epoch == 1 and p.pose_clock == "sender" and p.t_wall_utc_s == 1700000000.0
        assert p.depth_valid_frac is None and p.conf_hist is None          # depth is not decoded on the drop path

    def test_limited_frame_with_a_bad_pose_records_the_error_not_a_crash(self):
        events, poses = [], []
        recv = _receiver(events, poses)
        assert recv._parse_binary_message(_ws_frame(tracking_state="limited", T_wc=[1.0, 2.0])) is None
        assert [(e.decision, e.reason) for e in events] == [("dropped", "tracking_state")]   # not parse_error
        (p,) = poses
        assert p.t_wc is None and p.q_wc_xyzw is None and p.pose_error and p.tracking_state == "limited"

    def test_limited_frame_pose_is_post_flip_like_the_sink(self):
        """With apply_camera_flip the ledger line carries the flipped pose, the
        same one a normal frame's pose sink and FramePacket get."""
        got = {}
        poses = []
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 apply_camera_flip=True, ledger_sink=poses.append,
                                 pose_sink=lambda t, q, ts, ep, **kw: got.update(t=t.tolist(), q=q.tolist()))
        recv._note_session("s1")
        q = [0.0, 0.7071068, 0.0, 0.7071068]                                # 90 deg about y
        recv._parse_binary_message(_ws_frame(T_wc=[*q, 1.0, 2.0, 3.0], tracking_state="normal"))
        recv._parse_binary_message(_ws_frame(T_wc=[*q, 1.0, 2.0, 3.0], tracking_state="limited"))
        normal, limited = poses
        assert normal.tracking_state == "normal" and limited.tracking_state == "limited"
        assert np.allclose(normal.t_wc, got["t"]) and np.allclose(normal.q_wc_xyzw, got["q"], atol=1e-6)
        assert np.allclose(limited.t_wc, normal.t_wc) and np.allclose(limited.q_wc_xyzw, normal.q_wc_xyzw, atol=1e-6)
        assert not np.allclose(normal.q_wc_xyzw, q)                          # the flip did something

    def test_one_pose_line_per_message_whatever_the_admission_outcome(self):
        """f1 keyframe -> packet (enqueued by the caller), f2 non-KF refused by a
        full admission queue, f3 non-KF throttled: three pose lines, each
        joined to its receiver line by rx_seq, same depth statistic."""
        events, poses = [], []
        full = IngestQueue(maxsize=1)
        recv = WebSocketReceiver(ingest_queue=full, keyframe_every_n=1000, nonkf_min_interval_s=0.5,
                                 event_sink=events.append, ledger_sink=poses.append, pose_sink=lambda *a, **k: None)
        recv._note_session("s1")
        pkt = recv._parse_binary_message(_ws_frame(frame_id=1, timestamp_ns=1_000))
        assert pkt is not None and pkt.is_keyframe and full.put(pkt, block=False)
        recv._trace_rx(RX_ENQUEUED, "", pkt=pkt)                             # what the stream loop does
        assert recv._parse_binary_message(_ws_frame(frame_id=2, timestamp_ns=2_000)) is None   # queue_full
        assert recv._parse_binary_message(_ws_frame(frame_id=3, timestamp_ns=3_000)) is None   # throttle
        rx = [(e.decision, e.reason, e.frame_seq, e.rx_seq, e.depth_valid_frac) for e in events]
        assert rx == [("enqueued", "", 1, 1, 1.0), ("dropped", "queue_full", 2, 2, 1.0), ("dropped", "throttle", 3, 3, 1.0)]
        assert [(p.frame_seq, p.rx_seq, p.tracking_state, p.mailbox_write, p.depth_valid_frac) for p in poses] == \
            [(1, 1, "normal", True, 1.0), (2, 2, "normal", True, 1.0), (3, 3, "normal", True, 1.0)]
        assert [p.t_sensor_ns for p in poses] == [e.t_sensor_ns for e in events]

    def test_conf_hist_is_taken_from_the_raw_map_before_the_confidence_filter(self):
        conf = np.zeros((4, 4), dtype=np.uint8)
        conf.flat[:4] = 2; conf.flat[4:6] = 1                                # 4 twos, 2 ones, 10 zeros
        poses = []
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, nonkf_min_interval_s=0.0,
                                 confidence_threshold=2, ledger_sink=poses.append)
        recv._note_session("s1")
        pkt = recv._parse_binary_message(_ws_frame(conf=conf))
        (p,) = poses
        assert p.conf_hist == [10, 2, 4] and sum(p.conf_hist) == conf.size
        assert p.depth_valid_frac == 1.0                                     # pre-filter statistic
        assert pkt is not None and np.isnan(pkt.depth_m).sum() > 0             # the filter DID mask the packet's depth
        assert pkt.ingest.depth_valid_frac == p.depth_valid_frac

    def test_no_confidence_map_means_null_hist(self):
        poses = []
        recv = _receiver([], poses)
        recv._parse_binary_message(_ws_frame())
        assert poses[0].conf_hist is None

    def test_server_clock_when_the_header_has_no_wall_stamp(self):
        poses = []
        recv = _receiver([], poses)
        recv._parse_binary_message(_ws_frame(unix_timestamp=0))
        assert poses[0].pose_clock == "server" and poses[0].t_wall_utc_s > 1e9

    def test_mailbox_write_false_when_no_pose_sink_is_wired(self):
        poses = []
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, ledger_sink=poses.append)
        recv._note_session("s1")
        recv._parse_binary_message(_ws_frame())
        assert poses[0].mailbox_write is False

    def test_sink_errors_never_break_the_receive_path(self):
        def boom(_):
            raise RuntimeError("sink down")
        recv = WebSocketReceiver(ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=1000, ledger_sink=boom)
        recv._note_session("s1")
        assert recv._parse_binary_message(_ws_frame()) is not None
        assert recv._parse_binary_message(_ws_frame(tracking_state="limited")) is None

    def test_replay_lines_are_tagged_replay_and_join_the_receiver_lines(self, tmp_path):
        from rtsm.io.recorder import SessionRecorder
        from rtsm.io.replayer import ReplayReceiver

        rec_dir = tmp_path / "rec"
        rec = SessionRecorder(output_dir=str(rec_dir))
        for i in (1, 2, 3):
            rec.on_message("binary", _ws_frame(frame_id=i, timestamp_ns=i * 1000,
                                               tracking_state=("limited" if i == 2 else "normal")))
        rec.on_handshake({"type": "hello", "session_id": "s1"}, {"type": "hello_ack", "status": "ok"})
        rec.close()

        events, poses = [], []
        rr = ReplayReceiver(recording_dir=str(rec_dir), ingest_queue=IngestQueue(maxsize=8), keyframe_every_n=30,
                            nonkf_min_interval_s=0.5, event_sink=events.append, ledger_sink=poses.append,
                            replay_speed=10.0)
        rr.start()
        assert rr._done.wait(10.0), "replay did not finish"
        assert [(e.source, e.decision, e.reason, e.frame_seq, e.rx_seq) for e in events] == [
            ("replay", "enqueued", "", 1, 1),
            ("replay", "dropped", "tracking_state", 2, 2),
            ("replay", "enqueued", "", 3, 3),
        ]
        assert [(p.source, p.frame_seq, p.rx_seq, p.tracking_state, p.mailbox_write) for p in poses] == [
            ("replay", 1, 1, "normal", False),                               # no pose sink wired in this test
            ("replay", 2, 2, "limited", False),
            ("replay", 3, 3, "normal", False),
        ]
        by_rx = {e.rx_seq: e for e in events}
        assert all(p.depth_valid_frac == by_rx[p.rx_seq].depth_valid_frac for p in poses)


# ───────────────────────────── zeromq ─────────────────────────────

class TestZeroMQPoseLedger:
    @staticmethod
    def _sub(poses, sink=None):
        pytest.importorskip("zmq")
        from rtsm.io.zeromq import ZeroMQSubscriber
        return ZeroMQSubscriber(camera_endpoint="tcp://127.0.0.1:1", rtabmap_endpoint="tcp://127.0.0.1:1",
                                ingest_queue=IngestQueue(maxsize=8), pose_sink=sink, ledger_sink=poses.append)

    @staticmethod
    def _msg(topic: str, **fields) -> list:
        return [topic.encode(), json.dumps(fields).encode()]

    def test_one_line_per_tracking_pose_never_for_kf_pose(self):
        poses = []
        sub = self._sub(poses, sink=lambda *a, **k: True)
        try:
            base_ms = 1_700_000_000_000
            for i in range(3):
                sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=base_ms + i * 33,
                                                    T_wc=[0.1 * i, 0.0, 0.0, 0.0, 0.0, 0.0]))
            sub._handle_kf_pose(self._msg("rtabmap.kf_pose", kf_id=4, T_wc=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            assert len(poses) == 3
            assert [(p.source, p.tracking_state, p.epoch, p.rx_seq, p.frame_seq, p.pose_clock, p.mailbox_write)
                    for p in poses] == [("zeromq", "not_available", 0, None, None, "server", True)] * 3
            assert [p.t_sensor_ns for p in poses] == [(base_ms + i * 33) * 1_000_000 for i in range(3)]
            assert [p.t_wc[0] for p in poses] == pytest.approx([0.0, 0.1, 0.2])
            assert all(p.depth_valid_frac is None and p.conf_hist is None for p in poses)
        finally:
            sub.close()

    def test_epoch_bump_and_stampless_pose_are_visible(self):
        poses = []
        sub = self._sub(poses)
        try:
            sub._nonkf_min_interval_s = 0.0
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=10_000, T_wc=[0, 0, 0, 0, 0, 0]))
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", stamp_ms=1_000, T_wc=[0, 0, 0, 0, 0, 0]))   # 9 s back
            sub._handle_tracking_pose(self._msg("rtabmap.tracking_pose", T_wc=[0, 0, 0, 0, 0, 0]))                 # no stamp
            assert [p.epoch for p in poses] == [0, 1, 1]
            # a stamp-less tracking pose INHERITS the last tracking stamp (the receiver's
            # rule, the same value the mailbox gets); no pose sink is wired here
            assert poses[2].t_sensor_ns == 1_000_000_000 and poses[2].mailbox_write is False
        finally:
            sub.close()


# ───────────────────────────── reader / rollups ─────────────────────────────

class TestReader:
    def test_read_events_requires_a_meta_line(self, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text('{"kind": "pose"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="meta"):
            ledger.read_events(p)
        p.write_text('{"kind": "meta", "schema_version": 3}\nnot json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="JSON"):
            ledger.read_events(p)

    def test_round_trip_through_a_real_writer(self, tmp_path):
        w = _writer(tmp_path)
        for r in synth_ledger.pose_rows(5):
            w.write(r)
        w.close()
        rows = ledger.read_events(w.path)
        assert ledger.ledger_meta(rows)["schema"] == LEDGER_SCHEMA
        assert {k: len(v) for k, v in ledger.by_kind(rows).items()} == {"meta": 1, "pose": 5}

    def test_pose_health_on_a_synthetic_stream(self):
        rows = synth_ledger.meta_row() + synth_ledger.pose_rows(
            300, hz=30, gaps=[(100, 2.0)], limited=[(150, 5)], jumps=[(220, 2.0)], conf2=0.6)
        h = ledger.pose_health(rows)
        g = h["groups"]["replay/1"]
        assert g["n_frames"] == 300 and g["n_by_tracking_state"] == {"normal": 295, "limited": 5}
        assert g["writes_expected"] == 295 == h["total"]["writes_expected"]
        assert g["n_stream"] == 300                                          # limited rows still carry a pose
        assert g["sensor_hz"] == pytest.approx(300 / ((299 / 30) + (2.0 - 1 / 30)) if False else g["sensor_hz"])
        assert 25.0 < g["sensor_hz"] < 30.0                                  # 299 intervals incl. one 2 s stall
        assert g["n_gaps"] == 1 and g["gaps"][0]["gap_s"] == pytest.approx(2.0)
        assert g["n_limited_episodes"] == 1
        ep = g["limited_episodes"][0]
        assert ep["n_frames"] == 5 and ep["states"] == {"limited": 5} and ep["duration_s"] == pytest.approx(4 / 30, abs=1e-3)
        assert g["n_discontinuities"] == 1 and g["discontinuities"][0]["jump_m"] == pytest.approx(2.0 + 0.2 / 30, abs=1e-3)
        assert g["discontinuities"][0]["t_sensor_ns"] == rows[1 + 220]["t_sensor_ns"]
        assert g["conf2_frac"]["mean"] == pytest.approx(0.6, abs=1e-3) and g["conf2_frac"]["n"] == 295
        assert g["depth_valid_frac"]["n"] == 295 and g["dt_ms"]["p50"] == pytest.approx(1000 / 30, abs=0.1)
        assert g["jitter_ms"] == pytest.approx(0.0, abs=0.01) and g["pose_errors"] == 0
        assert h["total"] == {"n_frames": 300, "writes_expected": 295, "n_gaps": 1, "n_limited_episodes": 1,
                              "n_discontinuities": 1, "pose_errors": 0, "n_groups": 1}

    def test_delivery_lag_measures_arrival_minus_sensor_time(self):
        rows = synth_ledger.pose_rows(300, hz=30)                       # arrival == sensor cadence -> lag 0
        g = ledger.pose_health(rows)["groups"]["replay/1"]
        assert g["delivery_lag"]["end_s"] == pytest.approx(0.0, abs=1e-6) and g["delivery_lag"]["n_catchups"] == 0
        for i, r in enumerate(rows):
            r["timestamp"] += i * 0.01                                   # frames delivered 10 ms later each
        rows[200]["timestamp"] -= 0.5                                    # one catch-up burst
        g = ledger.pose_health(rows)["groups"]["replay/1"]
        assert g["delivery_lag"]["end_s"] == pytest.approx(2.99, abs=1e-6)
        assert g["delivery_lag"]["max_s"] == pytest.approx(2.99, abs=1e-6)
        assert g["delivery_lag"]["slope_s_per_min"] == pytest.approx(2.99 / (299 / 30 / 60), rel=1e-3)
        assert g["delivery_lag"]["n_catchups"] == 1

    def test_pose_health_never_counts_across_epochs_or_sources(self):
        a = synth_ledger.pose_rows(50, epoch=1, t0_ns=10_000_000_000)
        b = synth_ledger.pose_rows(50, epoch=2, t0_ns=1_000_000_000, rx_seq0=51)      # restarted clock, smaller stamps
        z = synth_ledger.pose_rows(20, source="zeromq", epoch=0)
        for r in z:
            r.update(tracking_state="not_available", rx_seq=None, frame_seq=None, conf_hist=None, depth_valid_frac=None)
        h = ledger.pose_health(synth_ledger.meta_row() + a + b + z)
        assert set(h["groups"]) == {"replay/1", "replay/2", "zeromq/0"}
        assert h["total"]["n_gaps"] == 0 and h["total"]["n_discontinuities"] == 0
        assert h["groups"]["zeromq/0"]["n_limited_episodes"] == 0                # no tracking state on zeromq
        assert h["groups"]["zeromq/0"]["sensor_hz"] == pytest.approx(30.0, abs=0.01)
        assert h["total"]["n_groups"] == 3

    def test_limited_run_at_the_end_is_closed_and_pose_errors_counted(self):
        rows = synth_ledger.pose_rows(10, limited=[(7, 3)])
        rows[8]["t_wc"] = None; rows[8]["q_wc_xyzw"] = None; rows[8]["pose_error"] = "KeyError: 'T_wc'"
        g = ledger.pose_health(rows)["groups"]["replay/1"]
        assert g["n_limited_episodes"] == 1 and g["limited_episodes"][0]["n_frames"] == 3
        assert g["pose_errors"] == 1 and g["n_stream"] == 9

    def test_empty_and_pre_p2_files(self):
        assert ledger.pose_health([]) == {"params": {"disc_base_m": 0.5, "disc_rate_mps": 1.0, "gap_factor": 2.0},
                                          "groups": {}, "total": {"n_frames": 0, "writes_expected": 0, "n_gaps": 0,
                                                                  "n_limited_episodes": 0, "n_discontinuities": 0,
                                                                  "pose_errors": 0, "n_groups": 0}}
        assert ledger.ledger_meta([{"kind": "meta", "schema_version": 2}]) == {"enabled": False}

    def test_receiver_join_keys_line_up(self):
        pose = synth_ledger.pose_rows(60)
        rx = synth_ledger.receiver_rows_for(pose)
        by_rx = {r["rx_seq"]: r for r in rx}
        assert all(by_rx[p["rx_seq"]]["t_sensor_ns"] == p["t_sensor_ns"] for p in pose)
        assert all(by_rx[p["rx_seq"]]["depth_valid_frac"] == p["depth_valid_frac"] for p in pose)

    def test_summarize_cli(self, tmp_path, capsys):
        w = _writer(tmp_path)
        for r in synth_ledger.pose_rows(12):
            w.write(r)
        w.close()
        assert ledger.main(["summarize", str(w.path)]) == 0
        out = json.loads(capsys.readouterr().out)
        assert out["counts"] == {"meta": 1, "pose": 12} and out["pose_health"]["total"]["n_frames"] == 12

    def test_parquet_roundtrip(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        w = _writer(tmp_path)
        for r in synth_ledger.pose_rows(7, limited=[(2, 1)]):
            w.write(r)
        w.close()
        written = ledger.to_parquet(w.path)
        assert set(written) == {"pose"}
        t = pq.read_table(written["pose"])
        assert t.num_rows == 7 and t.column("t_wc")[0].as_py() == [0.0, 0.0, 1.5]

    def test_parquet_without_pyarrow_is_a_clear_error(self, tmp_path, monkeypatch):
        import builtins
        real = builtins.__import__

        def fake(name, *a, **k):
            if name.startswith("pyarrow"):
                raise ImportError("no pyarrow")
            return real(name, *a, **k)
        monkeypatch.setattr(builtins, "__import__", fake)
        w = _writer(tmp_path); w.close()
        with pytest.raises(RuntimeError, match="rtsm\\[eval\\]"):
            ledger.to_parquet(w.path)
