"""The session-split clone (scripts/split_recording_session.py) and the replay
source's mid-stream hello: a clone of a recording whose second half carries a
new session id replays as two sessions."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from rtsm.io.ingest_queue import IngestQueue

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "split_recording_session.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("split_recording_session", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_recording(root: pathlib.Path, n: int = 6, session_id: str = "A-SESSION") -> pathlib.Path:
    rec = root / "rec"
    rec.mkdir()
    blob = b""
    lines = []
    for i in range(n):
        chunk = bytes([i]) * (10 + i)
        lines.append({"seq": i, "offset": len(blob), "length": len(chunk), "t_mono_s": 10.0 + i * 0.5})
        blob += chunk
    (rec / "messages.bin").write_bytes(blob)
    (rec / "index.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    (rec / "meta.json").write_text(json.dumps({"format_version": 1, "session_id": session_id, "device_name": "iPhone",
                                              "hello": {"type": "hello", "protocol_version": 1, "session_id": session_id}}))
    (rec / "text_messages.jsonl").write_text(json.dumps({"t_mono_s": 10.2, "after_seq": 1, "payload": json.dumps({"type": "pose_corrections", "corrections": {}})}) + "\n")
    return rec


def test_split_writes_a_clone_with_a_hello_at_the_split(tmp_path):
    mod = _load_script()
    src = _fake_recording(tmp_path)
    dst = tmp_path / "rec_split"
    summary = mod.split_recording(src, dst, at=0.5, session_id="B-SESSION")
    assert summary["split_index"] == 3 and summary["split_seq"] == 3
    assert summary["frames_first_half"] == 3 and summary["frames_second_half"] == 3
    assert summary["first_session_id"] == "A-SESSION" and summary["second_session_id"] == "B-SESSION"
    assert (dst / "messages.bin").stat().st_size == (src / "messages.bin").stat().st_size
    assert (dst / "index.jsonl").read_text() == (src / "index.jsonl").read_text()
    texts = [json.loads(l) for l in (dst / "text_messages.jsonl").read_text().splitlines()]
    assert len(texts) == 2 and texts[0]["after_seq"] == 1                      # the existing message kept, in order
    hello = texts[1]
    assert hello["after_seq"] == 3 and hello["t_mono_s"] == pytest.approx(11.5 - 1e-6, abs=1e-9)
    assert json.loads(hello["payload"]) == {"type": "hello", "protocol_version": 1, "session_id": "B-SESSION", "device_name": "iPhone"}
    meta = json.loads((dst / "meta.json").read_text())
    assert meta["session_id"] == "A-SESSION" and meta["split"]["second_session_id"] == "B-SESSION"


def test_split_refuses_bad_inputs(tmp_path):
    mod = _load_script()
    src = _fake_recording(tmp_path)
    with pytest.raises(ValueError):
        mod.split_recording(src, tmp_path / "x", at=1.5)
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "f").write_text("x")
    with pytest.raises(FileExistsError):
        mod.split_recording(src, tmp_path / "busy")
    with pytest.raises(FileNotFoundError):
        mod.split_recording(tmp_path / "nope", tmp_path / "y")
    assert mod.main([str(src), str(tmp_path / "cli_out"), "--session-id", "C"]) == 0
    assert (tmp_path / "cli_out" / "text_messages.jsonl").is_file()


def test_replay_of_the_clone_announces_two_sessions(tmp_path, monkeypatch):
    mod = _load_script()
    src = _fake_recording(tmp_path)
    dst = tmp_path / "rec_split"
    mod.split_recording(src, dst, at=0.5, session_id="B-SESSION")
    import rtsm.io.replayer as replayer
    monkeypatch.setattr(replayer, "parse_lens_message", lambda fe, raw, **kw: None)   # frames: skipped, stamps still flow
    seen = []
    rx = replayer.ReplayReceiver(recording_dir=str(dst), ingest_queue=IngestQueue(8), replay_speed=1000.0,
                                 on_session_change=lambda sid, e: seen.append((sid, e)))
    rx.start()
    assert rx.wait(10.0)
    assert seen == [("A-SESSION", 1), ("B-SESSION", 2)]
    assert rx.frontend._epoch_session_id == "B-SESSION" and rx.frontend.frame_epoch == 2
    assert rx.frontend.frame_count == 0                                    # reset at the hello, as the live handshake does
