"""The timing run (B3): --speed paces the input, a non-lossless ingest policy passes through to eval (with a warning when it is
not paced), the resolved record names the compute, and the receiver lines carry the transport's arrival stamps."""
from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")

from rtsm.cfg import load_config  # noqa: E402
from rtsm.evaluation import runner as R  # noqa: E402
from test_eval_runner import fake_runtime_factory  # noqa: E402


@pytest.fixture
def bag(tmp_path):
    from test_bag_reader import simple_bag_messages, write_bag
    return write_bag(tmp_path / "bag", simple_bag_messages(12, seq_topic=True))


@pytest.fixture
def cfg():
    return load_config()


def test_platform_facts_and_the_text_readers(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    f = R._platform_facts()
    assert set(f) == {"gpu", "cuda", "device_model", "power_mode"} and f["gpu"] is None and f["cuda"] is None
    st = tmp_path / "status"
    st.write_text("pmode:0002 fmode:quiet\nsecond line\n", encoding="utf-8")
    assert R._read_text(st) == "pmode:0002 fmode:quiet"
    dt = tmp_path / "model"
    dt.write_bytes(b"NVIDIA Jetson Orin Nano Developer Kit\x00")
    assert R._read_text(dt) == "NVIDIA Jetson Orin Nano Developer Kit"
    assert R._read_text(tmp_path / "missing") is None and R._read_text(tmp_path) is None
    (tmp_path / "empty").write_text("  \n", encoding="utf-8")
    assert R._read_text(tmp_path / "empty") is None


def test_resolve_speed_and_policy_pass_through(cfg, bag, caplog):
    r = R.resolve_eval(cfg, R.EvalOptions(input=str(bag), speed=2.0))
    assert r.speed == 2.0 and r.policy == "lossless" and r.policy_configured == "auto"
    assert {"gpu", "cuda", "device_model", "power_mode"} <= set(r.__dict__)
    cfg2 = json.loads(json.dumps(cfg))
    cfg2.setdefault("ingest", {})["policy"] = "latest"
    with caplog.at_level(logging.WARNING, logger="rtsm.evaluation.runner"):
        r2 = R.resolve_eval(cfg2, R.EvalOptions(input=str(bag)))
    assert r2.policy == "latest" and r2.policy_configured == "latest" and r2.speed is None
    assert any("without --speed" in rec.getMessage() for rec in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rtsm.evaluation.runner"):
        r3 = R.resolve_eval(cfg2, R.EvalOptions(input=str(bag), speed=1.0))
    assert r3.speed == 1.0 and not any("without --speed" in rec.getMessage() for rec in caplog.records)
    cfg3 = json.loads(json.dumps(cfg))
    cfg3["eval"]["speed"] = 0.5
    assert R.resolve_eval(cfg3, R.EvalOptions(input=str(bag))).speed == 0.5
    for bad in (0, -1.0, "fast", True):
        cfg4 = json.loads(json.dumps(cfg))
        cfg4["eval"]["speed"] = bad
        with pytest.raises(ValueError, match="eval.speed"):
            R.resolve_eval(cfg4, R.EvalOptions(input=str(bag)))


def test_cli_speed_flag(cfg, bag, tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_eval(cfg, opts):
        seen["opts"] = opts
        r = R.resolve_eval(cfg, opts)
        res = R.EvalResult(out_dir=tmp_path / "o", resolved=r)
        res.runs.append({"memory": {"fingerprint": "abc", "objects_count": 1, "confirmed_count": 0}, "frames": {"processed": 1},
                         "wall_s": 0.1, "aborted": None, "run_dir": "x"})
        return res

    monkeypatch.setattr(R, "run_eval", fake_run_eval)
    assert R.main([str(bag), "--speed", "3"]) == 0
    assert seen["opts"].speed == 3.0 and "speed=3x" in capsys.readouterr().out
    assert R.main([str(bag)]) == 0 and seen["opts"].speed is None and "speed=as fast as accepted" in capsys.readouterr().out


def test_paced_run_records_speed_policy_and_arrival_stamps(cfg, bag, tmp_path):
    cfg2 = json.loads(json.dumps(cfg))
    cfg2.setdefault("ingest", {})["policy"] = "latest"
    opts = R.EvalOptions(input=str(bag), repeats=1, out=str(tmp_path / "out"), speed=40.0, report=False)
    result = R.run_eval(cfg2, opts, models=object(), runtime_factory=fake_runtime_factory)
    assert len(result.runs) == 1 and result.runs[0]["aborted"] is None
    res = json.loads((tmp_path / "out" / "resolved.json").read_text(encoding="utf-8"))
    assert res["speed"] == 40.0 and res["policy"] == "latest" and res["policy_configured"] == "latest"
    assert "gpu" in res and "cuda" in res and "device_model" in res and "power_mode" in res
    rows = [json.loads(l) for l in (tmp_path / "out" / "run_1" / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["kind"] == "meta" and rows[0]["speed"] == 40.0 and rows[0]["ingest_policy"] == "latest"
    rx = [r for r in rows if r["kind"] == "receiver"]
    enq = [r for r in rx if r["decision"] == "enqueued"]
    assert enq and all(isinstance(r["t_arrival_mono"], float) and isinstance(r["t_arrival_wall_utc"], float) and r["t_arrival_mono"] <= r["timestamp"] for r in enq)
    thr = [r for r in rx if r["reason"] == "throttle"]
    assert thr and all(r["t_arrival_mono"] is not None for r in thr)
    # the arrivals are paced: 12 frames at the synthetic 100 ms cadence span 1.1 s of recording, at 40x at least 27 ms of wall
    arrivals = sorted(r["t_arrival_mono"] for r in rx if r.get("t_arrival_mono") is not None)
    assert arrivals[-1] - arrivals[0] >= 0.02
