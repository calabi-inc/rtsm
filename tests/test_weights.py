"""Validated weight files: the status of a file, the policy, the factory refusal, the provenance field, the fetch
script's verification."""
import hashlib
import importlib.util
import logging
from pathlib import Path

import pytest

from rtsm.models import weights as W


def _file(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_status_of_validated_mismatch_unknown_and_missing(tmp_path, monkeypatch):
    good = b"the validated bytes"
    monkeypatch.setitem(W.VALIDATED, "FastSAM-x.pt", {"sha256": hashlib.sha256(good).hexdigest(), "bytes": len(good), "note": "test"})
    ok = W.check_weight_file(_file(tmp_path, "FastSAM-x.pt", good))
    assert ok.state == "validated" and ok.ok and ok.bytes == len(good) and "validated" in ok.line()
    (tmp_path / "d").mkdir()
    bad = W.check_weight_file(_file(tmp_path / "d", "FastSAM-x.pt", b"other bytes"))
    assert bad.state == "mismatch" and not bad.ok and bad.expected_sha256 == hashlib.sha256(good).hexdigest() and "NOT the validated file" in bad.line()
    custom = W.check_weight_file(_file(tmp_path, "my-finetune.pt", b"whatever"))
    assert custom.state == "unknown" and custom.ok and custom.expected_sha256 is None
    gone = W.check_weight_file(tmp_path / "FastSAM-x.pt.absent")
    assert gone.state == "missing" and not gone.exists and not gone.ok
    # a relative path resolves against root
    rel = W.check_weight_file("FastSAM-x.pt", root=tmp_path)
    assert rel.state == "validated"


def test_validated_table_is_well_formed():
    for name, rec in W.VALIDATED.items():
        assert name.endswith(".pt") and len(rec["sha256"]) == 64 and int(rec["sha256"], 16) and rec["bytes"] > 1_000_000
    assert W.validation_state("model_store/yolo/yoloe-26s-seg-pf.pt", W.VALIDATED["yoloe-26s-seg-pf.pt"]["sha256"]) == "validated"
    assert W.validation_state("x/yoloe-26s-seg-pf.pt", "0" * 64) == "mismatch"
    assert W.validation_state("x/custom.pt", "0" * 64) is None


def test_policy_warn_strict_off(tmp_path, caplog):
    missing = W.check_weight_file(tmp_path / "yoloe-26s-seg-pf.pt")
    mismatch = W.check_weight_file(_file(tmp_path, "FastSAM-x.pt", b"not it"))
    custom = W.check_weight_file(_file(tmp_path, "custom.pt", b"mine"))
    with caplog.at_level(logging.INFO, logger="rtsm.models.weights"):
        W.enforce(missing, "warn"); W.enforce(mismatch, "warn"); W.enforce(custom, "warn")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2 and "download whatever it serves" in warnings[0].message and "2026-09-17" in warnings[1].message
    assert any(r.levelno == logging.INFO and "custom.pt" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rtsm.models.weights"):
        W.enforce(missing, "off"); W.enforce(mismatch, "off")
    assert not caplog.records
    with pytest.raises(W.WeightsError, match="missing"):
        W.enforce(missing, "strict")
    with pytest.raises(W.WeightsError, match="NOT the validated file"):
        W.enforce(mismatch, "strict")
    W.enforce(custom, "strict")                                   # custom weights pass every policy
    assert W.policy_from({}) == "warn" and W.policy_from({"weights_policy": "STRICT"}) == "strict"
    with pytest.raises(ValueError):
        W.policy_from({"weights_policy": "maybe"})


def test_factory_refuses_before_importing_the_backend(tmp_path):
    from rtsm.models.segmentation import get_segmenter
    cfg = {"segmentation": {"backend": "fastsam", "weights_policy": "strict",
                            "fastsam": {"model_path": str(tmp_path / "FastSAM-x.pt")}}}
    with pytest.raises(W.WeightsError, match="missing"):
        get_segmenter(cfg)
    cfg["segmentation"]["backend"] = "dual"
    cfg["segmentation"]["yoloe"] = {"model_path": str(tmp_path / "yoloe-26s-seg-pf.pt")}
    with pytest.raises(W.WeightsError):
        get_segmenter(cfg)


def test_provenance_records_the_validation_state(tmp_path):
    from rtsm.evaluation import runner as R
    (tmp_path / "FastSAM-x.pt").write_bytes(b"not the validated bytes")
    (tmp_path / "custom.pt").write_bytes(b"mine")
    cfg = {"segmentation": {"backend": "dual", "fastsam": {"model_path": str(tmp_path / "FastSAM-x.pt")},
                            "yoloe": {"model_path": str(tmp_path / "custom.pt")}}}
    files = R._model_files(cfg, root=tmp_path)
    assert files["segmentation.fastsam.model_path"]["validated"] == "mismatch"
    assert files["segmentation.yoloe.model_path"]["validated"] is None


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_models", Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_script_verify_reports_every_validated_file(tmp_path, capsys):
    mod = _fetch_module()
    (tmp_path / "fastsam").mkdir(); (tmp_path / "yolo").mkdir()
    (tmp_path / "fastsam" / "FastSAM-x.pt").write_bytes(b"wrong")
    statuses = mod.verify_all(model_store=str(tmp_path))
    by = {s.name: s.state for s in statuses}
    assert by == {"FastSAM-x.pt": "mismatch", "yoloe-26s-seg.pt": "missing", "yoloe-26s-seg-pf.pt": "missing"}
    assert mod.main(["--verify", "--model-store", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "NOT the validated file" in out and "missing" in out
