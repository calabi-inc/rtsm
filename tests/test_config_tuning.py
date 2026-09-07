"""Configuration safety and reproducibility, independent of GPU dependencies."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from rtsm.cfg import ConfigError, cfg_path, config_fingerprint, load_config
from rtsm.cfg.cli import add_config_arguments, config_from_args, main
from rtsm.cfg.tuning import explain_tuning, validate_tuning


@pytest.mark.parametrize("name", ["rtsm.yaml", "demo_config.yaml"])
def test_default_values_are_unchanged(name):
    assert load_config(name) == yaml.safe_load(cfg_path(name).read_text(encoding="utf-8"))
    validate_tuning(load_config(name))


def test_explicit_base_does_not_silently_fall_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_config(Path("rtsm.yaml"))
    Path("rtsm.yaml").write_text("object:\n  promote_hits: 7\n", encoding="utf-8")
    assert load_config(Path("rtsm.yaml"))["object"]["promote_hits"] == 7
    assert load_config("rtsm.yaml")["object"]["promote_hits"] == 2
    assert cfg_path("clip/vocab.yaml").is_file()


def test_profiles_then_cli_preserve_unrelated_values_and_replace_lists(tmp_path):
    room = tmp_path / "room.yaml"
    trial = tmp_path / "trial.yaml"
    room.write_text("object:\n  promote_hits: 4\nsegmentation:\n  grounded_sam2:\n    vocab: [cup, book]\n", encoding="utf-8")
    trial.write_text("object:\n  promote_hits: 5\n", encoding="utf-8")
    baseline = load_config()
    cfg = load_config(profiles=[room, trial], set_values=["object.promote_hits=6"])
    assert cfg["object"]["promote_hits"] == 6
    assert cfg["segmentation"]["grounded_sam2"]["vocab"] == ["cup", "book"]
    assert cfg["object"]["stability_promote"] == baseline["object"]["stability_promote"]
    assert cfg["assoc"] == baseline["assoc"]
    assert load_config() == baseline


@pytest.mark.parametrize("text", [
    "assoc: null\n",
    "assoc:\n  cos_mni: 0.8\n",
    "unknown_section:\n  threshold: 1\n",
])
def test_malformed_or_misspelled_profiles_fail(tmp_path, text):
    profile = tmp_path / "bad.yaml"
    profile.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(profiles=[profile])


def test_typo_points_to_real_control():
    with pytest.raises(ConfigError, match="Did you mean 'assoc.cos_min'"):
        load_config(set_values=["assoc.cos_mni=0.8"])


def test_required_filter_is_not_reported_as_an_invented_default():
    cfg = load_config()
    del cfg["filters"]["min_area_px"]
    with pytest.raises(ConfigError, match="filters.min_area_px"):
        validate_tuning(cfg)


@pytest.mark.parametrize("text", [
    "object:\n  promote_hits: 2\n  promote_hits: 9\n",
    "[]",
    "",
    "value: .nan",
    "value: 2026-01-01",
    "value: &cycle [*cycle]",
])
def test_ambiguous_or_nonportable_yaml_fails(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.parametrize("assignment", [
    "assoc.cos_min=1.1",
    "assoc.cos_min=true",
    "assoc.cos_min='0.8'",
    "object.promote_hits=1.5",
    "object.promote_hits=0",
    "staging.topk_preclip=-1",
    "staging.depth_valid_min=-0.1",
    "filters.depth.sigma_max_m=0",
    "object.stability_promote=null",
    "segmentation.backend=unregistered",
])
def test_invalid_active_tuning_values_fail(assignment):
    with pytest.raises(ConfigError):
        validate_tuning(load_config(set_values=[assignment]))


@pytest.mark.parametrize("assignment", ["broken", "assoc..cos_min=.8", "assoc.cos_min=.inf"])
def test_invalid_assignments_fail(assignment):
    with pytest.raises(ConfigError):
        load_config(set_values=[assignment])


def test_expert_settings_in_full_base_can_be_overridden(tmp_path):
    base = load_config()
    base["diagnostics"] = {"enabled": False, "track_drops": False}
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    cfg = load_config(path, set_values=["diagnostics.track_drops=true"])
    assert cfg["diagnostics"]["track_drops"] is True
    assert cfg["diagnostics"]["enabled"] is False


def test_demo_can_select_a_backend_absent_from_demo_yaml():
    cfg = load_config("demo_config.yaml", set_values=[
        "segmentation.backend=fastsam", "segmentation.fastsam.conf=0.65",
    ])
    validate_tuning(cfg)
    report = explain_tuning(cfg, "missing")
    assert "segmentation.fastsam.conf = 0.65" in report
    assert "segmentation.grounded_sam2.box_threshold =" not in report


def test_guide_exposes_effective_controls_and_inactive_ones():
    cfg = load_config()
    report = explain_tuning(cfg, "pollution")
    assert "staging.depth_valid_min = 0.02" in report
    assert "filters.depth.valid_min_pct has no effect" in report
    assert "gates section is not consumed" in report
    assert "segmentation.fastsam.conf =" not in report
    assert "not a probability of correctness" in report
    assert "Detection vocabulary:" in report
    custom = load_config(set_values=["segmentation.grounded_sam2.vocab=[teddy bear]"])
    assert "Detection vocabulary: ['teddy bear']" in explain_tuning(custom, "pollution")


def test_upsert_mismatch_is_advisory_not_unsupported_constraint():
    cfg = load_config(set_values=["ltm.ltm_min_view_bins=3"])
    assert any("more view bins than confirmation" in warning for warning in validate_tuning(cfg))
    demo = load_config("demo_config.yaml", set_values=["object.require_view_bins=4"])
    assert "ltm.ltm_min_view_bins = 4 [component default]" in explain_tuning(demo, "search")


def test_config_arguments_use_runner_default_and_override_it():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args(["--set", "object.promote_hits=4"])
    cfg = config_from_args(args, "demo_config.yaml")
    assert cfg["object"]["promote_hits"] == 4
    assert cfg["object"]["stability_promote"] == .4


def test_snapshot_is_valid_yaml_and_reusable_without_gpu(tmp_path, capsys):
    main(["show", "--demo", "--set", "assoc.gate_dist_base_m=0.4"])
    captured = capsys.readouterr()
    cfg = yaml.safe_load(captured.out)
    expected = load_config("demo_config.yaml", set_values=["assoc.gate_dist_base_m=0.4"])
    assert cfg == expected
    assert "Advisory:" not in captured.out
    snapshot = tmp_path / "frozen.yaml"
    snapshot.write_text(captured.out, encoding="utf-8")
    assert load_config(snapshot) == expected
    assert config_fingerprint(cfg) == config_fingerprint(expected)
    assert config_fingerprint(cfg) != config_fingerprint(load_config("demo_config.yaml"))
    assert config_fingerprint(cfg) == config_fingerprint(json.loads(json.dumps(cfg, sort_keys=True)))


def test_cli_errors_exit_without_traceback(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--set", "assoc.cos_min=2"])
    assert exc.value.code == 2
    assert "assoc.cos_min" in capsys.readouterr().err


def test_config_dispatch_does_not_import_runtime_or_models():
    # Make accidental model/runtime imports fail even on GPU-equipped machines.
    code = """
import importlib.abc
import runpy
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch', 'numpy', 'rtsm.run', 'open_clip', 'sam2'):
            raise RuntimeError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.argv = ['rtsm', 'config', 'validate']
runpy.run_module('rtsm', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Tuning controls valid" in result.stdout
