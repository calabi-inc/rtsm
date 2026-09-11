"""Configuration safety and reproducibility, independent of GPU dependencies."""

import argparse
import json
import logging
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
    # Packaged defaults carry no legacy keys, so no ineffective-setting advisories.
    assert "has no effect" not in report and "not consumed" not in report
    assert "segmentation.fastsam.conf =" not in report
    assert "not a probability of correctness" in report
    assert "Detection vocabulary:" in report
    custom = load_config(set_values=["segmentation.grounded_sam2.vocab=[teddy bear]"])
    assert "Detection vocabulary: ['teddy bear']" in explain_tuning(custom, "pollution")


def test_legacy_keys_in_user_config_are_advisories_not_overrides(tmp_path):
    legacy = load_config()
    legacy["masks"] = {"min_coverage": .005, "max_coverage": .8, "max_border_fraction": .15}
    legacy["staging"]["min_area_px"] = 120
    legacy["filters"].update(aspect_ratio=[.2, 5.], solidity_min=.3, border_touch_max_pct=.15)
    legacy["filters"]["depth"]["valid_min_pct"] = .1
    legacy["filters"]["border"] = {"partial_min_pct": .3, "extreme_drop_pct": .9, "tiny_px": 150}
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    cfg = load_config(path)
    warnings = validate_tuning(cfg)
    for needle in ("masks section", "staging.min_area_px",
                   "filters.depth.valid_min_pct", "filters.aspect_ratio",
                   "filters.solidity_min", "filters.border_touch_max_pct",
                   "filters.border section"):
        assert any(needle in warning for warning in warnings), needle
    assert "masks section is not consumed" in explain_tuning(cfg, "pollution")
    # The legacy keys no longer exist in any packaged base, so they are not valid overrides.
    with pytest.raises(ConfigError, match="Unknown or malformed override"):
        load_config(set_values=["masks.max_coverage=0.8"])


def test_frame_gate_controls_follow_gates_enable():
    assert "gates.min_brightness = 5.0" in explain_tuning(load_config(), "latency")
    off = load_config(set_values=["gates.enable=false"])
    assert "gates.min_brightness" not in explain_tuning(off, "latency")
    with pytest.raises(ConfigError, match="gates.min_brightness"):
        validate_tuning(load_config(set_values=["gates.min_brightness=300"]))


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


# ---------------- deprecated paths (P1 task 6: io.websocket.* -> ingest.*) ----------------


def _deprecations(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "rtsm.cfg" and r.levelno >= logging.WARNING]


def test_old_websocket_paths_are_moved_via_set_and_profile(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="rtsm.cfg")
    with pytest.warns(DeprecationWarning, match="io.websocket.keyframe_every_n"):
        cfg = load_config(set_values=["io.websocket.keyframe_every_n=7"])
    assert cfg["ingest"]["keyframe_every_n"] == 7
    assert "keyframe_every_n" not in cfg["io"]["websocket"]
    assert cfg["io"]["websocket"]["port"] == 8765                      # the rest of io.websocket is intact
    assert any("io.websocket.keyframe_every_n" in m and "ingest.keyframe_every_n" in m for m in _deprecations(caplog))
    profile = tmp_path / "old.yaml"
    profile.write_text("io:\n  websocket:\n    port: 9000\n    nonkf_min_interval_s: 0.25\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="io.websocket.nonkf_min_interval_s"):
        cfg = load_config(profiles=[profile])
    assert cfg["ingest"]["nonkf_min_interval_s"] == 0.25 and cfg["io"]["websocket"]["port"] == 9000


def test_old_path_resolves_to_the_same_config_and_fingerprint_as_the_new_path(tmp_path):
    with pytest.warns(DeprecationWarning):
        old = load_config(set_values=["io.websocket.nonkf_min_interval_s=1.0"])
    new = load_config(set_values=["ingest.nonkf_min_interval_s=1.0"])
    assert old == new and config_fingerprint(old) == config_fingerprint(new)
    # a full base file written in the old layout (host / port stay under io.websocket)
    base = load_config()
    base["io"]["websocket"]["keyframe_every_n"] = base["ingest"].pop("keyframe_every_n")
    path = tmp_path / "old_base.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.warns(DeprecationWarning):
        aliased = load_config(path)                                    # no profiles / --set: the early-return path
    assert aliased == load_config() and config_fingerprint(aliased) == config_fingerprint(load_config())


def test_new_path_wins_when_one_source_names_both(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="rtsm.cfg")
    profile = tmp_path / "both.yaml"
    profile.write_text("ingest:\n  keyframe_every_n: 9\nio:\n  websocket:\n    keyframe_every_n: 4\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="IGNORED"):
        cfg = load_config(profiles=[profile])
    assert cfg["ingest"]["keyframe_every_n"] == 9
    assert any("IGNORED" in m for m in _deprecations(caplog))


def test_last_write_wins_across_sources_with_old_paths(tmp_path):
    profile = tmp_path / "p.yaml"
    profile.write_text("io:\n  websocket:\n    keyframe_every_n: 5\n", encoding="utf-8")
    with pytest.warns(DeprecationWarning):
        cfg = load_config(profiles=[profile], set_values=["io.websocket.keyframe_every_n=7"])
    assert cfg["ingest"]["keyframe_every_n"] == 7


def test_typos_of_old_paths_are_hinted_to_the_new_path():
    with pytest.raises(ConfigError, match="Did you mean 'ingest.keyframe_every_n'"):
        load_config(set_values=["io.websocket.keyfram_every_n=5"])
    with pytest.raises(ConfigError, match="Did you mean 'ingest.nonkf_min_interval_s'"):
        load_config(set_values=["io.websocket.nonkf_min_interval=0.4"])


def test_dotted_literal_old_key_is_rejected_not_silently_merged(tmp_path):
    profile = tmp_path / "dotted.yaml"
    profile.write_text("io.websocket.keyframe_every_n: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Did you mean 'ingest.keyframe_every_n'"):
        load_config(profiles=[profile])


def test_shim_refuses_to_overwrite_a_malformed_target_section(tmp_path):
    base = tmp_path / "bad.yaml"
    base.write_text("ingest: 5\nio:\n  websocket:\n    keyframe_every_n: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="'ingest' is not a mapping"):
        load_config(base)


def test_packaged_bases_carry_the_moved_keys_and_no_old_ones():
    for name, expected in (("rtsm.yaml", (30, 0.5)), ("demo_config.yaml", (5, 0.3))):
        cfg = load_config(name)
        assert (cfg["ingest"]["keyframe_every_n"], cfg["ingest"]["nonkf_min_interval_s"]) == expected, name
        assert "keyframe_every_n" not in cfg["io"]["websocket"] and "nonkf_min_interval_s" not in cfg["io"]["websocket"]
        assert cfg["ingest"]["non_kf_grace_s"] == 0.0
        assert (cfg["ingest"]["pair_window_s"], cfg["ingest"]["pair_window_fps"]) == (2.0, 30)


def test_throttle_control_lives_under_ingest_for_every_receiver():
    assert "ingest.nonkf_min_interval_s = 0.5" in explain_tuning(load_config(), "latency")
    zmq = load_config(set_values=["io.receiver=zeromq"])
    assert "ingest.nonkf_min_interval_s = 0.5" in explain_tuning(zmq, "latency")      # no longer hidden under ZeroMQ
    assert "io.websocket.nonkf_min_interval_s" not in explain_tuning(load_config(), "latency")
    with pytest.raises(ConfigError, match="ingest.nonkf_min_interval_s"):
        validate_tuning(load_config(set_values=["ingest.nonkf_min_interval_s=-1"]))


def test_config_show_prints_the_new_layout_for_an_old_style_file(tmp_path, capsys):
    base = load_config()
    base["io"]["websocket"]["nonkf_min_interval_s"] = base["ingest"].pop("nonkf_min_interval_s")
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.warns(DeprecationWarning):
        main(["show", "--config", str(path)])
    shown = yaml.safe_load(capsys.readouterr().out)
    assert shown["ingest"]["nonkf_min_interval_s"] == 0.5
    assert "nonkf_min_interval_s" not in shown["io"]["websocket"]
