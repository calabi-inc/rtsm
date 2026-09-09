"""Pins the P0 config decision of the demo2 -> main reconcile (2026-09).

The packaged ``rtsm/cfg/rtsm.yaml`` keeps main's PUBLIC perception defaults;
the E1 (RC-car) campaign tuning lives only in the sparse profile
``examples/rc_car_agent/e1-demo2.profile.yaml``.

Why this exists: ``git merge-tree 73aa8e2 f7a0880`` auto-merges rtsm.yaml with
demo2's ``box_threshold 0.30`` and 5-class vocabulary as the shipped default,
and NOTHING else catches that -- ``tests/test_config_tuning.py`` guards dead
keys, not values, and the ``dual`` G0 floor never reads
``segmentation.grounded_sam2.*`` (dual builds FastSAM + YOLOE only).  A silent
regression here moves the public grounded_sam2 floor from ~133-139 objects to
~25 on recordings/session1.

These tests read the packaged file through ``rtsm.cfg.load_config`` (CPU-only,
no models) so they run under the canonical command.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from rtsm.cfg import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
E1_PROFILE = REPO_ROOT / "examples" / "rc_car_agent" / "e1-demo2.profile.yaml"
E1_ROSTER = ["teddy bear", "water bottle", "scissors", "tissue box", "dumbbell"]


def _leaves(node: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a config mapping to {dotted.path: leaf_value}."""
    out: Dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(_leaves(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = node
    return out


def test_packaged_grounded_sam2_defaults_are_public():
    cfg = load_config("rtsm.yaml")
    assert cfg["segmentation"]["backend"] == "grounded_sam2"
    g = cfg["segmentation"]["grounded_sam2"]
    assert g["box_threshold"] == pytest.approx(0.20)
    assert g["text_threshold"] == pytest.approx(0.15)
    assert g["vocab"] is None, "E1 roster must live in the profile, not the packaged yaml"


def test_packaged_yaml_carries_the_reconcile_blocks():
    """The blocks P0 task 3 adds on purpose, and the dead keys it must not resurrect."""
    cfg = load_config("rtsm.yaml")
    # main (PR #27 / #22) wins over demo2's merge-base copies
    assert cfg["gates"]["enable"] is True
    assert cfg["gates"]["min_depth_valid"] == pytest.approx(0.02)
    assert cfg["health"]["watchdog"]["enable"] is True
    # demo2 blocks taken into the packaged default (known keys for --set/--profile)
    assert cfg["ltm"]["ltm_min_view_bins"] == 1
    assert cfg["ltm"]["ltm_min_view_bins"] <= cfg["object"]["require_view_bins"]
    assert cfg["diagnostics"]["enabled"] is False
    assert cfg["staging"]["judge_crop_pad_frac"] == pytest.approx(0.20)
    assert cfg["staging"]["judge_crop_max_px"] == 640
    # receive-time clearance is opt-in (carve-clearance)
    assert cfg["io"]["clearance"]["enable"] is False
    # dead keys stay dead (PR #26)
    assert "masks" not in cfg
    assert "min_area_px" not in cfg["staging"]
    for dead in ("aspect_ratio", "solidity_min", "border_touch_max_pct", "border"):
        assert dead not in cfg["filters"], dead
    assert "valid_min_pct" not in cfg["filters"]["depth"]
    # ...while the live filter controls are present
    assert "min_area_px" in cfg["filters"]
    assert "sigma_max_m" in cfg["filters"]["depth"]


@pytest.mark.skipif(not E1_PROFILE.is_file(), reason="E1 profile not checked out")
def test_e1_profile_layers_only_the_campaign_tuning():
    base = load_config("rtsm.yaml")
    cfg = load_config("rtsm.yaml", profiles=[E1_PROFILE])
    g = cfg["segmentation"]["grounded_sam2"]
    assert g["box_threshold"] == pytest.approx(0.30)
    assert g["vocab"] == E1_ROSTER
    assert cfg["io"]["clearance"]["enable"] is True
    # The profile must NOT pin the backend: benchmark harnesses patch it in the
    # base file and a profile is layered after the base.
    assert cfg["segmentation"]["backend"] == base["segmentation"]["backend"]

    changed = {
        path
        for path in set(_leaves(base)) | set(_leaves(cfg))
        if _leaves(base).get(path, object()) != _leaves(cfg).get(path, object())
    }
    assert changed == {
        "segmentation.grounded_sam2.box_threshold",
        "segmentation.grounded_sam2.vocab",
        "io.clearance.enable",
    }, f"profile touches unexpected keys: {sorted(changed)}"
