"""Smoke tests: the shipped config.yaml loads, types are right, guards fire."""

import dataclasses

import pytest

from config import load_config


def test_shipped_config_loads():
    cfg = load_config()
    assert cfg.rtsm.url.startswith("http")
    assert cfg.rtsm.lifecycle in ("attach", "spawn", "off")
    assert cfg.rtsm.kill_on_exit is False  # condition-(a) memory must survive restarts
    assert 0 < cfg.nav.max_speed <= 1.0
    assert cfg.nav.timeout_baseline_s >= cfg.nav.timeout_rtsm_s
    assert cfg.esp32.heartbeat_s < 0.3  # inside the firmware watchdog window
    assert cfg.calibration.lever_arm_rf == (0.0, 0.0)
    assert cfg.calibration.is_calibrated is False  # provenance empty until calibrate.py runs
    assert cfg.server.default_condition == "rtsm"


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.nav = None  # type: ignore[misc]
