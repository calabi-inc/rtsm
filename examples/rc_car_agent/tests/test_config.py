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


def test_dotenv_loads_but_never_overrides(tmp_path, monkeypatch):
    from config import load_dotenv

    envfile = tmp_path / ".env"
    envfile.write_text(
        "# comment\n"
        "FAKE_NEW_KEY=abc123\n"
        'FAKE_EXISTING_KEY="from-file"\n'
        "malformed line without equals\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FAKE_NEW_KEY", raising=False)
    monkeypatch.setenv("FAKE_EXISTING_KEY", "from-environment")

    n = load_dotenv(envfile)

    assert n == 1                                            # only the new key
    import os
    assert os.environ["FAKE_NEW_KEY"] == "abc123"            # quotes stripped, set
    assert os.environ["FAKE_EXISTING_KEY"] == "from-environment"  # env wins
    monkeypatch.delenv("FAKE_NEW_KEY", raising=False)


def test_dotenv_missing_file_is_noop(tmp_path):
    from config import load_dotenv

    assert load_dotenv(tmp_path / "does_not_exist.env") == 0
