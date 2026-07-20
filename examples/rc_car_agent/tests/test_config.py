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


# ── watchdog-chain validators (review fix: deleting them must regress) ───


def _bad(nav=None, esp32=None):
    from config import _validate

    cfg = load_config()
    if nav:
        cfg = dataclasses.replace(cfg, nav=dataclasses.replace(cfg.nav, **nav))
    if esp32:
        cfg = dataclasses.replace(cfg, esp32=dataclasses.replace(cfg.esp32, **esp32))
    return lambda: _validate(cfg)


def test_tick_slower_than_heartbeat_rejected():
    with pytest.raises(ValueError, match="tick_s"):
        _bad(nav={"tick_s": 0.26})()                  # >= heartbeat_s 0.25


def test_heartbeat_at_watchdog_window_rejected():
    with pytest.raises(ValueError):
        _bad(esp32={"heartbeat_s": 0.31})()


def test_low_drive_rate_starves_heartbeats_rejected():
    """1/drive_rate_hz = 0.33 s > 300 ms watchdog: a hold phase would
    stall wire sends past the watchdog even with a good heartbeat_s."""
    with pytest.raises(ValueError, match="watchdog"):
        _bad(esp32={"drive_rate_hz": 3.0})()


def test_poll_faster_than_tick_rejected():
    with pytest.raises(ValueError, match="poll_hz"):
        _bad(nav={"poll_hz": 1000.0})()               # 1 ms < tick_s 50 ms
