"""Typed loader for config.yaml — one place, validated once at startup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"


@dataclass(frozen=True)
class RtsmCfg:
    url: str
    lifecycle: str          # attach | spawn | off
    spawn_cmd: str          # RTSM/GPU env interpreter + args (NOT this venv's python)
    kill_on_exit: bool
    diagnostics_for_trials: bool


@dataclass(frozen=True)
class Esp32Cfg:
    url: str
    drive_rate_hz: float
    heartbeat_s: float
    change_epsilon: float
    http_timeout_s: float
    battery_min_mv: int


@dataclass(frozen=True)
class ServerCfg:
    host: str
    port: int
    default_condition: str  # rtsm | baseline


@dataclass(frozen=True)
class NavCfg:
    arrival_threshold_m: float
    rotate_in_place_deg: float
    max_speed: float
    max_turn: float
    kp_steer: float
    pose_stale_s: float
    pose_frozen_polls: int
    timeout_rtsm_s: float
    timeout_baseline_s: float


@dataclass(frozen=True)
class PlannerCfg:
    model: str
    api_timeout_s: float
    fallback: str


@dataclass(frozen=True)
class BaselineCfg:
    freshness_gate_s: float
    rng_seed: int


@dataclass(frozen=True)
class CalibrationCfg:
    yaw_offset_rad: float
    lever_arm_right_m: float
    lever_arm_forward_m: float
    speed_scale_mps: float
    turn_scale_rps: float
    calibrated_at: Optional[str]
    rig_id: Optional[str]

    @property
    def lever_arm_rf(self) -> Tuple[float, float]:
        return (self.lever_arm_right_m, self.lever_arm_forward_m)

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated_at is not None


@dataclass(frozen=True)
class Config:
    rtsm: RtsmCfg
    esp32: Esp32Cfg
    server: ServerCfg
    nav: NavCfg
    planner: PlannerCfg
    baseline: BaselineCfg
    calibration: CalibrationCfg
    trials_output_dir: str
    source_path: str = field(default="", compare=False)


def load_config(path: Optional[str] = None) -> Config:
    p = Path(path) if path else _DEFAULT_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))

    cal = raw["calibration"]
    prov = cal.get("provenance") or {}
    cfg = Config(
        rtsm=RtsmCfg(**raw["rtsm"]),
        esp32=Esp32Cfg(**raw["esp32"]),
        server=ServerCfg(**raw["server"]),
        nav=NavCfg(**raw["nav"]),
        planner=PlannerCfg(**raw["planner"]),
        baseline=BaselineCfg(**raw["baseline"]),
        calibration=CalibrationCfg(
            yaw_offset_rad=float(cal["yaw_offset_rad"]),
            lever_arm_right_m=float(cal["lever_arm_right_m"]),
            lever_arm_forward_m=float(cal["lever_arm_forward_m"]),
            speed_scale_mps=float(cal["speed_scale_mps"]),
            turn_scale_rps=float(cal["turn_scale_rps"]),
            calibrated_at=prov.get("calibrated_at"),
            rig_id=prov.get("rig_id"),
        ),
        trials_output_dir=str(raw["trials"]["output_dir"]),
        source_path=str(p),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.rtsm.lifecycle not in ("attach", "spawn", "off"):
        raise ValueError(f"rtsm.lifecycle must be attach|spawn|off, got {cfg.rtsm.lifecycle!r}")
    if cfg.server.default_condition not in ("rtsm", "baseline"):
        raise ValueError(
            f"server.default_condition must be rtsm|baseline, got {cfg.server.default_condition!r}"
        )
    if not (0 < cfg.nav.max_speed <= 1.0 and 0 < cfg.nav.max_turn <= 1.0):
        raise ValueError("nav.max_speed / nav.max_turn must be in (0, 1]")
    if cfg.esp32.heartbeat_s >= 0.3:
        raise ValueError(
            "esp32.heartbeat_s must stay below the 300 ms firmware watchdog window"
        )
    if cfg.nav.timeout_baseline_s < cfg.nav.timeout_rtsm_s:
        raise ValueError("baseline timeout must be >= rtsm timeout (censoring policy)")
