"""Typed loader for config.yaml — one place, validated once at startup.

Also loads an optional `.env` file (same folder, gitignored) for secrets
like ANTHROPIC_API_KEY. Real environment variables always win over .env —
so `setx ANTHROPIC_API_KEY ...` (recommended) needs no file at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"
_DOTENV_PATH = Path(__file__).resolve().parent / ".env"


def load_dotenv(path: Optional[Path] = None) -> int:
    """Minimal .env loader (no dependency): KEY=VALUE lines, `#` comments.

    Sets os.environ ONLY for keys not already present — the process
    environment always takes precedence. Returns how many keys were set.
    """
    p = path if path is not None else _DOTENV_PATH
    if not p.exists():
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            n += 1
    return n


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
    # Refuse ALL motion goals (bench included) until the kill switch is
    # bound AND live-fire verified (a real button press on the CURRENT
    # binding). Software enforcement of the E1 protocol rule; disable only
    # in mock-hardware tests.
    require_verified_estop: bool = True
    # Shared secret for POST endpoints when serving the LAN (--host
    # 0.0.0.0): clients send X-Auth-Token. None = no auth (localhost use).
    api_token: Optional[str] = None


@dataclass(frozen=True)
class NavCfg:
    arrival_threshold_m: float
    rotate_in_place_deg: float
    max_speed: float
    max_turn: float
    kp_steer: float
    tick_s: float
    poll_hz: float
    pose_frozen_polls: int
    stale_abort_s: float
    drift_margin_m: float
    discontinuity_base_m: float
    discontinuity_rate_mps: float
    timeout_rtsm_s: float
    timeout_baseline_s: float
    # Drive-phase obstacle guard (2026-08-16, after a live wall hit during
    # a search trial's DRIVE: the acquired coordinate routed through a
    # wall and blind-hold pushed into it). Applies to BOTH conditions.
    # Trip rule: fresh clearance below blocked_clearance_m while the
    # believed target is still farther than blocked_min_target_dist_m
    # (so the target itself filling the camera never trips it), for
    # blocked_debounce_polls consecutive fresh polls -> verdict "blocked".
    blocked_clearance_m: float = 0.30    # <=0 disables the drive guard
    blocked_min_target_dist_m: float = 0.75
    blocked_debounce_polls: int = 3
    # Blind-hold refinement: if the pose feed goes stale while last-known
    # clearance was below this, hold ZERO (stop) instead of holding the
    # last drive command — never push blind toward a known-near obstacle.
    blind_hold_min_clearance_m: float = 0.50
    clearance_max_age_s: float = 2.0     # stale clearance sample = ignore it


@dataclass(frozen=True)
class PlannerCfg:
    model: str
    api_timeout_s: float
    fallback: str
    # Attach each candidate's latest camera crop to the selection call so
    # the LLM judges by the IMAGE, not the captioner's label (labels lie:
    # the teddy bear was captioned 'audio' and rejected mid-search,
    # t20260811-200758-001). Best-effort — a missing crop degrades that
    # candidate to text-only, never blocks.
    include_snapshots: bool = True
    # Must cover baseline.query_top_k (validated): a candidate without its
    # crop degrades to text-only, and in the flat single-standpoint score
    # band (0.028-0.045 measured 2026-08-28) labels+scores cannot carry a
    # rejection on their own.
    snapshot_max_candidates: int = 10
    # Shared retrieval policy (2026-08-28): "label_first" = detector-label
    # search (prompted vocabulary; reaches protos), semantic fallback on
    # miss/error/off-vocab goal; "semantic" = embeddings only (pre-
    # grounded behavior). Applies to condition (a)'s plan(); the baseline
    # has its own copy of the knob so both are visible in trial configs.
    retrieval: str = "label_first"


@dataclass(frozen=True)
class BaselineCfg:
    freshness_gate_s: float
    rng_seed: int
    sweep_step_turn: float
    sweep_step_s: float
    dwell_s: float
    steps_per_sweep: int
    walk_speed: float
    walk_s: float
    query_top_k: int
    gate_fetch_k: int
    clock_skew_tol_s: float
    # Depth wall guard (2026-08-16; superseded the short-lived geofence —
    # the car senses walls with the live depth stream instead of a
    # per-session corner-captured box): the relocate walk requires this
    # many meters of MEASURED open space ahead (10th-percentile of the
    # phone's live depth band, served by RTSM). Blocked -> the searcher
    # rotates step-by-step and walks the first open direction; a full
    # circle with none -> stays put and keeps sweeping. <= 0 disables.
    # Session-independent: no setup, survives stream restarts.
    min_walk_clearance_m: float = 0.60
    clearance_max_age_s: float = 2.0     # stale clearance = blind = no walk
    # Steered stride (2026-08-17): walk HALF the measured open depth
    # toward the most open heading — the next sweep happens mid-open-area
    # with visibility all around, instead of a fixed 12 cm hop (a relic
    # of the blind-walk era). Capped, floored, and walked in chunks with
    # a clearance re-check between chunks.
    walk_max_m: float = 1.0
    walk_min_m: float = 0.12
    walk_chunk_m: float = 0.30
    # Observe-then-confirm rounds (2026-08-28): full in-place 360° sweeps
    # per round BEFORE the single batched selection call — multiple passes
    # give the pipeline multiple views (better crops, more view bins)
    # before any LLM time is spent. Operator-set: >= 3 spins, then judge.
    sweeps_per_round: int = 3
    # Search cap (2026-08-28): the acquisition phase gets at most this
    # much of the trial budget; exhausting it ends the trial as NOT FOUND
    # — an explicit, analyzable outcome (every standpoint reached was
    # observed and judged; none contained the target) instead of a
    # generic timeout, and the remainder stays reserved for a drive the
    # ~0.04 m/s rig could actually complete. <= 0 disables (cap=budget).
    search_cap_s: float = 480.0
    # Search leash (2026-08-17): depth can see PAST the venue (the tape
    # boundary is not a wall), so open-space steering alone would walk
    # the car out of the experiment area. The searcher stays within this
    # radius of its START pose: strides are trimmed at the leash, and
    # when every open heading leads outside, it walks back toward the
    # start instead. Sized for a car starting centered in a ~4.9 x 3 m
    # venue (half-diagonal ~2.9 m). <= 0 disables.
    search_leash_m: float = 2.0
    # Relevance floor — DISABLED by default (0.0). Measured 2026-08-28:
    # the single-standpoint score band is FLAT (top-15 for "tissue box"
    # spanned 0.028-0.045 with the true target mid-pack), so ANY usable
    # floor also kills the target; the 0.05 floor shipped that morning
    # would have filtered even rank 1. Selection pressure is ranking
    # (query_top_k) + ONE batched image-verified LLM call per round.
    # The knob stays for venues with a measured separated band. <= 0 off.
    min_candidate_score: float = 0.0
    # Round-query retrieval (2026-08-28): "label_first" | "semantic" —
    # see PlannerCfg.retrieval. The round window / masking / batched
    # image selection are identical either way.
    retrieval: str = "label_first"


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
    load_dotenv()  # secrets first (no-op if .env absent; env vars win)
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
    # Watchdog chain: the nav loop must tick faster than the bridge heartbeat,
    # which must beat the 300 ms firmware watchdog — otherwise a "hold last
    # command" phase silently stops the car mid-trial.
    if not (0 < cfg.nav.tick_s < cfg.esp32.heartbeat_s):
        raise ValueError(
            "nav.tick_s must be positive and < esp32.heartbeat_s "
            "(hold phases keep the watchdog fed only if ticks outpace heartbeats)"
        )
    if cfg.nav.poll_hz <= 0 or (1.0 / cfg.nav.poll_hz) < cfg.nav.tick_s:
        raise ValueError("nav.poll_hz must be positive and no faster than the tick rate")
    # Wire cadence: during a hold, sends happen every max(heartbeat_s,
    # 1/drive_rate_hz) — that interval must beat the firmware watchdog too
    # (a legal-looking drive_rate_hz of 3 would starve heartbeats).
    if max(cfg.esp32.heartbeat_s, 1.0 / cfg.esp32.drive_rate_hz) >= 0.3:
        raise ValueError(
            "max(esp32.heartbeat_s, 1/esp32.drive_rate_hz) must stay below "
            "the 300 ms firmware watchdog window"
        )
    if cfg.nav.timeout_baseline_s < cfg.nav.timeout_rtsm_s:
        raise ValueError("baseline timeout must be >= rtsm timeout (censoring policy)")
    if cfg.baseline.sweeps_per_round < 1:
        raise ValueError("baseline.sweeps_per_round must be >= 1")
    if cfg.baseline.search_cap_s > cfg.nav.timeout_baseline_s:
        raise ValueError(
            "baseline.search_cap_s cannot exceed the trial budget "
            "(the cap carves the acquisition phase OUT of timeout_baseline_s)"
        )
    for name, val in (("planner", cfg.planner.retrieval),
                      ("baseline", cfg.baseline.retrieval)):
        if val not in ("label_first", "semantic"):
            raise ValueError(
                f"{name}.retrieval must be label_first|semantic, got {val!r}")
    if cfg.planner.retrieval != cfg.baseline.retrieval:
        raise ValueError(
            "planner.retrieval and baseline.retrieval must MATCH — the E1 "
            "protocol claims an identical retrieval policy in both arms; "
            "a silent divergence here would invalidate the comparison"
        )
    if (cfg.planner.include_snapshots
            and cfg.planner.snapshot_max_candidates < cfg.baseline.query_top_k):
        raise ValueError(
            "planner.snapshot_max_candidates must cover baseline.query_top_k "
            "— a batched candidate without its crop is judged by a label, "
            "and labels lie (the teddy bear was captioned 'audio')"
        )
