"""
Per-trial JSONL logger — the paper's E1 raw-data writer.

One file per trial: {output_dir}/{trial_id}.jsonl, three record types:

    trial_start  — provenance: goal, condition, planner pick (path/target/
                   score), plan_pose + plan-time frame_epoch, config
                   snapshot, calibration state. Protocol-era fields the
                   software cannot know (tape_cm, layout_id, video_file,
                   notes) are present-but-null so aggregate.py sees one
                   stable schema; the operator fills them post-trial.
    tick         — one line per monitor poll (~poll_hz): rel time, pose
                   (xyz/timestamp/frame_epoch), freshness, verdict status,
                   ground_dist, heading_err, wheel command.
    trial_end    — result + detail, elapsed_s, tick count, final_dist_m,
                   tta_s (elapsed iff arrived, else null), censored flag
                   (timeout per the E1 conservative-censoring policy).

Writes are appended and flushed per record so a crash mid-trial loses at
most one line — trial data is paper data. Logging failures are swallowed
after init: a broken disk must not crash the control loop.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1


def _git_commit() -> Optional[str]:
    """Best-effort repo provenance for the paper; never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True,
            text=True, timeout=3.0,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


class TrialLogger:
    def __init__(self, output_dir, trial_id: str, goal: str, condition: str, cfg):
        self.dir = Path(output_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{trial_id}.jsonl"
        # Never merge two trials into one file (append mode + an id
        # collision, e.g. across restarts, would corrupt E1 data).
        n = 0
        while self.path.exists():
            n += 1
            self.path = self.dir / f"{trial_id}-r{n}.jsonl"
        self.trial_id = trial_id
        self._goal = goal
        self._condition = condition
        self._cfg = cfg
        self._ticks = 0
        self._last_dist: Optional[float] = None
        self._fh = open(self.path, "a", encoding="utf-8")
        self._closed = False

    # ── records ──────────────────────────────────────────────────────────

    def log_plan(self, plan, rng_seed=None, rtsm_stats=None) -> None:
        nav = self._cfg.nav
        cal = self._cfg.calibration
        last_seen = getattr(plan, "target_last_seen_wall_utc", None)
        self._write({
            "type": "trial_start",
            "schema_version": SCHEMA_VERSION,
            "trial_id": self.trial_id,
            "goal": self._goal,
            "condition": self._condition,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "planner": {
                "status": plan.status,
                "planner_path": plan.planner_path,
                # The actual selection model (2026-08-30: swapped mid-
                # project; planner_path's "haiku" token is a historical
                # label for the LLM path, not the model).
                "model": self._cfg.planner.model,
                "target_id": plan.target_id,
                "label": plan.label,
                "score": plan.score,
                "confirmed": plan.confirmed,
                "stability": plan.stability,
                "xyz_world": plan.xyz_world,
                # Coordinate-age provenance: when the chosen target was
                # last observed vs when this plan ran — auditable answer
                # to "did memory plan on the scan or a fresher upsert?"
                "target_last_seen_wall_utc": last_seen,
                "target_age_at_plan_s": (round(time.time() - last_seen, 3)
                                         if last_seen is not None else None),
                "reason": plan.reason,
                # Which retrieval sources produced the candidate set —
                # label | semantic | label+semantic (2026-08-28; the
                # baseline stamps the same field per round).
                "retrieval": getattr(plan, "retrieval", None),
            },
            # Pipeline-throughput audit point (upserts delta start->end
            # reveals thermal/throughput drift within a session).
            "rtsm_stats": rtsm_stats,
            "frame_epoch": plan.frame_epoch,
            "plan_pose": self._pose_dict(plan.plan_pose),
            "config": {
                "arrival_threshold_m": nav.arrival_threshold_m,
                "max_speed": nav.max_speed,
                "max_turn": nav.max_turn,
                "kp_steer": nav.kp_steer,
                "tick_s": nav.tick_s,
                "poll_hz": nav.poll_hz,
                "stale_abort_s": nav.stale_abort_s,
                "drift_margin_m": nav.drift_margin_m,
                "timeout_rtsm_s": nav.timeout_rtsm_s,
                "timeout_baseline_s": nav.timeout_baseline_s,
                "heartbeat_s": self._cfg.esp32.heartbeat_s,
                "is_calibrated": cal.is_calibrated,
                "yaw_offset_rad": cal.yaw_offset_rad,
                "rig_id": cal.rig_id,
            },
            "provenance": {
                "git_commit": _git_commit(),
                "config_path": getattr(self._cfg, "source_path", None),
            },
            "rng_seed": rng_seed,         # baseline condition only
            # Operator-filled after the trial (kept null by software;
            # trial_start is the ONLY operator-editable record):
            "layout_id": None,
            "start_pose_id": None,
            "session_id": None,
            "tape_cm": None,
            "video_file": None,
            "notes": None,
        })

    def log_event(self, name: str, t_rel_s: float, **fields) -> None:
        """Free-form milestone record (e.g. baseline_acquired). Fields must
        be JSON-serializable."""
        self._write({"type": "event", "name": name,
                     "t": round(t_rel_s, 3), **fields})

    def log_tick(self, t_rel_s: float, pose, tick, left: float, right: float) -> None:
        self._ticks += 1
        if tick.ground_dist is not None:
            self._last_dist = tick.ground_dist
        self._write({
            "type": "tick",
            "t": round(t_rel_s, 3),
            "pose": self._pose_dict(pose),
            "fresh": tick.pose_fresh,
            "status": tick.status,
            "ground_dist_m": (round(tick.ground_dist, 3)
                              if tick.ground_dist is not None else None),
            "heading_err_rad": (round(tick.heading_err, 4)
                                if tick.heading_err is not None else None),
            "cmd": [round(left, 3), round(right, 3)],
        })

    def log_end(self, result: str, detail: str = "",
                elapsed_s: Optional[float] = None, rtsm_stats=None) -> None:
        # trial_end is MACHINE-ONLY (operators edit trial_start's nulls).
        # Interventions are encoded in result ("estopped"); photos are
        # recoverable by the <task_id>.jpg naming convention.
        self._write({
            "type": "trial_end",
            "trial_id": self.trial_id,
            "result": result,
            "detail": detail,
            "elapsed_s": round(elapsed_s, 3) if elapsed_s is not None else None,
            "ticks": self._ticks,
            "final_dist_m": (round(self._last_dist, 3)
                             if self._last_dist is not None else None),
            "tta_s": (round(elapsed_s, 3)
                      if result == "arrived" and elapsed_s is not None else None),
            "censored": result == "timeout",
            "rtsm_stats": rtsm_stats,
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._fh.close()
            except OSError:
                pass

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pose_dict(pose) -> Optional[dict]:
        if pose is None:
            return None
        return {
            "xyz": list(pose.xyz),
            "timestamp": pose.timestamp,
            "frame_epoch": pose.frame_epoch,
        }

    def _write(self, record: dict) -> None:
        if self._closed:
            return
        try:
            self._fh.write(json.dumps(record) + "\n")
            self._fh.flush()
        except (OSError, ValueError):
            pass    # a broken disk must not crash the control loop
