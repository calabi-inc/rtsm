"""
Closed-loop navigation — deterministic steering + the control loop.

Division of labor (locked): the monitor is the SOLE arrival/abort
authority; nav converts the monitor's geometry into wheel commands and
honors every interrupt. Nav NEVER declares arrival.

════════════════════════════════════════════════════════════════════════
STEER SIGN (load-bearing — hardware-verified conventions)
════════════════════════════════════════════════════════════════════════
Positive heading error err = wrap(desired_yaw − car_yaw) means the target
is to the LEFT (CCW viewed from above; geometry.py fact #3). To turn CCW a
differential-drive car speeds up the RIGHT wheel:

    left  = base − kp_steer·err
    right = base + kp_steer·err          # err > 0  →  right > left  →  CCW

⚠ Do NOT reuse teleop's arcade_mix with steer=+err: teleop's positive
steer is a RIGHT/CW turn (stick right), i.e. ANTI-ALIGNED with positive
(CCW) yaw error. That sign flip drives away from the target and is the
single most likely silent bug in this file — test_steer_sign pins it.

Rotate-in-place (|err| > rotate_in_place_deg): throttle ≈ 0, pure
differential with the SAME sign rule: err > 0 → left backward, right
forward → CCW.

════════════════════════════════════════════════════════════════════════
HOLD RULE (the most-misimplemented rule, per the plan)
════════════════════════════════════════════════════════════════════════
Between fresh poses nav HOLDS the last (left, right) — but KEEPS CALLING
bridge.drive() every tick. The bridge's change-or-heartbeat gate decides
what actually hits the wire; the 0.25 s heartbeat is what feeds the ESP32's
300 ms firmware watchdog. An implementation that "holds" by NOT calling
drive() silently stops the car mid-trial (watchdog fires at 300 ms).
config._validate enforces tick_s < heartbeat_s < 0.3.

The monitor bounds the hold: past stale_abort_s it verdicts stale_stop and
the loop safe-stops. The car never drives blind until the trial timeout.

Fail-safe inheritance: if THIS loop dies (exception, GIL stall, process
kill), drive() calls cease and the firmware watchdog stops the car within
300 ms. bridge.drive() returning False (e-stop latch, rate gate, HTTP
drop) is EXPECTED and never retried against — the next tick tries again.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

from config import Config
from esp32_bridge import Esp32Bridge
from monitor import MissionMonitor
from planner import PlanResult
from rtsm_client import RtsmClient
from trial_logger import TrialLogger


def drive_command(heading_err: float, nav_cfg) -> Tuple[float, float, str]:
    """Heading error (rad, +=CCW/left) → (left, right, mode) in [-1, 1].

    Pure and deterministic — the whole steering policy, unit-tested in
    isolation (test_nav.py pins both signs and both modes).
    """
    import math

    rotate_rad = math.radians(nav_cfg.rotate_in_place_deg)
    if abs(heading_err) > rotate_rad:
        t = nav_cfg.max_turn
        if heading_err > 0.0:            # target LEFT → rotate CCW
            return -t, t, "rotate"
        return t, -t, "rotate"           # target RIGHT → rotate CW

    base = nav_cfg.max_speed
    d = max(-nav_cfg.max_turn, min(nav_cfg.max_turn,
                                   nav_cfg.kp_steer * heading_err))
    left = max(-1.0, min(1.0, base - d))
    right = max(-1.0, min(1.0, base + d))
    return left, right, "translate"


class NavRunner:
    """Runs one mission to completion. Returns (result, detail).

    result ∈ arrived | timeout | stale_stop | frame_reset | drift |
             degenerate_heading | estopped | preempted | cancelled | shutdown
    """

    def __init__(
        self,
        cfg: Config,
        bridge: Esp32Bridge,
        rtsm: RtsmClient,
        plan: PlanResult,
        condition: str,
        stop_event: threading.Event,
        preempt_event: threading.Event,
        cancel_event: threading.Event,
        shutdown_event: threading.Event,
        logger: Optional[TrialLogger] = None,
        progress: Optional[dict] = None,
        timeout_s_override: Optional[float] = None,
        log_t0_mono: Optional[float] = None,
    ):
        self._cfg = cfg
        self._bridge = bridge
        self._rtsm = rtsm
        self._plan = plan
        self._stop_event = stop_event
        self._preempt = preempt_event
        self._cancel = cancel_event
        self._shutdown = shutdown_event
        self._logger = logger
        self._progress = progress if progress is not None else {}

        # Override: the baseline's search phase spends part of the trial
        # budget before the drive starts — the drive gets the remainder.
        timeout_s = timeout_s_override if timeout_s_override is not None else (
            cfg.nav.timeout_rtsm_s if condition == "rtsm"
            else cfg.nav.timeout_baseline_s)
        self._t0 = time.monotonic()
        # Log clock: all records in one trial JSONL must share ONE epoch
        # (the mission's t0), even though the monitor's timeout budget runs
        # from drive start — for baseline trials the two differ by the
        # whole search phase.
        self._log_t0 = log_t0_mono if log_t0_mono is not None else self._t0
        self._monitor = MissionMonitor(
            nav=cfg.nav,
            calibration=cfg.calibration,
            target_xyz=list(plan.xyz_world or []),
            plan_pose=plan.plan_pose,
            plan_epoch=plan.frame_epoch,
            timeout_s=timeout_s,
            t0_mono=self._t0,
        )

    def run(self) -> Tuple[str, str]:
        nav = self._cfg.nav
        poll_interval = 1.0 / nav.poll_hz
        next_poll = 0.0                          # poll immediately
        left = right = 0.0
        have_cmd = False                         # no motion before first fresh pose

        while True:
            # ── interrupts, highest priority, every tick ────────────────
            if self._shutdown.is_set():
                self._bridge.stop()
                return "shutdown", "server shutting down"
            if self._stop_event.is_set():
                # E-stop path already latched the bridge and posted /stop —
                # do NOT command motors again (drive() would no-op anyway).
                return "estopped", "hard e-stop"
            if self._preempt.is_set():
                self._preempt.clear()
                self._bridge.stop()              # safe stop BEFORE goal swap
                return "preempted", "new command preempted this drive"
            if self._cancel.is_set():
                self._cancel.clear()
                self._bridge.stop()
                return "cancelled", "operator cancel"

            if time.monotonic() >= next_poll:
                pose = self._safe_pose()
                # Clock AFTER the fetch: if RTSM blocked us for seconds,
                # the staleness/timeout verdicts must see that time.
                now = time.monotonic()
                next_poll = now + poll_interval
                tick = self._monitor.assess(pose, now)
                self._note_progress(tick)
                if (tick.status == "ongoing" and tick.pose_fresh
                        and tick.heading_err is not None):
                    left, right, _mode = drive_command(tick.heading_err, nav)
                    have_cmd = True
                if self._logger is not None:
                    # Logged AFTER the command update: each fresh tick line
                    # pairs heading_err with the command DERIVED FROM IT
                    # (hold/terminal ticks carry the held command).
                    self._logger.log_tick(now - self._log_t0, pose, tick,
                                          left, right)
                if tick.status != "ongoing":
                    self._bridge.stop()
                    return tick.status, tick.detail

            # HOLD RULE: called every tick, fresh pose or not (see module
            # docstring) — the bridge decides what hits the wire. Skipped
            # the instant an interrupt is pending (handled next iteration):
            # a cancelled/e-stopped mission must not push one more command.
            if have_cmd and not (self._stop_event.is_set()
                                 or self._preempt.is_set()
                                 or self._cancel.is_set()):
                self._bridge.drive(left, right)

            time.sleep(nav.tick_s)

    # ── helpers ──────────────────────────────────────────────────────────

    def _safe_pose(self):
        try:
            return self._rtsm.get_robot_pose()
        except Exception:  # noqa: BLE001 — an RTSM hiccup is a stale tick,
            return None    # never a crashed control loop

    def _note_progress(self, tick) -> None:
        self._progress["ticks"] = self._progress.get("ticks", 0) + 1
        self._progress["phase"] = "driving"
        if tick.ground_dist is not None:
            self._progress["ground_dist_m"] = round(tick.ground_dist, 3)
