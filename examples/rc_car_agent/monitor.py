"""
Mission monitor — the SOLE arrival/abort authority (invariant: nav steers,
monitor judges; nav never declares arrival).

Pure decision logic: no I/O, no clocks of its own. The nav loop feeds it
one (pose, now_mono) sample per poll and acts on the verdict. Everything
is injected, so every safety rule here is unit-testable without sleeping.

Verdicts (MonitorTick.status):
    ongoing            keep driving (geometry fields valid iff pose_fresh)
    arrived            ground_dist <= arrival_threshold_m on a FRESH pose
    timeout            trial exceeded its condition's time budget
    stale_stop         no fresh pose for stale_abort_s — BOUNDED blind hold
    frame_reset        world frame can no longer be trusted:
                         - frame_epoch changed (int vs int inequality), or
                         - pose teleported beyond base + rate*dt
    drift              ground_dist exceeded best-so-far + drift_margin_m
    degenerate_heading camera forward near-vertical (mount failure)

frame_epoch None matrix (locked 2026-07-20): the equality gate fires ONLY
when both the reference and the current epoch are ints. Any None side
(server predates the field, ZMQ/replay path, benign post-/reset transient)
defers to the pose-discontinuity heuristic — a None must NEVER hard-abort.
If the plan-time epoch was None, the first int epoch seen is adopted as
the reference so a mid-drive session restart is still caught.

Staleness (locked): freshness = the sensor timestamp ADVANCED since the
last sample seen, measured on OUR monotonic clock (never sensor-vs-wall).
The reference is seeded from plan_pose.timestamp, so a feed that died
BEFORE the mission (still serving the plan-time pose) is never "fresh" —
the car does not launch on a dead feed. Not-fresh -> nav holds the last
command (still heartbeating); blind for stale_abort_s -> safe stop +
abort — never blind until the trial timeout (audit pin 2026-07-20).
pose_frozen_polls only labels the hold "frozen" in logs (diagnostic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from config import CalibrationCfg, NavCfg
from geometry import (
    DegenerateHeadingError,
    camera_to_car,
    desired_yaw,
    ground_vec_xz,
    wrap_angle,
    yaw_from_quat_xyzw,
)
from rtsm_client import PoseSample


@dataclass(frozen=True)
class MonitorTick:
    status: str                      # see module docstring
    detail: str = ""
    pose_fresh: bool = False
    # Geometry snapshot — populated only when pose_fresh (car body frame).
    car_x: Optional[float] = None
    car_z: Optional[float] = None
    car_yaw: Optional[float] = None
    ground_dist: Optional[float] = None
    heading_err: Optional[float] = None


class MissionMonitor:
    def __init__(
        self,
        nav: NavCfg,
        calibration: CalibrationCfg,
        target_xyz: list,
        plan_pose: Optional[PoseSample],
        plan_epoch: Optional[int],
        timeout_s: float,
        t0_mono: float,
    ):
        self._nav = nav
        self._cal = calibration
        self._target = [float(v) for v in target_xyz]
        self._timeout_s = float(timeout_s)
        self._t0 = float(t0_mono)

        self._ref_epoch: Optional[int] = plan_epoch
        # Freshness reference seeded from the plan-time pose: a dead feed
        # still serving that exact timestamp must never count as fresh.
        self._last_ts: Optional[float] = (
            plan_pose.timestamp if plan_pose is not None else None
        )
        self._frozen_polls = 0
        self._last_fresh_mono = self._t0
        self._best_dist: Optional[float] = None
        # Discontinuity reference: (car_x, car_z) of the last accepted fresh
        # pose; seeded from plan_pose so the FIRST live pose is also checked
        # against where the plan said the car was.
        self._last_car: Optional[tuple] = None
        if plan_pose is not None:
            try:
                cam_yaw = yaw_from_quat_xyzw(plan_pose.quaternion_xyzw)
                cx, cz, _ = camera_to_car(
                    plan_pose.xyz, cam_yaw,
                    calibration.yaw_offset_rad, calibration.lever_arm_rf,
                )
                self._last_car = (cx, cz)
            except (DegenerateHeadingError, ValueError):
                self._last_car = None    # no reference; heuristic starts live

    # ── the one entry point ──────────────────────────────────────────────

    def assess(self, pose: Optional[PoseSample], now_mono: float) -> MonitorTick:
        if (now_mono - self._t0) > self._timeout_s:
            return MonitorTick(status="timeout",
                               detail=f"exceeded {self._timeout_s:.0f}s budget")

        if pose is None:
            return self._stale_or_hold(now_mono, detail="no pose from RTSM")

        fresh = self._last_ts is None or pose.timestamp != self._last_ts
        if not fresh:
            self._frozen_polls += 1
            frozen = self._frozen_polls >= self._nav.pose_frozen_polls
            return self._stale_or_hold(
                now_mono,
                detail=("pose feed frozen (timestamp not advancing)"
                        if frozen else "waiting for fresh pose"),
            )
        self._frozen_polls = 0
        self._last_ts = pose.timestamp

        # Epoch equality gate — ints only; None defers to the heuristic.
        if pose.frame_epoch is not None:
            if self._ref_epoch is None:
                self._ref_epoch = pose.frame_epoch      # adopt first int seen
            elif pose.frame_epoch != self._ref_epoch:
                return MonitorTick(
                    status="frame_reset", pose_fresh=True,
                    detail=(f"frame_epoch changed {self._ref_epoch} -> "
                            f"{pose.frame_epoch} (new sender session)"),
                )

        try:
            cam_yaw = yaw_from_quat_xyzw(pose.quaternion_xyzw)
        except DegenerateHeadingError as e:
            return MonitorTick(status="degenerate_heading", pose_fresh=True,
                               detail=str(e))
        except ValueError as e:
            # Malformed quaternion (wrong length / zero norm) that slipped
            # past the client — a verdict, never a crashed control loop.
            return MonitorTick(status="degenerate_heading", pose_fresh=True,
                               detail=f"malformed quaternion: {e}")
        car_x, car_z, car_yaw = camera_to_car(
            pose.xyz, cam_yaw, self._cal.yaw_offset_rad, self._cal.lever_arm_rf
        )

        # Pose-discontinuity heuristic (belt-and-braces for in-session ARKit
        # resets the epoch cannot see): a jump no real car could make.
        if self._last_car is not None:
            dt = max(now_mono - self._last_fresh_mono, 1e-3)
            jump = math.hypot(car_x - self._last_car[0], car_z - self._last_car[1])
            thresh = (self._nav.discontinuity_base_m
                      + self._nav.discontinuity_rate_mps * dt)
            if jump > thresh:
                return MonitorTick(
                    status="frame_reset", pose_fresh=True,
                    detail=(f"pose jumped {jump:.2f} m in {dt:.2f} s "
                            f"(threshold {thresh:.2f} m) — world frame moved"),
                )
        self._last_car = (car_x, car_z)
        self._last_fresh_mono = now_mono

        dx, dz = ground_vec_xz(self._target, (car_x, 0.0, car_z))
        dist = math.hypot(dx, dz)
        err = wrap_angle(desired_yaw(dx, dz) - car_yaw)

        if dist <= self._nav.arrival_threshold_m:
            return MonitorTick(status="arrived", pose_fresh=True,
                               detail=f"ground_dist {dist:.2f} m",
                               car_x=car_x, car_z=car_z, car_yaw=car_yaw,
                               ground_dist=dist, heading_err=err)

        if self._best_dist is None or dist < self._best_dist:
            self._best_dist = dist
        elif dist > self._best_dist + self._nav.drift_margin_m:
            return MonitorTick(
                status="drift", pose_fresh=True,
                detail=(f"ground_dist {dist:.2f} m exceeds best "
                        f"{self._best_dist:.2f} m + {self._nav.drift_margin_m} m"),
                car_x=car_x, car_z=car_z, car_yaw=car_yaw,
                ground_dist=dist, heading_err=err,
            )

        return MonitorTick(status="ongoing", pose_fresh=True,
                           car_x=car_x, car_z=car_z, car_yaw=car_yaw,
                           ground_dist=dist, heading_err=err)

    # ── helpers ──────────────────────────────────────────────────────────

    def _stale_or_hold(self, now_mono: float, detail: str) -> MonitorTick:
        blind_s = now_mono - self._last_fresh_mono
        if blind_s >= self._nav.stale_abort_s:
            return MonitorTick(
                status="stale_stop",
                detail=f"{detail}; blind for {blind_s:.1f} s "
                       f">= stale_abort_s {self._nav.stale_abort_s}",
            )
        return MonitorTick(status="ongoing", pose_fresh=False, detail=detail)
