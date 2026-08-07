"""
Phase-G calibration — measures the four rig constants and writes them
back into config.yaml's calibration block.

    .venv/Scripts/python.exe calibrate.py --rig-id hiwonder4wd-iphone13-tray-v1 --write

The car's own pose stream is the instrument. Four routines, in this order
(the lever-arm math needs the yaw offset first):

  A. yaw_offset_rad   Drive straight ~2 s. The direction the CAMERA moved
                      through the world is the car's true forward; the
                      camera's reported heading is where the phone points.
                      offset = wrap(motion_dir - mean_cam_yaw), matching
                      geometry.camera_to_car's car_yaw = cam_yaw + offset.
  B. lever_arm_(r,f)  Rotate in place ~360°. The camera traces a circle
                      around the drive center; the fitted circle center IS
                      the drive center, and (center - camera) projected on
                      the car body axes is the lever arm.
  C. speed_scale_mps  Timed straight run at a known command magnitude ->
                      meters/second at command 1.0 (linear model).
  D. turn_scale_rps   Timed rotate at a known differential -> rad/second
                      at differential 1.0 (yaw slope on sensor time).

Safety: same stack as every drive — commands go through the send-gated
Esp32Bridge (firmware watchdog behind it), an EstopMonitor thread arms the
gamepad X, every routine stops the car in a finally, motion is low-speed
and time-bounded, and the operator must confirm floor space before any
wheel turns. A frame_epoch change or a stale pose feed mid-routine aborts.

Requires (hardware session): car on open floor, Calabi Lens streaming,
RTSM up. All math below is unit-tested against the kinematic fake car
with planted constants (tests/test_calibrate.py).
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from config import load_config
from geometry import wrap_angle, yaw_from_quat_xyzw
from rtsm_client import PoseSample

# Motion parameters — deliberately gentle; a calibration run should never
# need more than ~1.5 m of floor.
STRAIGHT_SPEED = 0.35
STRAIGHT_DURATION_S = 2.0
ROTATE_TURN = 0.4
ROTATE_DURATION_S = 8.0
TICK_S = 0.05                     # < heartbeat_s (0.25) — hold rule applies here too
POLL_EVERY_TICKS = 2              # pose poll ~10 Hz
STALE_ABORT_S = 2.5

MIN_STRAIGHT_DISP_M = 0.25        # below this, motion_dir is noise
MAX_STRAIGHT_SWEEP_RAD = math.radians(10)  # more heading change than this in
                                  # routine A = the car is ARCING (weak motor,
                                  # surface pull); the lever arm then biases
                                  # the chord by ~f*sweep and yaw_offset is off
MIN_YAW_SWEEP_RAD = math.radians(270)
MIN_TURN_SWEEP_RAD = math.radians(45)
MIN_LEVER_RADIUS_M = 0.03         # smaller than this = camera at drive center
MAX_CIRCLE_RMS_M = 0.06
MAX_LEVER_SPREAD_M = 0.02         # per-sample lever estimates must agree; a
                                  # bigger spread = the circle fit is lying
MAX_SAMPLE_GAP_S = 0.5            # sensor-time gap between fresh poses above
                                  # which np.unwrap can alias (lose 2*pi)


class CalibrationError(RuntimeError):
    pass


# ── maneuver: drive + collect fresh poses (heartbeat-correct) ────────────


def collect_maneuver(
    drive_fn: Callable[[float, float], object],
    stop_fn: Callable[[], object],
    pose_fn: Callable[[], Optional[PoseSample]],
    left: float,
    right: float,
    duration_s: float,
    tick_s: float = TICK_S,
    poll_every: int = POLL_EVERY_TICKS,
    stale_abort_s: float = STALE_ABORT_S,
) -> List[PoseSample]:
    """Command (left, right) for duration_s while collecting FRESH poses
    (sensor timestamp advancing). Epoch change or stale feed aborts.
    Always stops the car before returning/raising."""
    track: List[PoseSample] = []
    last_ts: Optional[float] = None
    ref_epoch: Optional[int] = None
    last_fresh_mono = time.monotonic()
    t_end = time.monotonic() + duration_s
    tick = 0
    try:
        while time.monotonic() < t_end:
            drive_fn(left, right)                 # every tick: feeds watchdog
            if tick % poll_every == 0:
                pose = pose_fn()
                now = time.monotonic()
                if pose is not None and (last_ts is None or pose.timestamp != last_ts):
                    last_ts = pose.timestamp
                    last_fresh_mono = now
                    if pose.frame_epoch is not None:
                        if ref_epoch is None:
                            ref_epoch = pose.frame_epoch
                        elif pose.frame_epoch != ref_epoch:
                            raise CalibrationError(
                                "frame_epoch changed mid-routine — Calabi Lens "
                                "restarted; redo this routine")
                    track.append(pose)
                elif now - last_fresh_mono > stale_abort_s:
                    raise CalibrationError(
                        f"pose feed stale for {stale_abort_s} s mid-routine — "
                        "is Calabi Lens still streaming?")
            tick += 1
            time.sleep(tick_s)
    finally:
        stop_fn()
    return track


# ── pure math ────────────────────────────────────────────────────────────


def _cam_yaws(track: Sequence[PoseSample]) -> np.ndarray:
    return np.array([yaw_from_quat_xyzw(p.quaternion_xyzw) for p in track])


def _xz(track: Sequence[PoseSample]) -> np.ndarray:
    return np.array([[p.xyz[0], p.xyz[2]] for p in track], dtype=np.float64)


def circular_mean(angles: np.ndarray) -> float:
    return float(math.atan2(float(np.mean(np.sin(angles))),
                            float(np.mean(np.cos(angles)))))


def _max_sensor_gap(track: Sequence[PoseSample]) -> float:
    ts = np.array([p.timestamp for p in track])
    return float(np.max(np.diff(ts))) if len(ts) > 1 else 0.0


def compute_yaw_offset(track: Sequence[PoseSample]) -> Tuple[float, dict]:
    """Routine A. Camera displacement direction minus mean camera heading.
    Guards: the run must actually be STRAIGHT — with a lever arm, an arcing
    run displaces the camera chord by ~f*sweep and silently biases the
    offset by degrees."""
    if len(track) < 5:
        raise CalibrationError(f"straight run collected only {len(track)} fresh poses")
    pts = _xz(track)
    disp = pts[-1] - pts[0]
    disp_norm = float(np.hypot(disp[0], disp[1]))
    if disp_norm < MIN_STRAIGHT_DISP_M:
        raise CalibrationError(
            f"straight run moved only {disp_norm:.2f} m "
            f"(need >= {MIN_STRAIGHT_DISP_M}) — wheels slipping or speed too low?")
    yaws = np.unwrap(_cam_yaws(track))
    sweep = float(abs(yaws[-1] - yaws[0]))
    if sweep > MAX_STRAIGHT_SWEEP_RAD:
        raise CalibrationError(
            f"straight run swept {math.degrees(sweep):.0f}° of heading "
            f"(need < {math.degrees(MAX_STRAIGHT_SWEEP_RAD):.0f}°) — car is "
            "arcing (one motor weak? surface pull?)")
    motion_yaw = math.atan2(disp[0], disp[1])          # atan2(dx, dz) — pinned
    cam_mean = circular_mean(_cam_yaws(track))
    offset = wrap_angle(motion_yaw - cam_mean)
    return offset, {"disp_m": disp_norm, "motion_yaw": motion_yaw,
                    "cam_yaw_mean": cam_mean, "sweep_deg": math.degrees(sweep),
                    "samples": len(track)}


def fit_circle(points: np.ndarray) -> Tuple[float, float, float, float]:
    """Kåsa algebraic least-squares circle fit -> (cx, cz, radius, rms).

    Points are CENTERED first: the raw Kåsa system is near-singular for a
    degenerate (tiny/point-like) blob, and lstsq's min-norm answer then
    depends on the distance from the WORLD ORIGIN — a zero-lever rig
    rotating at (5, 5) would fit a phantom 3.5 m circle. Centered, the
    same blob fits center ≈ centroid with millimeter radius and correctly
    lands in the camera-at-drive-center branch."""
    mu = points.mean(axis=0)
    p = points - mu
    x, z = p[:, 0], p[:, 1]
    A = np.column_stack([2 * x, 2 * z, np.ones_like(x)])
    b = x * x + z * z
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cz, c = sol
    r = math.sqrt(max(float(c + cx * cx + cz * cz), 0.0))
    residuals = np.hypot(x - cx, z - cz) - r
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    return float(cx + mu[0]), float(cz + mu[1]), r, rms


def compute_lever_arm(
    track: Sequence[PoseSample], yaw_offset: float
) -> Tuple[Tuple[float, float], dict]:
    """Routine B. (right, forward) from CAMERA to DRIVE CENTER, car body
    frame — the exact tuple geometry.camera_to_car consumes."""
    if len(track) < 10:
        raise CalibrationError(f"rotation collected only {len(track)} fresh poses")
    gap = _max_sensor_gap(track)
    if gap > MAX_SAMPLE_GAP_S:
        raise CalibrationError(
            f"{gap:.2f} s sensor gap between poses — yaw unwrap unreliable "
            "(pose feed stalled mid-rotation?); redo the routine")
    yaws = np.unwrap(_cam_yaws(track))
    sweep = float(abs(yaws[-1] - yaws[0]))
    if sweep < MIN_YAW_SWEEP_RAD:
        raise CalibrationError(
            f"rotation swept only {math.degrees(sweep):.0f}° "
            f"(need >= {math.degrees(MIN_YAW_SWEEP_RAD):.0f}°)")
    pts = _xz(track)
    cx, cz, radius, rms = fit_circle(pts)
    if radius < MIN_LEVER_RADIUS_M:
        return (0.0, 0.0), {"radius_m": radius, "rms_m": rms,
                            "sweep_deg": math.degrees(sweep), "note": "camera ~at drive center"}
    if rms > MAX_CIRCLE_RMS_M:
        raise CalibrationError(
            f"circle fit rms {rms:.3f} m > {MAX_CIRCLE_RMS_M} — car not "
            "rotating in place (surface slip? one motor weak?)")
    rs, fs = [], []
    for p, cam_yaw in zip(track, _cam_yaws(track)):
        car_yaw = wrap_angle(cam_yaw + yaw_offset)
        s, c = math.sin(car_yaw), math.cos(car_yaw)
        vx, vz = cx - p.xyz[0], cz - p.xyz[2]          # camera -> center
        fs.append(vx * s + vz * c)                     # dot(v, forward=(s, c))
        rs.append(vx * -c + vz * s)                    # dot(v, right=(-c, s))
    spread_r, spread_f = float(np.std(rs)), float(np.std(fs))
    if spread_r > MAX_LEVER_SPREAD_M or spread_f > MAX_LEVER_SPREAD_M:
        # A consistent rig gives near-identical per-sample (r, f); a big
        # spread means the fitted center is not the real rotation center.
        raise CalibrationError(
            f"lever-arm per-sample estimates inconsistent (spread right "
            f"{spread_r:.3f} m / forward {spread_f:.3f} m > "
            f"{MAX_LEVER_SPREAD_M}) — circle fit unreliable; redo the rotate")
    lever = (float(np.mean(rs)), float(np.mean(fs)))
    return lever, {"radius_m": radius, "rms_m": rms,
                   "sweep_deg": math.degrees(sweep),
                   "spread_r": spread_r, "spread_f": spread_f}


def compute_speed_scale(track: Sequence[PoseSample], commanded: float) -> Tuple[float, dict]:
    """Routine C. Displacement over SENSOR time, linearly scaled to 1.0."""
    if len(track) < 5:
        raise CalibrationError("speed run collected too few fresh poses")
    pts = _xz(track)
    dt = track[-1].timestamp - track[0].timestamp
    if dt <= 0.2:
        raise CalibrationError("speed run sensor-time span too short")
    disp = float(np.hypot(*(pts[-1] - pts[0])))
    v = disp / dt
    scale = v / abs(commanded)
    return scale, {"disp_m": disp, "dt_s": dt, "v_mps": v}


def compute_turn_scale(track: Sequence[PoseSample], commanded_turn: float) -> Tuple[float, dict]:
    """Routine D. Unwrapped yaw slope over sensor time, scaled to 1.0.
    Camera yaw rate == car yaw rate (constant mount offset). Guards: a
    sensor gap wide enough for unwrap to alias (lose 2*pi) aborts, and the
    sweep must be substantial — a silently 2-3x-low scale is worse than a
    redo."""
    if len(track) < 8:
        raise CalibrationError("turn run collected too few fresh poses")
    gap = _max_sensor_gap(track)
    if gap > MAX_SAMPLE_GAP_S:
        raise CalibrationError(
            f"{gap:.2f} s sensor gap between poses — yaw unwrap unreliable "
            "(pose feed stalled mid-rotation?); redo the routine")
    yaws = np.unwrap(_cam_yaws(track))
    dt = track[-1].timestamp - track[0].timestamp
    if dt <= 0.5:
        raise CalibrationError("turn run sensor-time span too short")
    sweep = float(abs(yaws[-1] - yaws[0]))
    if sweep < MIN_TURN_SWEEP_RAD:
        raise CalibrationError(
            f"turn run swept only {math.degrees(sweep):.0f}° "
            f"(need >= {math.degrees(MIN_TURN_SWEEP_RAD):.0f}°) — too little "
            "rotation to measure a rate")
    rate = float(sweep / dt)
    scale = rate / abs(commanded_turn)
    return scale, {"sweep_deg": math.degrees(sweep), "dt_s": dt, "rate_rps": rate}


# ── config writeback (regex, comment-preserving) ─────────────────────────


_RIG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def write_calibration(config_path, values: dict, rig_id: str) -> None:
    """Replace the calibration numbers in config.yaml IN PLACE, preserving
    every comment. values keys: yaw_offset_rad, lever_arm_right_m,
    lever_arm_forward_m, speed_scale_mps, turn_scale_rps.

    rig_id is restricted to [A-Za-z0-9._-]: it is interpolated into a
    regex replacement AND into YAML — a space would parse on the first
    write but brick the file on the SECOND (the matcher consumes one
    token), a backslash would expand as a group reference, and a quote
    would break the YAML. Refusing here beats corrupting a file whose
    header says "do not hand-edit". Replacements use a callable so no
    character in a value is ever regex-expanded."""
    if not _RIG_ID_RE.match(rig_id):
        raise CalibrationError(
            f"rig_id {rig_id!r} invalid — use only letters, digits, '.', "
            "'_' and '-' (e.g. hiwonder4wd-iphone13-tray-v1)")
    text = Path(config_path).read_text(encoding="utf-8")
    for key, val in values.items():
        num = repr(round(float(val), 6))
        pattern = rf"^(\s*{re.escape(key)}:\s*)[^\s#]+(.*)$"
        text, n = re.subn(pattern, lambda m, v=num: m.group(1) + v + m.group(2),
                          text, count=1, flags=re.MULTILINE)
        if n != 1:
            raise CalibrationError(f"config key {key!r} not found for writeback")
    stamp = datetime.now().isoformat(timespec="seconds")
    text, n1 = re.subn(r"^(\s*calibrated_at:\s*)\S+(.*)$",
                       lambda m: m.group(1) + f'"{stamp}"' + m.group(2),
                       text, count=1, flags=re.MULTILINE)
    text, n2 = re.subn(r"^(\s*rig_id:\s*)\S+(.*)$",
                       lambda m: m.group(1) + f'"{rig_id}"' + m.group(2),
                       text, count=1, flags=re.MULTILINE)
    if n1 != 1 or n2 != 1:
        raise CalibrationError("provenance keys not found for writeback")
    Path(config_path).write_text(text, encoding="utf-8")


# ── CLI (hardware session) ───────────────────────────────────────────────


def main() -> int:
    from esp32_bridge import Esp32Bridge
    from estop import EstopMonitor
    from rtsm_client import RtsmClient

    ap = argparse.ArgumentParser(description="RC-car rig calibration (Phase G)")
    ap.add_argument("--rig-id", required=True,
                    help='e.g. "hiwonder4wd-iphone13-tray-v1"')
    ap.add_argument("--write", action="store_true",
                    help="write results into config.yaml (default: print only)")
    args = ap.parse_args()

    cfg = load_config()
    rtsm = RtsmClient(cfg.rtsm.url)
    bridge = Esp32Bridge(cfg.esp32.url,
                         drive_rate_hz=cfg.esp32.drive_rate_hz,
                         heartbeat_s=cfg.esp32.heartbeat_s,
                         change_epsilon=cfg.esp32.change_epsilon,
                         http_timeout_s=cfg.esp32.http_timeout_s)
    stop_event = threading.Event()
    estop = EstopMonitor(bridge, stop_event).start()

    pose = rtsm.get_robot_pose()
    if pose is None:
        print("ABORT: no robot_pose — is Calabi Lens streaming into RTSM?")
        return 2
    if not estop.gamepad_available:
        print("WARNING: no gamepad — e-stop is Ctrl-C + firmware watchdog only.")
    else:
        # Detection is NOT proof the button reads (2026-08-07: a bound pad
        # was deaf). Live-fire before trusting it: hw_estop_check.py.
        print("Gamepad detected — NOT yet live-fire verified. If you have "
              "not pressed X on this binding (hw_estop_check.py or E1 "
              "protocol check), do not trust it as the kill switch.")
    print("Car on OPEN FLOOR (>= 1.5 m clear), wheels down, Lens streaming.")
    if input("Type 'go' to start the calibration drives: ").strip() != "go":
        return 1

    def drive(l, r):
        if stop_event.is_set():
            raise CalibrationError("e-stop triggered")
        bridge.drive(l, r)

    def pose_fn():
        try:
            return rtsm.get_robot_pose()
        except Exception:  # noqa: BLE001
            return None

    try:
        print("\n[A] straight drive (yaw offset)...")
        track = collect_maneuver(drive, bridge.stop, pose_fn,
                                 STRAIGHT_SPEED, STRAIGHT_SPEED, STRAIGHT_DURATION_S)
        yaw_offset, diag_a = compute_yaw_offset(track)
        print(f"    yaw_offset = {math.degrees(yaw_offset):+.1f}°  {diag_a}")

        print("[B] rotate in place (lever arm)...")
        track = collect_maneuver(drive, bridge.stop, pose_fn,
                                 -ROTATE_TURN, ROTATE_TURN, ROTATE_DURATION_S)
        (lever_r, lever_f), diag_b = compute_lever_arm(track, yaw_offset)
        print(f"    lever_arm right={lever_r:+.3f} m forward={lever_f:+.3f} m  {diag_b}")

        print("[C] timed straight (speed scale)...")
        track = collect_maneuver(drive, bridge.stop, pose_fn,
                                 STRAIGHT_SPEED, STRAIGHT_SPEED, STRAIGHT_DURATION_S)
        speed_scale, diag_c = compute_speed_scale(track, STRAIGHT_SPEED)
        print(f"    speed_scale = {speed_scale:.2f} m/s @1.0  {diag_c}")

        print("[D] timed rotate (turn scale)...")
        track = collect_maneuver(drive, bridge.stop, pose_fn,
                                 -ROTATE_TURN, ROTATE_TURN, ROTATE_DURATION_S / 2)
        turn_scale, diag_d = compute_turn_scale(track, ROTATE_TURN)
        print(f"    turn_scale = {turn_scale:.2f} rad/s @1.0  {diag_d}")
    except CalibrationError as e:
        bridge.stop()
        print(f"\nCALIBRATION FAILED: {e}")
        return 2
    finally:
        bridge.stop()
        estop.shutdown()

    values = {
        "yaw_offset_rad": yaw_offset,
        "lever_arm_right_m": lever_r,
        "lever_arm_forward_m": lever_f,
        "speed_scale_mps": speed_scale,
        "turn_scale_rps": turn_scale,
    }
    print(f"\nResults: {values}")
    if args.write:
        write_calibration(cfg.source_path, values, args.rig_id)
        print(f"Written to {cfg.source_path} (provenance stamped).")
    else:
        print("Dry run — re-run with --write to persist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
