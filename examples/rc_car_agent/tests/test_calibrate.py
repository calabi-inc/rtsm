"""calibrate.py tests — plant known rig constants in the kinematic fake
car, run the calibration routines against it, assert they RECOVER the
planted values. Plus abort guards and the comment-preserving config
writeback."""

import math
import shutil
from pathlib import Path

import pytest

from calibrate import (
    CalibrationError,
    collect_maneuver,
    compute_lever_arm,
    compute_speed_scale,
    compute_turn_scale,
    compute_yaw_offset,
    fit_circle,
    write_calibration,
)
from config import load_config
from fake_car import FakeCar
from rtsm_client import RtsmClient

# Planted rig: camera skewed +0.3 rad, mounted 15 cm behind-of-center
# translated (5 cm right of) the drive center; sim scales 2.0 / 3.0.
PLANT_YAW = 0.3
PLANT_LEVER = (0.05, 0.15)
PLANT_SPEED = 2.0
PLANT_TURN = 3.0


def _rig(**kw):
    car = FakeCar(x=0.0, z=0.0, yaw=0.4, speed_scale_mps=PLANT_SPEED,
                  turn_scale_rps=PLANT_TURN, cam_yaw_offset=PLANT_YAW,
                  cam_lever_arm_rf=PLANT_LEVER, **kw)
    stops = []

    def stop_fn():
        stops.append(True)
        car.stop()

    pose_fn = lambda: RtsmClient._parse_pose(car.pose_payload())
    return car, car.set_drive, stop_fn, pose_fn, stops


# ── recovery of planted constants ────────────────────────────────────────


def test_yaw_offset_recovered():
    car, drive, stop, pose, stops = _rig()
    track = collect_maneuver(drive, stop, pose, 0.35, 0.35, duration_s=0.8)
    offset, diag = compute_yaw_offset(track)
    assert offset == pytest.approx(PLANT_YAW, abs=0.02)
    assert diag["disp_m"] > 0.25
    assert stops, "maneuver must stop the car"


def test_lever_arm_recovered():
    car, drive, stop, pose, _ = _rig()
    track = collect_maneuver(drive, stop, pose, -0.4, 0.4, duration_s=4.5)
    lever, diag = compute_lever_arm(track, PLANT_YAW)
    assert lever[0] == pytest.approx(PLANT_LEVER[0], abs=0.01)
    assert lever[1] == pytest.approx(PLANT_LEVER[1], abs=0.01)
    assert diag["sweep_deg"] >= 270
    assert diag["radius_m"] == pytest.approx(math.hypot(*PLANT_LEVER), abs=0.01)


def test_zero_lever_arm_at_drive_center():
    car = FakeCar(turn_scale_rps=PLANT_TURN)          # camera == drive center
    pose = lambda: RtsmClient._parse_pose(car.pose_payload())
    track = collect_maneuver(car.set_drive, car.stop, pose, -0.4, 0.4,
                             duration_s=4.5)
    lever, diag = compute_lever_arm(track, 0.0)
    assert lever == (0.0, 0.0)
    assert "drive center" in diag["note"]


def test_speed_scale_recovered():
    car, drive, stop, pose, _ = _rig()
    track = collect_maneuver(drive, stop, pose, 0.35, 0.35, duration_s=0.8)
    scale, diag = compute_speed_scale(track, 0.35)
    assert scale == pytest.approx(PLANT_SPEED, rel=0.05)


def test_turn_scale_recovered():
    car, drive, stop, pose, _ = _rig()
    track = collect_maneuver(drive, stop, pose, -0.4, 0.4, duration_s=2.0)
    scale, diag = compute_turn_scale(track, 0.4)
    assert scale == pytest.approx(PLANT_TURN, rel=0.05)


def test_fit_circle_exact():
    import numpy as np
    ang = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    pts = np.column_stack([1.5 + 0.2 * np.cos(ang), -0.7 + 0.2 * np.sin(ang)])
    cx, cz, r, rms = fit_circle(pts)
    assert (cx, cz, r) == (pytest.approx(1.5), pytest.approx(-0.7), pytest.approx(0.2))
    assert rms < 1e-9


# ── abort guards ─────────────────────────────────────────────────────────


def test_epoch_change_mid_routine_aborts_and_stops():
    car, drive, stop, pose, stops = _rig()
    calls = {"n": 0}

    def flaky_pose():
        calls["n"] += 1
        if calls["n"] == 4:
            car.epoch += 1                            # Lens "restarted"
        return pose()

    with pytest.raises(CalibrationError, match="frame_epoch"):
        collect_maneuver(drive, stop, flaky_pose, 0.3, 0.3, duration_s=3.0)
    assert stops, "abort must stop the car"


def test_stale_feed_mid_routine_aborts():
    car, drive, stop, pose, stops = _rig()
    calls = {"n": 0}

    def dying_pose():
        calls["n"] += 1
        if calls["n"] == 3:
            car.freeze()
        return pose()

    with pytest.raises(CalibrationError, match="stale"):
        collect_maneuver(drive, stop, dying_pose, 0.3, 0.3,
                         duration_s=5.0, stale_abort_s=0.4)
    assert stops


def test_too_little_motion_rejected():
    car, drive, stop, pose, _ = _rig()
    track = collect_maneuver(drive, stop, pose, 0.05, 0.05, duration_s=0.5)
    with pytest.raises(CalibrationError, match="moved only"):
        compute_yaw_offset(track)


# ── config writeback ─────────────────────────────────────────────────────


def test_write_calibration_roundtrip(tmp_path):
    src = Path(load_config().source_path)
    copy = tmp_path / "config.yaml"
    shutil.copy(src, copy)

    values = {"yaw_offset_rad": 0.2951, "lever_arm_right_m": 0.05,
              "lever_arm_forward_m": 0.152, "speed_scale_mps": 1.98,
              "turn_scale_rps": 2.87}
    write_calibration(copy, values, "hiwonder4wd-iphone13-tray-v1")

    cfg = load_config(str(copy))
    assert cfg.calibration.yaw_offset_rad == pytest.approx(0.2951)
    assert cfg.calibration.lever_arm_rf == (pytest.approx(0.05), pytest.approx(0.152))
    assert cfg.calibration.speed_scale_mps == pytest.approx(1.98)
    assert cfg.calibration.turn_scale_rps == pytest.approx(2.87)
    assert cfg.calibration.is_calibrated is True
    assert cfg.calibration.rig_id == "hiwonder4wd-iphone13-tray-v1"
    # Comments survive the rewrite (hand-edit warning + any field comment).
    text = copy.read_text(encoding="utf-8")
    assert "do not hand-edit" in text
    assert "# preflight refuses below this" in text


def test_write_calibration_missing_key_raises(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("calibration:\n  yaw_offset_rad: 0.0\n", encoding="utf-8")
    with pytest.raises(CalibrationError, match="lever_arm_right_m"):
        write_calibration(bad, {"yaw_offset_rad": 0.1,
                                "lever_arm_right_m": 0.0,
                                "lever_arm_forward_m": 0.0,
                                "speed_scale_mps": 1.0,
                                "turn_scale_rps": 1.0}, "rig")
