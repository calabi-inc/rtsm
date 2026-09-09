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


def test_zero_lever_arm_at_drive_center_far_from_origin():
    """Review fix: the raw Kåsa fit's min-norm solution depended on the
    distance from the WORLD ORIGIN — a zero-lever rig rotating at (5, 5)
    fitted a phantom 3.5 m circle and produced a garbage lever. The
    centered fit must land in the camera-at-drive-center branch anywhere."""
    car = FakeCar(x=5.0, z=5.0, turn_scale_rps=PLANT_TURN)   # camera == center
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


# ── synthetic-track guards (review fixes; pure math, no sleeps) ──────────


def _sample(x, z, yaw, ts):
    from rtsm_client import PoseSample
    half = 0.5 * yaw
    return PoseSample(xyz=[x, 0.3, z],
                      quaternion_xyzw=[0.0, math.sin(half), 0.0, math.cos(half)],
                      timestamp=ts, fetched_at_mono=0.0, frame_epoch=7)


def _curved_track(sweep_rad, length_m=0.6, n=24, lever=(0.05, 0.15),
                  yaw_offset=PLANT_YAW):
    """Constant-curvature drive-center path + the camera mount model —
    what a weak motor produces during 'straight' routine A."""
    track = []
    x = z = 0.0
    psi = 0.4
    ds = length_m / n
    for i in range(n + 1):
        r, f = lever
        s, c = math.sin(psi), math.cos(psi)
        cam_x = x - f * s + r * c
        cam_z = z - f * c - r * s
        track.append(_sample(cam_x, cam_z, psi - yaw_offset, 100.0 + 0.1 * i))
        x += ds * s
        z += ds * c
        psi += sweep_rad / n
    return track


def test_arcing_straight_run_rejected():
    """Review fix: a 25° arc during routine A + a real lever arm biases
    yaw_offset by ~5-6° with healthy-looking diagnostics — must abort."""
    with pytest.raises(CalibrationError, match="arcing"):
        compute_yaw_offset(_curved_track(math.radians(25)))


def test_gentle_run_passes_with_sweep_diagnostic():
    offset, diag = compute_yaw_offset(_curved_track(math.radians(2)))
    assert offset == pytest.approx(PLANT_YAW, abs=0.03)
    assert diag["sweep_deg"] == pytest.approx(2.0, abs=0.5)


def test_turn_scale_sensor_gap_aborts():
    """Review fix: a pose-feed stall under the 2.5 s abort threshold lets
    np.unwrap alias (lose 2*pi) and silently write a 2-3x-low turn_scale."""
    track = [_sample(0, 0, 0.3 * i, 100.0 + 0.1 * i) for i in range(10)]
    track += [_sample(0, 0, 0.3 * i, 102.2 + 0.1 * i) for i in range(10, 20)]
    with pytest.raises(CalibrationError, match="sensor gap"):
        compute_turn_scale(track, 0.4)


def test_lever_arm_sensor_gap_aborts():
    track = [_sample(0.2 * math.sin(t), 0.2 * math.cos(t), t, 100.0 + 0.1 * i)
             for i, t in enumerate([0.3 * j for j in range(12)])]
    gap = [_sample(0.2 * math.sin(t), 0.2 * math.cos(t), t, 103.5 + 0.1 * i)
           for i, t in enumerate([0.3 * j for j in range(12, 22)])]
    with pytest.raises(CalibrationError, match="sensor gap"):
        compute_lever_arm(track + gap, 0.0)


def test_turn_scale_min_sweep_rejected():
    track = [_sample(0, 0, 0.01 * i, 100.0 + 0.1 * i) for i in range(12)]
    with pytest.raises(CalibrationError, match="swept only"):
        compute_turn_scale(track, 0.4)


def test_inconsistent_lever_projections_rejected():
    """Review fix: a clean-looking circle whose yaws don't match the
    rotation (fit center is not the true rotation center) must abort
    instead of averaging garbage — the per-sample spread gate."""
    thetas = [0.25 * i for i in range(20)]            # position angle
    track = [_sample(0.2 * math.sin(t), 0.2 * math.cos(t), 2.0 * t,   # yaw 2x
                     100.0 + 0.1 * i) for i, t in enumerate(thetas)]
    with pytest.raises(CalibrationError, match="inconsistent"):
        compute_lever_arm(track, 0.0)


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


_VALUES = {"yaw_offset_rad": 0.1, "lever_arm_right_m": 0.02,
           "lever_arm_forward_m": 0.15, "speed_scale_mps": 1.9,
           "turn_scale_rps": 2.8}


def test_rig_id_with_unsafe_characters_refused(tmp_path):
    """Review fix: a spaced rig_id parses on the FIRST write but bricks
    config.yaml on the SECOND (the matcher consumes one token); quotes
    break the YAML; backslashes expand as group references. Refuse all."""
    src = Path(load_config().source_path)
    copy = tmp_path / "config.yaml"
    shutil.copy(src, copy)
    for bad in ("hiwonder 4wd v1", 'rig"x', "rig\\1", "", " lead"):
        with pytest.raises(CalibrationError, match="rig_id"):
            write_calibration(copy, _VALUES, bad)


def test_write_calibration_twice_still_loads(tmp_path):
    """Review fix: the second writeback (recalibration weeks later) must
    produce a loadable file, not corrupt it."""
    src = Path(load_config().source_path)
    copy = tmp_path / "config.yaml"
    shutil.copy(src, copy)
    write_calibration(copy, _VALUES, "rig-v1")
    second = dict(_VALUES, yaw_offset_rad=-0.271828)  # negative too
    write_calibration(copy, second, "rig-v2")
    cfg = load_config(str(copy))
    assert cfg.calibration.yaw_offset_rad == pytest.approx(-0.271828)
    assert cfg.calibration.rig_id == "rig-v2"
    assert cfg.calibration.is_calibrated is True
