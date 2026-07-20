"""nav.py tests — the steer-sign pins (audit safety pins 1-2), the hold/
heartbeat contract (pin 3), bounded blind-stop (pin 4), e-stop no-recommand
(pin 6), preempt-stops-first (pin 7), and pure-python closed-loop
convergence against the kinematic fake car (no HTTP)."""

import math
import threading
import time
from dataclasses import replace

import pytest

from config import load_config
from fake_car import FakeCar
from nav import NavRunner, drive_command
from planner import PlanResult
from rtsm_client import RtsmClient

BASE_CFG = load_config()
FAST_NAV = replace(
    BASE_CFG.nav,
    tick_s=0.01,
    poll_hz=50.0,
    stale_abort_s=0.5,
    timeout_rtsm_s=20.0,
)
FAST_CFG = replace(BASE_CFG, nav=FAST_NAV)
NAV = BASE_CFG.nav


# ── drive_command: the steer-sign pins ───────────────────────────────────


def test_steer_sign_err_pos_right_faster():
    """err > 0 (target LEFT/CCW) -> right wheel faster. THE pin: reusing
    teleop's arcade_mix with steer=+err would invert this and drive away
    from the target."""
    left, right, mode = drive_command(0.5, NAV)
    assert mode == "translate"
    assert right > left
    d = min(NAV.max_turn, NAV.kp_steer * 0.5)     # differential clamps first
    assert left == pytest.approx(NAV.max_speed - d)
    assert right == pytest.approx(min(1.0, NAV.max_speed + d))


def test_steer_sign_err_neg_left_faster():
    left, right, mode = drive_command(-0.5, NAV)
    assert mode == "translate"
    assert left > right


def test_steer_zero_err_straight():
    left, right, _ = drive_command(0.0, NAV)
    assert left == right == pytest.approx(NAV.max_speed)


def test_rotate_in_place_ccw_when_target_far_left():
    err = math.radians(NAV.rotate_in_place_deg) + 0.2
    left, right, mode = drive_command(err, NAV)
    assert mode == "rotate"
    assert left < 0 < right                       # CCW: left back, right fwd


def test_rotate_in_place_cw_when_target_far_right():
    err = -(math.radians(NAV.rotate_in_place_deg) + 0.2)
    left, right, mode = drive_command(err, NAV)
    assert mode == "rotate"
    assert right < 0 < left


def test_outputs_always_clamped():
    for err in (-3.5, -1.0, 0.0, 1.0, 3.5):
        left, right, _ = drive_command(err, NAV)
        assert -1.0 <= left <= 1.0 and -1.0 <= right <= 1.0

def test_differential_clamped_to_max_turn():
    err = math.radians(NAV.rotate_in_place_deg) - 0.01   # translate mode
    left, right, mode = drive_command(err, NAV)
    assert mode == "translate"
    assert (right - left) == pytest.approx(2 * NAV.max_turn)


# ── fakes for loop-contract tests ────────────────────────────────────────


class FakeBridge:
    def __init__(self):
        self.drive_calls = []            # (t_mono, left, right, accepted)
        self.stop_calls = []
        self.latched = False

    def drive(self, left, right):
        ok = not self.latched
        self.drive_calls.append((time.monotonic(), left, right, ok))
        return ok

    def stop(self, estop=False):
        if estop:
            self.latched = True
        self.stop_calls.append(time.monotonic())
        return True


class ScriptedRtsm:
    """get_robot_pose() delegates to a callable — scripts freshness,
    freezes, side effects, or a live FakeCar."""

    def __init__(self, fn):
        self._fn = fn

    def get_robot_pose(self):
        return self._fn()


def mk_plan(target=(0.0, 0.3, 3.0), plan_pose=None, epoch=None):
    return PlanResult(status="ok", goal="go to the red mug", query="red mug",
                      target_id="mug-1", xyz_world=list(target),
                      plan_pose=plan_pose, frame_epoch=epoch)


def run_nav(cfg, bridge, rtsm, plan, *, stop=None, preempt=None, cancel=None):
    events = dict(
        stop_event=stop or threading.Event(),
        preempt_event=preempt or threading.Event(),
        cancel_event=cancel or threading.Event(),
        shutdown_event=threading.Event(),
    )
    runner = NavRunner(cfg, bridge, rtsm, plan, "rtsm", logger=None, **events)
    return runner.run()


def _fresh_pose_then_frozen():
    """One fresh pose, then the identical (frozen) sample forever."""
    payload = {"xyz": [0.0, 0.3, 0.0], "quaternion_xyzw": [0, 0, 0, 1],
               "timestamp": 100.0, "frame_epoch": 7}
    return lambda: RtsmClient._parse_pose(dict(payload))


# ── hold rule: keep calling drive() while blind (audit pin 3) ────────────


def test_hold_keeps_calling_drive_until_bounded_stop():
    bridge = FakeBridge()
    rtsm = ScriptedRtsm(_fresh_pose_then_frozen())
    result, detail = run_nav(FAST_CFG, bridge, rtsm, mk_plan())

    assert result == "stale_stop"                 # bounded, not trial-timeout
    assert "blind" in detail
    # During the ~0.5 s hold the loop must KEEP calling drive() every tick
    # (the bridge heartbeat is what feeds the 300 ms firmware watchdog).
    assert len(bridge.drive_calls) >= 15
    assert bridge.stop_calls, "abort must safe-stop the car"


def test_no_motion_before_first_fresh_pose():
    bridge = FakeBridge()
    rtsm = ScriptedRtsm(lambda: None)             # RTSM never serves a pose
    result, _ = run_nav(FAST_CFG, bridge, rtsm, mk_plan())
    assert result == "stale_stop"
    assert bridge.drive_calls == []               # car never energized


# ── e-stop: exit without re-commanding motors (audit pin 6) ──────────────


def test_estop_exits_without_recommand():
    bridge = FakeBridge()
    stop = threading.Event()
    fired = {"t": None}
    inner = _fresh_pose_then_frozen()

    def poses():
        pose = inner()
        if len(bridge.drive_calls) >= 3 and not stop.is_set():
            # Simulate the EstopMonitor: latch the bridge FIRST, then wake.
            bridge.latched = True
            stop.set()
            fired["t"] = time.monotonic()
        return pose

    result, _ = run_nav(FAST_CFG, bridge, ScriptedRtsm(poses), mk_plan(),
                        stop=stop)
    assert result == "estopped"
    # Nav must NOT re-command: no bridge.stop() from the nav path (the
    # e-stop channel already stopped the car) ...
    assert bridge.stop_calls == []
    # ... and any in-flight drive attempts after the latch were rejected
    # no-ops nav did not retry-loop against.
    after = [c for c in bridge.drive_calls if c[0] >= fired["t"]]
    assert all(c[3] is False for c in after)
    assert len(after) <= 3


# ── preempt / cancel: safe stop BEFORE returning (audit pin 7) ───────────


def test_preempt_stops_before_return_and_clears_flag():
    bridge = FakeBridge()
    preempt = threading.Event()
    preempt.set()
    result, _ = run_nav(FAST_CFG, bridge, ScriptedRtsm(_fresh_pose_then_frozen()),
                        mk_plan(), preempt=preempt)
    assert result == "preempted"
    assert bridge.stop_calls, "preempt must safe-stop before the goal swap"
    assert not preempt.is_set()


def test_cancel_stops_and_clears():
    bridge = FakeBridge()
    cancel = threading.Event()
    cancel.set()
    result, _ = run_nav(FAST_CFG, bridge, ScriptedRtsm(_fresh_pose_then_frozen()),
                        mk_plan(), cancel=cancel)
    assert result == "cancelled"
    assert bridge.stop_calls
    assert not cancel.is_set()


# ── closed-loop convergence against the kinematic sim (no HTTP) ──────────


def _wire_fake_car(car: FakeCar):
    bridge = FakeBridge()
    orig_drive = bridge.drive

    def drive(left, right):
        car.set_drive(left, right)
        return orig_drive(left, right)

    bridge.drive = drive
    rtsm = ScriptedRtsm(lambda: RtsmClient._parse_pose(car.pose_payload()))
    return bridge, rtsm


def test_closed_loop_converges_forward():
    car = FakeCar(x=0.0, z=0.0, yaw=0.0)          # facing +Z
    target = [0.3, 0.3, 1.5]
    bridge, rtsm = _wire_fake_car(car)
    result, _ = run_nav(FAST_CFG, bridge, rtsm, mk_plan(target=target, epoch=7))
    assert result == "arrived"
    assert car.ground_dist_to(target) <= FAST_CFG.nav.arrival_threshold_m + 0.15
    assert bridge.stop_calls, "arrival must stop the car"


def test_closed_loop_converges_from_behind():
    """Target behind the car: |err| > 90° -> rotate in place, then drive."""
    car = FakeCar(x=0.0, z=0.0, yaw=0.0)
    target = [0.0, 0.3, -1.5]                      # directly behind
    bridge, rtsm = _wire_fake_car(car)
    result, _ = run_nav(FAST_CFG, bridge, rtsm, mk_plan(target=target, epoch=7))
    assert result == "arrived"
    rotate_ticks = [c for c in bridge.drive_calls
                    if (c[1] < 0 < c[2]) or (c[2] < 0 < c[1])]
    assert rotate_ticks, "should have rotated in place first"


# ── trial-log pairing (review fix: cmd derived from the SAME tick) ───────


class _RecLogger:
    def __init__(self):
        self.ticks = []

    def log_tick(self, t, pose, tick, left, right):
        self.ticks.append((tick, left, right))


def test_logged_cmd_pairs_with_same_ticks_heading_err():
    """E1 analysis pairs fields per JSONL line — each fresh 'ongoing' tick
    must log the command DERIVED FROM its own heading_err (a one-poll lag
    would produce phantom steer-sign violations in the paper data)."""
    car = FakeCar(x=0.0, z=0.0, yaw=0.0)
    target = [0.3, 0.3, 1.5]
    bridge, rtsm = _wire_fake_car(car)
    logger = _RecLogger()
    events = dict(stop_event=threading.Event(), preempt_event=threading.Event(),
                  cancel_event=threading.Event(), shutdown_event=threading.Event())
    runner = NavRunner(FAST_CFG, bridge, rtsm, mk_plan(target=target, epoch=7),
                       "rtsm", logger=logger, **events)
    result, _ = runner.run()
    assert result == "arrived"

    fresh = [(t, l, r) for (t, l, r) in logger.ticks
             if t.status == "ongoing" and t.pose_fresh and t.heading_err is not None]
    assert len(fresh) >= 5
    for tick, left, right in fresh:
        exp_l, exp_r, _mode = drive_command(tick.heading_err, FAST_CFG.nav)
        assert left == pytest.approx(exp_l), "logged cmd lags its heading_err"
        assert right == pytest.approx(exp_r)
