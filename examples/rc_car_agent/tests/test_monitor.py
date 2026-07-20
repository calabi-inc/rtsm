"""MissionMonitor unit tests — every verdict rule, fully clock-injected
(no sleeps). These pin the audit's safety rules: bounded blind-driving,
the frame_epoch None matrix, discontinuity, drift, and arrival authority."""

import math
from dataclasses import replace

import pytest

from config import load_config
from monitor import MissionMonitor
from rtsm_client import PoseSample


CFG = load_config()
NAV = replace(
    CFG.nav,
    arrival_threshold_m=0.40,
    stale_abort_s=2.5,
    drift_margin_m=0.8,
    discontinuity_base_m=0.5,
    discontinuity_rate_mps=1.0,
    timeout_rtsm_s=60.0,
)
CAL = CFG.calibration          # identity (yaw_offset 0, lever arm 0)


def mk_pose(x, z, yaw=0.0, ts=100.0, epoch=None):
    """Camera pose at ground (x, z) heading `yaw` — quaternion is a
    rotation about world +Y, matching the OpenCV-forward convention."""
    half = 0.5 * yaw
    return PoseSample(
        xyz=[float(x), 0.3, float(z)],
        quaternion_xyzw=[0.0, math.sin(half), 0.0, math.cos(half)],
        timestamp=float(ts),
        fetched_at_mono=0.0,
        frame_epoch=epoch,
    )


def mk_monitor(target=(0.0, 0.0, 3.0), plan_pose=None, plan_epoch=None,
               timeout_s=60.0, nav=NAV):
    return MissionMonitor(nav=nav, calibration=CAL, target_xyz=list(target),
                          plan_pose=plan_pose, plan_epoch=plan_epoch,
                          timeout_s=timeout_s, t0_mono=0.0)


# ── arrival (sole authority) ─────────────────────────────────────────────


def test_ongoing_with_geometry_when_far():
    m = mk_monitor(target=(0.0, 0.0, 3.0))
    t = m.assess(mk_pose(0, 0, yaw=0.0, ts=1.0), now_mono=0.1)
    assert t.status == "ongoing" and t.pose_fresh
    assert t.ground_dist == pytest.approx(3.0)
    assert t.heading_err == pytest.approx(0.0)


def test_arrived_within_threshold():
    m = mk_monitor(target=(0.0, 0.0, 3.0))
    t = m.assess(mk_pose(0, 2.7, ts=1.0), now_mono=0.1)
    assert t.status == "arrived"
    assert t.ground_dist == pytest.approx(0.3)


def test_no_arrival_on_stale_pose_even_if_near():
    """Arrival must be judged on FRESH geometry only."""
    m = mk_monitor(target=(0.0, 0.0, 3.0))
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.1)          # fresh, far
    t = m.assess(mk_pose(0, 2.9, ts=1.0), now_mono=0.2)    # same ts -> stale
    assert t.status == "ongoing" and not t.pose_fresh


def test_heading_err_sign_target_left_is_positive():
    """Target to the LEFT of the car's heading -> err > 0 (CCW)."""
    m = mk_monitor(target=(1.0, 0.0, 1.0))                  # 45° left of +Z... at +X
    t = m.assess(mk_pose(0, 0, yaw=0.0, ts=1.0), now_mono=0.1)
    # facing +Z (yaw 0), target at (+X, +Z): +X is the LEFT hand (fact #3)
    assert t.heading_err == pytest.approx(math.pi / 4)


# ── timeout ──────────────────────────────────────────────────────────────


def test_timeout_beats_everything():
    m = mk_monitor(timeout_s=10.0)
    t = m.assess(mk_pose(0, 2.7, ts=1.0), now_mono=10.5)   # would be arrived
    assert t.status == "timeout"


# ── bounded blind-driving (audit safety pin) ─────────────────────────────


def test_stale_holds_then_aborts_within_stale_abort_s():
    m = mk_monitor()
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.0)          # fresh reference
    hold = m.assess(mk_pose(0, 0, ts=1.0), now_mono=1.2)   # frozen ts -> hold
    assert hold.status == "ongoing" and not hold.pose_fresh
    aborted = m.assess(mk_pose(0, 0, ts=1.0), now_mono=2.6)
    assert aborted.status == "stale_stop"                  # < trial timeout!


def test_no_pose_at_all_aborts_bounded():
    m = mk_monitor()
    assert m.assess(None, now_mono=1.0).status == "ongoing"
    assert m.assess(None, now_mono=2.6).status == "stale_stop"


def test_frozen_polls_detail():
    m = mk_monitor()
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.0)
    t = None
    for i in range(NAV.pose_frozen_polls):
        t = m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.1 * (i + 1))
    assert t.status == "ongoing" and "frozen" in t.detail


def test_fresh_pose_resets_staleness_clock():
    m = mk_monitor()
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.0)
    m.assess(mk_pose(0, 0.2, ts=2.0), now_mono=2.0)        # fresh again
    t = m.assess(mk_pose(0, 0.2, ts=2.0), now_mono=3.5)    # 1.5 s blind < 2.5
    assert t.status == "ongoing"


# ── frame_epoch gate (audit safety pin: ints only, None never aborts) ────


def test_epoch_change_aborts():
    m = mk_monitor(plan_epoch=5)
    assert m.assess(mk_pose(0, 0, ts=1.0, epoch=5), 0.1).status == "ongoing"
    t = m.assess(mk_pose(0, 0.1, ts=2.0, epoch=6), 0.3)
    assert t.status == "frame_reset" and "epoch" in t.detail


def test_epoch_none_sides_never_abort():
    m = mk_monitor(plan_epoch=None)
    assert m.assess(mk_pose(0, 0, ts=1.0, epoch=None), 0.1).status == "ongoing"
    assert m.assess(mk_pose(0, 0.1, ts=2.0, epoch=None), 0.3).status == "ongoing"


def test_epoch_adopted_when_plan_epoch_none():
    """Plan-time epoch unknown -> first int seen becomes the reference, so
    a mid-drive session restart is STILL caught."""
    m = mk_monitor(plan_epoch=None)
    assert m.assess(mk_pose(0, 0, ts=1.0, epoch=3), 0.1).status == "ongoing"
    t = m.assess(mk_pose(0, 0.1, ts=2.0, epoch=4), 0.3)
    assert t.status == "frame_reset"


def test_pose_epoch_none_transient_does_not_abort():
    """Benign post-/reset transient: int -> None -> int(same) must ride
    through on the heuristic, never hard-abort."""
    m = mk_monitor(plan_epoch=5)
    assert m.assess(mk_pose(0, 0, ts=1.0, epoch=5), 0.1).status == "ongoing"
    assert m.assess(mk_pose(0, 0.1, ts=2.0, epoch=None), 0.3).status == "ongoing"
    assert m.assess(mk_pose(0, 0.2, ts=3.0, epoch=5), 0.5).status == "ongoing"


# ── pose-discontinuity heuristic (belt-and-braces) ───────────────────────


def test_teleport_aborts():
    m = mk_monitor()
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.0)
    t = m.assess(mk_pose(5.0, 0, ts=2.0), now_mono=0.1)    # 5 m in 0.1 s
    assert t.status == "frame_reset" and "jumped" in t.detail


def test_first_pose_checked_against_plan_pose():
    m = mk_monitor(plan_pose=mk_pose(0, 0, ts=0.5))
    t = m.assess(mk_pose(5.0, 0, ts=1.0), now_mono=0.2)
    assert t.status == "frame_reset"


def test_slow_motion_passes():
    m = mk_monitor()
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.0)
    t = m.assess(mk_pose(0, 0.1, ts=2.0), now_mono=0.2)    # 0.1 m in 0.2 s
    assert t.status == "ongoing"


def test_threshold_scales_with_blind_time():
    """After a long hold the car may legitimately have moved further —
    the threshold grows with dt instead of false-triggering."""
    m = mk_monitor()
    m.assess(mk_pose(0, 0, ts=1.0), now_mono=0.0)
    # 1.9 s later, 1.5 m away: threshold = 0.5 + 1.0*1.9 = 2.4 -> passes
    t = m.assess(mk_pose(0, 1.5, ts=2.0), now_mono=1.9)
    assert t.status == "ongoing"


# ── drift ────────────────────────────────────────────────────────────────


def test_drift_when_receding_beyond_margin():
    m = mk_monitor(target=(0.0, 0.0, 3.0))
    m.assess(mk_pose(0, 1.0, ts=1.0), now_mono=0.1)        # best = 2.0
    t = m.assess(mk_pose(0, 0.1, ts=2.0), now_mono=1.0)    # dist 2.9 > 2.0+0.8
    assert t.status == "drift"


def test_receding_within_margin_is_ongoing():
    m = mk_monitor(target=(0.0, 0.0, 3.0))
    m.assess(mk_pose(0, 1.0, ts=1.0), now_mono=0.1)        # best = 2.0
    t = m.assess(mk_pose(0, 0.5, ts=2.0), now_mono=0.6)    # dist 2.5 < 2.8
    assert t.status == "ongoing"


# ── degenerate heading / malformed pose robustness ───────────────────────


def test_camera_facing_down_is_a_fault():
    s = math.sin(math.pi / 4)
    pose = PoseSample(xyz=[0, 0.3, 0], quaternion_xyzw=[s, 0, 0, s],
                      timestamp=1.0, fetched_at_mono=0.0)   # pitch -90°
    m = mk_monitor()
    assert m.assess(pose, 0.1).status == "degenerate_heading"


def test_malformed_quaternion_is_a_verdict_not_a_crash():
    """Review fix: a zero-norm or wrong-length quaternion that reaches the
    monitor must yield a safe verdict, never an exception that kills the
    control loop (motors would then only stop via the firmware watchdog)."""
    m = mk_monitor()
    zero = PoseSample(xyz=[0, 0.3, 0], quaternion_xyzw=[0, 0, 0, 0],
                      timestamp=1.0, fetched_at_mono=0.0)
    assert m.assess(zero, 0.1).status == "degenerate_heading"
    short = PoseSample(xyz=[0, 0.3, 0], quaternion_xyzw=[0, 0, 0],
                       timestamp=2.0, fetched_at_mono=0.0)
    assert m.assess(short, 0.2).status == "degenerate_heading"


def test_parse_pose_rejects_malformed_payloads():
    """Boundary hardening: garbage payloads become None (-> bounded
    stale-hold path), never an invalid PoseSample."""
    from rtsm_client import RtsmClient
    ok = {"xyz": [0, 0.3, 0], "quaternion_xyzw": [0, 0, 0, 1],
          "timestamp": 1.0}
    assert RtsmClient._parse_pose(dict(ok)) is not None
    assert RtsmClient._parse_pose({**ok, "quaternion_xyzw": [0, 0, 0, 0]}) is None
    assert RtsmClient._parse_pose({**ok, "quaternion_xyzw": [0, 0, 1]}) is None
    assert RtsmClient._parse_pose({**ok, "xyz": [0, 0.3]}) is None
    assert RtsmClient._parse_pose({**ok, "timestamp": "not-a-number"}) is None


# ── dead-feed launch guard (review fix) ──────────────────────────────────


def test_feed_dead_since_plan_never_counts_as_fresh():
    """A feed that died BEFORE the mission still serves the plan-time
    timestamp — the first live poll must NOT be 'fresh' (the old behavior
    launched the car blind for up to stale_abort_s)."""
    plan_pose = mk_pose(0, 0, ts=100.0)
    m = mk_monitor(plan_pose=plan_pose)
    t = m.assess(mk_pose(0, 0, ts=100.0), now_mono=0.2)    # same ts as plan
    assert t.status == "ongoing" and not t.pose_fresh
    assert m.assess(mk_pose(0, 0, ts=100.0), now_mono=2.6).status == "stale_stop"


def test_live_feed_first_poll_is_fresh():
    plan_pose = mk_pose(0, 0, ts=100.0)
    m = mk_monitor(plan_pose=plan_pose)
    t = m.assess(mk_pose(0, 0.05, ts=100.2), now_mono=0.2)  # ts advanced
    assert t.status == "ongoing" and t.pose_fresh
