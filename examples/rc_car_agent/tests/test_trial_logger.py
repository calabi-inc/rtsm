"""TrialLogger schema tests — the JSONL is E1 paper data; its shape is a
contract (aggregate.py parses it in Phase H/I). Pins: three record types,
protocol-era fields present-but-null, tta/censoring semantics."""

import json

from config import load_config
from monitor import MonitorTick
from planner import PlanResult
from rtsm_client import PoseSample
from trial_logger import SCHEMA_VERSION, TrialLogger

CFG = load_config()


def _pose(ts=100.0, epoch=7):
    return PoseSample(xyz=[0.0, 0.3, 1.0], quaternion_xyzw=[0, 0, 0, 1],
                      timestamp=ts, fetched_at_mono=0.0, frame_epoch=epoch)


def _plan(status="ok"):
    return PlanResult(status=status, goal="go to the red mug", query="red mug",
                      target_id="mug-1", label="red mug", xyz_world=[0, 0.3, 3.0],
                      score=0.81, confirmed=True, stability=0.9,
                      planner_path="top1_no_llm", plan_pose=_pose(),
                      frame_epoch=7, reason=None)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _full_trial(tmp_path, result, detail="", elapsed=12.5, ticks=2):
    lg = TrialLogger(tmp_path, "t20260720-001", "go to the red mug", "rtsm", CFG)
    lg.log_plan(_plan())
    for i in range(ticks):
        tick = MonitorTick(status="ongoing", pose_fresh=True, car_x=0.0,
                           car_z=float(i), car_yaw=0.0,
                           ground_dist=3.0 - i, heading_err=0.05)
        lg.log_tick(0.1 * (i + 1), _pose(ts=100.0 + i), tick, 0.4, 0.6)
    lg.log_end(result, detail, elapsed_s=elapsed)
    return _read(lg.path)


def test_three_record_types_parse(tmp_path):
    records = _full_trial(tmp_path, "arrived")
    assert [r["type"] for r in records] == ["trial_start", "tick", "tick",
                                            "trial_end"]


def test_trial_start_schema(tmp_path):
    start = _full_trial(tmp_path, "arrived")[0]
    assert start["schema_version"] == SCHEMA_VERSION
    assert start["trial_id"] == "t20260720-001"
    assert start["condition"] == "rtsm"
    assert start["planner"]["planner_path"] == "top1_no_llm"
    assert start["planner"]["target_id"] == "mug-1"
    assert start["frame_epoch"] == 7
    assert start["plan_pose"]["frame_epoch"] == 7
    assert start["config"]["arrival_threshold_m"] == CFG.nav.arrival_threshold_m
    assert start["config"]["is_calibrated"] is CFG.calibration.is_calibrated
    assert "git_commit" in start["provenance"]         # best-effort, may be null
    assert start["rng_seed"] is None                   # baseline-only (Phase H)
    # Protocol-era fields: present-but-null, operator-filled post-trial.
    for key in ("layout_id", "start_pose_id", "session_id", "tape_cm",
                "video_file", "notes"):
        assert key in start and start[key] is None
    assert "target_age_at_plan_s" in start["planner"]   # coordinate-age audit
    assert "rtsm_stats" in start                        # throughput audit


def test_tick_schema(tmp_path):
    tick = _full_trial(tmp_path, "arrived")[1]
    assert tick["fresh"] is True
    assert tick["status"] == "ongoing"
    assert tick["pose"]["frame_epoch"] == 7
    assert tick["cmd"] == [0.4, 0.6]
    assert isinstance(tick["ground_dist_m"], float)
    assert isinstance(tick["heading_err_rad"], float)


def test_arrived_end_has_tta_not_censored(tmp_path):
    end = _full_trial(tmp_path, "arrived", elapsed=12.5)[-1]
    assert end["result"] == "arrived"
    assert end["tta_s"] == 12.5
    assert end["censored"] is False
    assert end["ticks"] == 2
    assert end["final_dist_m"] == 2.0              # last tick's ground_dist


def test_timeout_end_censored_no_tta(tmp_path):
    end = _full_trial(tmp_path, "timeout", elapsed=60.0)[-1]
    assert end["censored"] is True
    assert end["tta_s"] is None
    # trial_end is MACHINE-ONLY (review fix): no operator-filled fields —
    # interventions live in `result`, photos in the <task_id>.jpg convention.
    assert "human_interventions" not in end and "stop_photo" not in end


def test_same_trial_id_never_merges_files(tmp_path):
    """Append mode + an id collision (server restart resets the seq) must
    produce a SECOND file, never two trials merged into one JSONL."""
    a = TrialLogger(tmp_path, "t-dup", "g", "rtsm", CFG)
    a.log_plan(_plan()); a.log_end("arrived", elapsed_s=1.0)
    b = TrialLogger(tmp_path, "t-dup", "g", "rtsm", CFG)
    b.log_plan(_plan()); b.log_end("timeout", elapsed_s=60.0)
    assert a.path != b.path
    assert len(_read(a.path)) == 2 and len(_read(b.path)) == 2


def test_not_found_trial_no_ticks(tmp_path):
    lg = TrialLogger(tmp_path, "t20260720-002", "go to the unicorn", "rtsm", CFG)
    lg.log_plan(_plan(status="not_found"))
    lg.log_end("not_found", "no results", elapsed_s=0.4)
    records = _read(lg.path)
    assert [r["type"] for r in records] == ["trial_start", "trial_end"]
    assert records[-1]["final_dist_m"] is None
    assert records[-1]["censored"] is False


def test_writes_survive_close_and_are_flushed(tmp_path):
    lg = TrialLogger(tmp_path, "t20260720-003", "g", "baseline", CFG)
    lg.log_plan(_plan())
    # Flushed per record: readable BEFORE log_end (crash resilience).
    assert len(_read(lg.path)) == 1
    lg.log_end("estopped", "hard e-stop", elapsed_s=3.0)
    assert lg.path.read_text().count("trial_end") == 1
    lg.log_tick(9.9, _pose(), MonitorTick(status="ongoing"), 0, 0)  # after close
    assert len(_read(lg.path)) == 2                # silently dropped
