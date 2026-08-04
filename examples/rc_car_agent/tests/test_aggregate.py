"""aggregate.py tests — synthetic trial JSONLs through the real parser and
statistics: censoring at the cap, tape-based TCR vs believed, path
efficiency, Mann-Whitney sanity, speedup."""

import json
import math

import pytest

from aggregate import mann_whitney_u, parse_trial, summarize


def _write_trial(dirp, trial_id, condition, result, tta=None, elapsed=None,
                 tape_cm=None, ticks=None, target=(0.0, 0.3, 3.0),
                 start_xyz=(0.0, 0.3, 0.0), search_time=None):
    lines = [{
        "type": "trial_start", "schema_version": 1, "trial_id": trial_id,
        "condition": condition, "goal": "go to the red mug",
        "planner": {"planner_path": "top1_no_llm", "target_id": "mug-1",
                    "xyz_world": list(target)},
        "frame_epoch": 7,
        "plan_pose": {"xyz": list(start_xyz), "timestamp": 1.0, "frame_epoch": 7},
        "config": {}, "provenance": {"git_commit": "abc"}, "rng_seed": None,
        "layout_id": None, "start_pose_id": None, "tape_cm": tape_cm,
        "video_file": None, "notes": None,
    }]
    for t, xyz in (ticks or []):
        lines.append({"type": "tick", "t": t, "fresh": True,
                      "pose": {"xyz": list(xyz), "timestamp": 1.0 + t,
                               "frame_epoch": 7},
                      "status": "ongoing", "ground_dist_m": 1.0,
                      "heading_err_rad": 0.0, "cmd": [0.4, 0.6]})
    if search_time is not None:
        lines.append({"type": "event", "name": "baseline_acquired", "t": search_time,
                      "target_id": "mug-1", "search_time_s": search_time})
    lines.append({
        "type": "trial_end", "trial_id": trial_id, "result": result,
        "detail": "", "elapsed_s": elapsed if elapsed is not None else tta,
        "ticks": len(ticks or []), "final_dist_m": 0.3,
        "tta_s": tta, "censored": result == "timeout",
        "human_interventions": None, "stop_photo": None, "ended_at": "x",
    })
    p = dirp / f"{trial_id}.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return p


def test_parse_trial_fields(tmp_path):
    p = _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=12.5, tape_cm=30,
                     ticks=[(0.1, (0, 0.3, 0)), (0.2, (0, 0.3, 1.0)),
                            (0.3, (0, 0.3, 2.0))])
    t = parse_trial(p)
    assert t["condition"] == "rtsm" and t["result"] == "arrived"
    assert t["tta_s"] == 12.5 and t["tape_cm"] == 30
    assert t["path_len_m"] == pytest.approx(2.0)
    assert t["optimal_m"] == pytest.approx(3.0)


def test_malformed_file_skipped(tmp_path):
    (tmp_path / "junk.jsonl").write_text("not json\n", encoding="utf-8")
    _write_trial(tmp_path, "t-ok", "rtsm", "arrived", tta=5.0)
    s = summarize(tmp_path)
    assert s["n_files"] == 1


def test_censored_trials_enter_at_cap(tmp_path):
    _write_trial(tmp_path, "t-a1", "rtsm", "arrived", tta=10.0)
    _write_trial(tmp_path, "t-a2", "rtsm", "timeout", elapsed=60.0)
    _write_trial(tmp_path, "t-b1", "baseline", "arrived", tta=100.0)
    _write_trial(tmp_path, "t-b2", "baseline", "timeout", elapsed=180.0)
    s = summarize(tmp_path)
    r, b = s["conditions"]["rtsm"], s["conditions"]["baseline"]
    assert r["censored"] == 1 and b["censored"] == 1
    assert r["tta_median_s"] == pytest.approx((10.0 + 60.0) / 2)
    assert b["tta_median_s"] == pytest.approx((100.0 + 180.0) / 2)


def test_aborted_trials_excluded_from_tta_but_counted(tmp_path):
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0)
    _write_trial(tmp_path, "t-2", "rtsm", "estopped")
    s = summarize(tmp_path)
    r = s["conditions"]["rtsm"]
    assert r["aborted"] == 1
    assert r["tta_n"] == 1                          # only the arrival


def test_tcr_tape_vs_believed(tmp_path):
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0, tape_cm=30)
    _write_trial(tmp_path, "t-2", "rtsm", "arrived", tta=11.0, tape_cm=80)
    _write_trial(tmp_path, "t-3", "rtsm", "arrived", tta=12.0)   # not yet taped
    s = summarize(tmp_path, tape_success_cm=50.0)
    r = s["conditions"]["rtsm"]
    assert r["tcr_tape"] == pytest.approx(0.5)      # 30 in, 80 out
    assert r["tcr_tape_n"] == 2                     # untaped excluded, visibly
    assert r["tcr_believed"] == pytest.approx(1.0)  # robot believed all three


def test_path_efficiency(tmp_path):
    straight = [(0.1 * i, (0.0, 0.3, 0.3 * i)) for i in range(11)]  # 3.0 m direct
    _write_trial(tmp_path, "t-s", "rtsm", "arrived", tta=5.0, ticks=straight)
    zigzag = [(0.1 * i, (0.5 * (i % 2), 0.3, 0.3 * i)) for i in range(11)]
    _write_trial(tmp_path, "t-z", "rtsm", "arrived", tta=6.0, ticks=zigzag)
    s = summarize(tmp_path)
    pes = s["conditions"]["rtsm"]["pe_median"]
    assert pes > 1.0                                # zigzag pulls the median up


def test_search_time_surfaced(tmp_path):
    _write_trial(tmp_path, "t-b", "baseline", "arrived", tta=90.0, search_time=75.0)
    s = summarize(tmp_path)
    assert s["conditions"]["baseline"]["search_time_median_s"] == pytest.approx(75.0)


def test_speedup_median(tmp_path):
    for i, tta in enumerate([10.0, 12.0, 14.0]):
        _write_trial(tmp_path, f"t-r{i}", "rtsm", "arrived", tta=tta)
    for i, tta in enumerate([100.0, 120.0, 140.0]):
        _write_trial(tmp_path, f"t-b{i}", "baseline", "arrived", tta=tta)
    s = summarize(tmp_path)
    assert s["tta_speedup_median"] == pytest.approx(10.0)
    assert s["mann_whitney_tta"]["p_one_sided"] < 0.1


# ── Mann-Whitney sanity ──────────────────────────────────────────────────


def test_mw_separated_samples_small_p():
    out = mann_whitney_u([1, 2, 3, 4, 5], [10, 11, 12, 13, 14])
    assert out["U"] == 0.0
    assert out["p_one_sided"] < 0.01


def test_mw_identical_samples_p_half():
    out = mann_whitney_u([5, 5, 5, 5], [5, 5, 5, 5])
    assert 0.3 < out["p_one_sided"] <= 1.0


def test_mw_reversed_samples_large_p():
    out = mann_whitney_u([10, 11, 12], [1, 2, 3])
    assert out["p_one_sided"] > 0.9


def test_mw_empty_side_is_nan():
    assert math.isnan(mann_whitney_u([], [1, 2])["p_one_sided"])
