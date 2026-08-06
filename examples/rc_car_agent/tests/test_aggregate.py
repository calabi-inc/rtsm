"""aggregate.py tests — synthetic trial JSONLs in the EXACT shape the
server writes (baseline trial_starts carry no target/plan_pose; the
baseline_acquired event does), through the real parser and statistics.
Includes the review's regression scenarios: the anti-conservative
censoring inversion, the TCR denominator inversion, and abort exclusion."""

import json
import math

import pytest

from aggregate import (COMMON_HORIZON_S, mann_whitney_u, parse_trial,
                       summarize)


def _write_trial(dirp, trial_id, condition, result, tta=None, elapsed=None,
                 tape_cm=None, ticks=None, target=(0.0, 0.3, 3.0),
                 start_xyz=(0.0, 0.3, 0.0), search_time=None, budget=None,
                 layout=None, notes=None, calibrated=True, rig="test-rig"):
    """Server-faithful trial JSONL. Memory trials carry plan_pose+target in
    trial_start; baseline trials carry them ONLY on the acquired event."""
    is_baseline = condition == "baseline"
    if budget is None:
        budget = 180.0 if is_baseline else 60.0
    lines = [{
        "type": "trial_start", "schema_version": 1, "trial_id": trial_id,
        "condition": condition, "goal": "go to the red mug",
        "planner": {"planner_path": ("baseline_fresh" if is_baseline
                                     else "top1_no_llm"),
                    "target_id": None if is_baseline else "mug-1",
                    "xyz_world": None if is_baseline else list(target)},
        "frame_epoch": 7,
        "plan_pose": (None if is_baseline
                      else {"xyz": list(start_xyz), "timestamp": 1.0,
                            "frame_epoch": 7}),
        "config": {"timeout_rtsm_s": 60.0 if not is_baseline else 60.0,
                   "timeout_baseline_s": budget if is_baseline else 180.0,
                   **({"timeout_rtsm_s": budget} if not is_baseline else {}),
                   "is_calibrated": calibrated, "rig_id": rig},
        "provenance": {"git_commit": "abc"}, "rng_seed": 7 if is_baseline else None,
        "layout_id": layout, "start_pose_id": None, "session_id": None,
        "tape_cm": tape_cm, "video_file": None, "notes": notes,
    }]
    if search_time is not None:
        lines.append({"type": "event", "name": "baseline_acquired",
                      "t": search_time, "target_id": "mug-1",
                      "xyz_world": list(target),
                      "pose": {"xyz": list(start_xyz), "timestamp": 40.0,
                               "frame_epoch": 7},
                      "hit_age_s": 0.4, "n_fresh": 1,
                      "planner_path": "baseline_fresh_top1", "reason": None,
                      "sweeps": 2, "search_time_s": search_time})
    for t, xyz in (ticks or []):
        lines.append({"type": "tick", "t": t, "fresh": True,
                      "pose": {"xyz": list(xyz), "timestamp": 1.0 + t,
                               "frame_epoch": 7},
                      "status": "ongoing", "ground_dist_m": 1.0,
                      "heading_err_rad": 0.0, "cmd": [0.4, 0.6]})
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
    assert t["budget_s"] == 60.0


def test_baseline_pe_denominator_from_event(tmp_path):
    """Review fix: real baseline trial_starts carry NO target/plan_pose —
    the PE denominator must come from the baseline_acquired event, or the
    paper's PE comparison is silently one-armed."""
    p = _write_trial(tmp_path, "t-b", "baseline", "arrived", tta=90.0,
                     search_time=75.0,
                     ticks=[(76.0, (0, 0.3, 0)), (80.0, (0, 0.3, 1.5)),
                            (84.0, (0, 0.3, 3.0))])
    t = parse_trial(p)
    assert t["optimal_m"] == pytest.approx(3.0)      # event pose -> target
    assert t["path_len_m"] == pytest.approx(3.0)
    s = summarize(tmp_path)
    assert s["conditions"]["baseline"]["pe_median"] == pytest.approx(1.0)


def test_malformed_file_skipped(tmp_path):
    (tmp_path / "junk.jsonl").write_text("not json\n", encoding="utf-8")
    _write_trial(tmp_path, "t-ok", "rtsm", "arrived", tta=5.0)
    s = summarize(tmp_path)
    assert s["n_files"] == 1


def test_off_protocol_budget_excluded_and_counted(tmp_path):
    """Review fix: a pilot file run at a different budget must not
    contaminate the aggregate."""
    _write_trial(tmp_path, "t-ok", "rtsm", "arrived", tta=5.0)
    _write_trial(tmp_path, "t-pilot", "rtsm", "timeout", elapsed=90.0,
                 budget=90.0)
    s = summarize(tmp_path)
    assert s["n_files"] == 1
    assert s["off_protocol_n"] == 1


# ── exclusion machinery (protocol-review fixes) ──────────────────────────


def test_invalid_marked_trials_excluded_and_counted(tmp_path):
    """The protocol's INVALID marker must actually exclude — a phone-crash
    trial the operator voided must not enter TCR/TTA/MW."""
    _write_trial(tmp_path, "t-ok", "baseline", "arrived", tta=90.0,
                 search_time=70.0)
    _write_trial(tmp_path, "t-bad", "baseline", "stale_stop", elapsed=95.0,
                 notes="INVALID — app crash mid-trial")
    s = summarize(tmp_path)
    assert s["n_files"] == 1
    assert s["excluded"]["invalid_marked"] == 1
    assert s["conditions"]["baseline"]["n"] == 1        # only the valid one


def test_operator_results_excluded_and_counted(tmp_path):
    """cancelled/preempted/shutdown are interventions, not failures —
    never failures-at-cap (the shakedown-debris scenario)."""
    _write_trial(tmp_path, "t-ok", "rtsm", "arrived", tta=10.0)
    _write_trial(tmp_path, "t-c", "rtsm", "cancelled", elapsed=5.0)
    _write_trial(tmp_path, "t-p", "rtsm", "preempted", elapsed=3.0)
    s = summarize(tmp_path)
    assert s["excluded"]["operator_result"] == 2
    assert s["conditions"]["rtsm"]["tta_n"] == 1        # no phantom failures


def test_uncalibrated_trials_excluded(tmp_path):
    _write_trial(tmp_path, "t-ok", "rtsm", "arrived", tta=10.0)
    _write_trial(tmp_path, "t-uncal", "rtsm", "arrived", tta=9.0,
                 calibrated=False, rig=None)
    s = summarize(tmp_path)
    assert s["excluded"]["uncalibrated"] == 1
    assert s["n_files"] == 1


# ── verdict-gated TCR (pre-registered threshold semantics) ───────────────


def test_headline_tcr_requires_arrival_verdict(tmp_path):
    """A drift-abort that happens to halt at 45 cm is NOT a completion in
    the headline; it IS counted in the any-verdict secondary."""
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0, tape_cm=30)
    _write_trial(tmp_path, "t-2", "rtsm", "drift", elapsed=20.0, tape_cm=45)
    s = summarize(tmp_path)
    r = s["conditions"]["rtsm"]
    assert r["tcr_tape"] == pytest.approx(0.5)          # only the arrival
    assert r["tcr_tape_any_verdict"] == pytest.approx(1.0)
    assert set(r["tcr_sweep"].keys()) == {"40cm", "50cm", "60cm"}


def test_terminal_error_decomposition(tmp_path):
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0, tape_cm=60)
    # final_dist_m is 0.3 in the fixture end record -> |0.3 - 0.6| = 0.3
    s = summarize(tmp_path)
    r = s["conditions"]["rtsm"]
    assert r["terminal_err_median_m"] == pytest.approx(0.3)
    assert r["believed_arrived_tape_failed"] == 1       # 60 > 50 threshold


# ── clustering robustness (per-layout sign test) ─────────────────────────


def test_sign_test_by_layout(tmp_path):
    for lid, (r_tta, b_tta) in {"L1": (10.0, 100.0), "L2": (12.0, 120.0),
                                "L3": (14.0, 90.0)}.items():
        _write_trial(tmp_path, f"t-r-{lid}", "rtsm", "arrived", tta=r_tta,
                     layout=lid)
        _write_trial(tmp_path, f"t-b-{lid}", "baseline", "arrived", tta=b_tta,
                     layout=lid, search_time=b_tta - 15)
    s = summarize(tmp_path)
    st = s["sign_test_by_layout"]
    assert st["memory_wins"] == 3 and st["of_layouts"] == 3
    assert st["p_one_sided"] == pytest.approx(1 / 8)    # exact binomial 3/3


# ── censoring (the review's CRITICAL finding) ────────────────────────────


def test_failed_memory_trials_cannot_beat_successful_baseline():
    """Anti-conservative inversion regression: 3 rtsm TIMEOUTS vs 3
    baseline ARRIVALS slower than 60 s. The old per-condition-cap ranks
    declared memory significantly faster (p=0.032) with ZERO memory
    arrivals; at the common horizon the failures rank at/above every
    baseline success and p must be large."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        dirp = Path(d)
        for i in range(3):
            _write_trial(dirp, f"t-r{i}", "rtsm", "timeout", elapsed=60.0)
        for i, tta in enumerate([100.0, 110.0, 120.0]):
            _write_trial(dirp, f"t-b{i}", "baseline", "arrived", tta=tta,
                         search_time=80.0)
        s = summarize(dirp)
        assert s["mann_whitney_tta"]["p_one_sided"] > 0.5
        assert s["mann_whitney_tta_arrivals_only"]["U"] is None  # no rtsm arrivals


def test_censored_enter_per_condition_cap_for_medians(tmp_path):
    _write_trial(tmp_path, "t-a1", "rtsm", "arrived", tta=10.0)
    _write_trial(tmp_path, "t-a2", "rtsm", "timeout", elapsed=60.0)
    _write_trial(tmp_path, "t-b1", "baseline", "arrived", tta=100.0,
                 search_time=70.0)
    _write_trial(tmp_path, "t-b2", "baseline", "timeout", elapsed=180.0)
    s = summarize(tmp_path)
    assert s["conditions"]["rtsm"]["tta_median_s"] == pytest.approx(35.0)
    assert s["conditions"]["baseline"]["tta_median_s"] == pytest.approx(140.0)


def test_aborted_trials_enter_at_cap_not_excluded(tmp_path):
    """Review fix: an e-stop is a failure-before-arrival — silently
    dropping it from TTA deletes one arm's failures from the statistic."""
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0)
    _write_trial(tmp_path, "t-2", "rtsm", "estopped", elapsed=50.0)
    s = summarize(tmp_path)
    r = s["conditions"]["rtsm"]
    assert r["aborted"] == 1
    assert r["tta_n"] == 2                          # abort included at cap
    assert r["tta_median_s"] == pytest.approx((10.0 + 60.0) / 2)


# ── TCR (the review's denominator inversion) ─────────────────────────────


def test_tcr_untaped_failures_count_in_denominator():
    """Inversion regression: 8 untaped timeouts + 2 taped arrivals @ 20 cm
    must give TCR 20%, not 100%."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        dirp = Path(d)
        for i in range(8):
            _write_trial(dirp, f"t-b{i}", "baseline", "timeout", elapsed=180.0)
        for i in range(2):
            _write_trial(dirp, f"t-ba{i}", "baseline", "arrived", tta=100.0,
                         tape_cm=20, search_time=70.0)
        s = summarize(dirp)
        b = s["conditions"]["baseline"]
        assert b["tcr_tape"] == pytest.approx(0.2)
        assert b["tcr_tape_denom"] == 10


def test_tcr_untaped_arrival_reported_not_silently_dropped(tmp_path):
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0, tape_cm=30)
    _write_trial(tmp_path, "t-2", "rtsm", "arrived", tta=11.0, tape_cm=80)
    _write_trial(tmp_path, "t-3", "rtsm", "arrived", tta=12.0)   # not yet taped
    s = summarize(tmp_path, tape_success_cm=50.0)
    r = s["conditions"]["rtsm"]
    assert r["tcr_tape"] == pytest.approx(0.5)      # 30 in, 80 out
    assert r["tcr_tape_denom"] == 2
    assert r["tcr_untaped_arrivals"] == 1           # loud, not silent
    assert r["tcr_believed"] == pytest.approx(1.0)


def test_non_numeric_tape_skipped_and_counted(tmp_path):
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=10.0, tape_cm="n/a")
    _write_trial(tmp_path, "t-2", "rtsm", "arrived", tta=11.0, tape_cm=30)
    s = summarize(tmp_path)                          # must not raise
    r = s["conditions"]["rtsm"]
    assert r["tape_invalid_n"] == 1
    assert r["tcr_tape"] == pytest.approx(1.0)
    assert r["tcr_tape_denom"] == 1


# ── misc surfaces ────────────────────────────────────────────────────────


def test_search_time_surfaced(tmp_path):
    _write_trial(tmp_path, "t-b", "baseline", "arrived", tta=90.0,
                 search_time=75.0)
    s = summarize(tmp_path)
    assert s["conditions"]["baseline"]["search_time_median_s"] == pytest.approx(75.0)
    assert s["conditions"]["baseline"]["planner_paths"] == {"baseline_fresh": 1}


def test_speedup_median(tmp_path):
    for i, tta in enumerate([10.0, 12.0, 14.0]):
        _write_trial(tmp_path, f"t-r{i}", "rtsm", "arrived", tta=tta)
    for i, tta in enumerate([100.0, 120.0, 140.0]):
        _write_trial(tmp_path, f"t-b{i}", "baseline", "arrived", tta=tta,
                     search_time=90.0)
    s = summarize(tmp_path)
    assert s["tta_speedup_median"] == pytest.approx(10.0)
    assert s["mann_whitney_tta"]["p_one_sided"] < 0.1


def test_summary_json_serializable_with_one_armed_data(tmp_path):
    """Review fix: mid-collection (one condition only) the summary must
    stay valid strict JSON — no bare NaN."""
    _write_trial(tmp_path, "t-1", "rtsm", "arrived", tta=5.0)
    s = summarize(tmp_path)
    text = json.dumps(s, indent=2, allow_nan=False)  # raises on NaN
    assert json.loads(text)["mann_whitney_tta"]["U"] is None


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


def test_mw_empty_side_is_none():
    out = mann_whitney_u([], [1, 2])
    assert out["U"] is None and out["p_one_sided"] is None


def test_common_horizon_constant():
    assert COMMON_HORIZON_S == 180.0
