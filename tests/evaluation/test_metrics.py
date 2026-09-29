"""
P3 task 3 -- the metrics over synthetic ledgers and the report over synthetic
run directories (CPU only). Every metric is exercised on rows whose answer is
known by construction; the real numbers are gated on session1 (gate G3-3,
2026-09-29; the record is kept with the local development notes).

Contract under test:
  * clusters are leader clusters on the radius, deterministic and independent
    of the (random) object ids;
  * detection over views counts re-identified / duplicated / missed per
    cluster per in-frustum frame, and the gate mask removes the frames the
    deployed gate would have rejected from numerator AND denominator;
  * label disagreement, along-ray vs lateral scatter, duplicate spawns with
    the alive rule and the reason classes, revisits with the lower bound,
    worst moments that exist in the rows, admission counts equal to the
    frame-outcome histogram, transient objects;
  * the floor: spread over >= 3 runs, "insufficient" below;
  * write_report is a pure function of the run directories (byte-identical on
    regeneration) and the CLI reaches it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rtsm.evaluation import metrics as M
from rtsm.evaluation import report as RP
from rtsm.evaluation.ledger import observation_summary, outcome_histogram

import synth_ledger as S

S_ = 1_000_000_000          # one second in ns
T0 = 10 * S_


def P(**kw) -> M.MetricParams:
    base = dict(cluster_radius_m=0.5, cluster_radii_m=(0.25, 0.5, 1.0), revisit_gap_s=5.0, range_bin_m=0.5, min_obs_scatter=3,
                worst_n=10, moments_cap=100, proto_ttl_s=10.0, cos_min=0.9)
    base.update(kw)
    return M.MetricParams(**base)


def rows_of(*parts) -> list:
    out = S.meta_row()
    for part in parts:
        out += list(part)
    return out


# ───────────────────────────── clusters ─────────────────────────────

def test_leader_clusters_radius_order_and_sensitivity():
    rows = rows_of([
        S.obs_line(T0, "A", "created", [0.0, 0.0, 2.0]),
        S.obs_line(T0 + 1 * S_, "B", "created", [0.1, 0.0, 2.0]),
        S.obs_line(T0 + 2 * S_, "C", "created", [1.0, 0.0, 2.0]),
    ])
    tracks = M.object_tracks(rows, [])
    c05 = M.leader_clusters(tracks, 0.5)
    assert [c.members for c in c05] == [["A", "B"], ["C"]] and c05[0].leader == "A"
    assert len(M.leader_clusters(tracks, 0.05)) == 3 and len(M.leader_clusters(tracks, 2.0)) == 1
    m = M.compute_metrics(rows, S.summary_for([]), P())
    n = m["scalars"]
    assert n["clusters.n"] == 2 and n["clusters.duplicates_all"] == 1
    assert n["clusters.n_at_radius.0.25"] >= n["clusters.n_at_radius.0.5"] >= n["clusters.n_at_radius.1"]


def test_clusters_do_not_depend_on_object_ids():
    """Two runs with identical ledgers but different (random) ids must cluster identically:
    the creation-order tie-break is the file position, never the id."""
    def run(ids):
        a, b, c, d = ids
        rows = rows_of([
            S.obs_line(T0, a, "created", [0.0, 0.0, 2.0], cand_idx=0),
            S.obs_line(T0, b, "created", [0.4, 0.0, 2.0], cand_idx=1),        # same stamp as a: joins a's cluster
            S.obs_line(T0, c, "created", [0.8, 0.0, 2.0], cand_idx=2),        # 0.8 from a, 0.4 from b: leader is a -> new cluster
            S.obs_line(T0 + S_, d, "created", [1.1, 0.0, 2.0], cand_idx=0),   # joins c
        ])
        m = M.compute_metrics(rows, S.summary_for([]), P())
        return [c["n_members"] for c in m["clusters"]], m["scalars"]["clusters.n"]
    assert run(["a", "b", "c", "d"]) == run(["z9", "b2", "a1", "k0"]) == ([2, 2], 2)


# ───────────────────────────── detection over views ─────────────────────────────

def test_detection_over_views_and_gate_mask():
    # A created at T0; in the frustum on 4 later frames; matched on the first three, missed on the fourth,
    # and the fourth is a frame the deployed gate would have rejected (gate_shadow set).
    rows = rows_of(
        [S.dequeue_line(T0, is_keyframe=True), S.obs_line(T0, "A", "created", [0.0, 0.0, 2.0], frame_seq=1)],
    )
    for k in range(1, 5):
        ts = T0 + k * S_
        shadow = "skip" if k == 4 else None
        rows.append(S.dequeue_line(ts, reason="parallax", gate_shadow=shadow, frame_seq=k + 1))
        rows.append(S.view_line(ts, ["A"], frame_seq=k + 1))
        if k < 4:
            rows.append(S.obs_line(ts, "A", "matched", [0.0, 0.0, 2.0], cos_sim=0.95, dist_m=0.02, n_nearby=1, n_gate_survivors=1, max_cos=0.95, frame_seq=k + 1))
    m = M.compute_metrics(rows, S.summary_for([{"id": "A", "hits": 4, "confirmed": True}]), P())
    d = m["detection"]
    assert d["all"] == {"views": 4, "reidentified": 3, "duplicated": 0, "missed": 1, "detection_rate": 0.75, "reid_rate": 0.75}
    assert m["shadow_mode"] is True and d["masked"] == {"views": 3, "reidentified": 3, "duplicated": 0, "missed": 0, "detection_rate": 1.0, "reid_rate": 1.0}
    # id level equals the P2 view / obs join
    vj = observation_summary(rows)["view"]
    assert d["id_level"]["reidentified"] == vj["in_frustum_and_matched"] == 3 and d["id_level"]["missed"] == vj["in_frustum_and_missed"] == 1
    # the by-range table at 2.0 m expected depth
    assert d["by_range"]["2.00-2.50"]["views"] == 4 and d["by_range"]["2.00-2.50"]["reidentified"] == 3
    # per-frame records: the missed frame is the worst moment and carries the shadow
    assert m["worst_moments"][0]["t_sensor_ns"] == T0 + 4 * S_ and m["worst_moments"][0]["detail"]["gate_shadow"] == "skip"
    assert m["scalars"]["admission.shadowed"] == 1 and m["scalars"]["admission.processed"] == 5


def test_duplicated_counts_as_detected_but_not_reidentified():
    rows = rows_of([
        S.dequeue_line(T0), S.obs_line(T0, "A", "created", [0.0, 0.0, 2.0]),
        S.dequeue_line(T0 + S_), S.view_line(T0 + S_, ["A"]),
        S.obs_line(T0 + S_, "B", "created", [0.2, 0.0, 2.0], n_nearby=1, n_gate_survivors=0),   # a spawn on A's cluster instead of a match
    ])
    m = M.compute_metrics(rows, S.summary_for([{"id": "A"}, {"id": "B"}]), P())
    assert m["detection"]["all"] == {"views": 1, "reidentified": 0, "duplicated": 1, "missed": 0, "detection_rate": 1.0, "reid_rate": 0.0}
    assert m["scalars"]["duplicates.n"] == 1 and m["duplicate_spawns"][0]["dup_of"] == "A" and m["duplicate_spawns"][0]["original_in_view"] is True


# ───────────────────────────── labels ─────────────────────────────

def test_label_disagreement():
    rows = rows_of([
        S.obs_line(T0, "A", "created", [0, 0, 2], label="cup"),
        S.obs_line(T0 + S_, "A", "matched", [0, 0, 2], label="cup"),
        S.obs_line(T0 + 2 * S_, "A", "matched", [0, 0, 2], label="mug"),
        S.obs_line(T0, "B", "created", [3, 0, 2], label="box"),                       # single observation -> 0, not in the >= 2 stats
        S.obs_line(T0 + S_, "C", "created", [6, 0, 2], label="lamp"),
        S.obs_line(T0 + 2 * S_, "C", "matched", [6, 0, 2], label="lamp"),
    ])
    m = M.compute_metrics(rows, S.summary_for([{"id": "A", "label_primary": "cup"}, {"id": "B", "label_primary": "box"}]), P())
    by_id = {o["id"]: o for o in m["objects"]}
    assert by_id["A"]["label_disagreement"] == pytest.approx(1 / 3) and by_id["A"]["n_distinct_labels"] == 2 and by_id["A"]["label_modal"] == "cup"
    assert by_id["B"]["label_disagreement"] == 0.0 and by_id["C"]["label_disagreement"] == 0.0
    s = m["scalars"]
    assert s["labels.tracks_ge2"] == 2 and s["labels.frac_tracks_disagreeing"] == 0.5 and s["labels.disagreement_mean"] == pytest.approx(1 / 6, abs=1e-5)
    assert s["labels.clusters_with_survivor_label_conflict"] == 0


def test_survivor_label_conflict_inside_a_cluster():
    rows = rows_of([S.obs_line(T0, "A", "created", [0, 0, 2], label="cup"), S.obs_line(T0 + S_, "B", "created", [0.1, 0, 2], label="cup")])
    m = M.compute_metrics(rows, S.summary_for([{"id": "A", "label_primary": "cup"}, {"id": "B", "label_primary": "bowl"}]), P())
    assert m["scalars"]["labels.clusters_with_survivor_label_conflict"] == 1 and m["clusters"][0]["labels"]["survivor_label_primary"] == ["bowl", "cup"]


# ───────────────────────────── scatter ─────────────────────────────

def test_scatter_splits_along_ray_from_lateral_and_regresses_on_range():
    cam = (0.0, 0.0, 0.0)
    rows = S.meta_row()
    # A at 2 m straight ahead, observations spread ALONG the ray by +-0.1 m
    for k, dz in enumerate([-0.1, 0.0, 0.1, -0.1, 0.1]):
        rows.append(S.obs_line(T0 + k * S_, "A", "created" if k == 0 else "matched", [0.0, 0.0, 2.0 + dz], cam=cam))
    # B at 4 m, spread LATERALLY by +-0.1 m (x)
    for k, dx in enumerate([-0.1, 0.0, 0.1, -0.1, 0.1]):
        rows.append(S.obs_line(T0 + k * S_, "B", "created" if k == 0 else "matched", [5.0 + dx, 0.0, 4.0], cam=(5.0, 0.0, 0.0)))
    # C at 6 m with a larger along spread (+-0.3) so |along| grows with range
    for k, dz in enumerate([-0.3, 0.0, 0.3, -0.3, 0.3]):
        rows.append(S.obs_line(T0 + k * S_, "C", "created" if k == 0 else "matched", [10.0, 0.0, 6.0 + dz], cam=(10.0, 0.0, 0.0)))
    m = M.compute_metrics(rows, S.summary_for([]), P())
    by_id = {o["id"]: o for o in m["objects"]}
    assert by_id["A"]["along_rms_m"] == pytest.approx(np.sqrt(0.04 / 5), abs=1e-6) and by_id["A"]["lateral_rms_m"] == pytest.approx(0.0, abs=1e-9)
    assert by_id["B"]["lateral_rms_m"] == pytest.approx(np.sqrt(0.04 / 5), abs=1e-6) and by_id["B"]["along_rms_m"] == pytest.approx(0.0, abs=1e-9)
    sc = m["scatter"]
    assert sc["n_tracks"] == 3 and sc["n_obs"] == 15
    assert sc["along_abs_vs_range"]["slope_per_m"] > 0 and sc["along_abs_vs_range"]["n"] == 15
    # bins follow each observation's own range: A at 1.9 / 2.0 / 2.1 m, B at 4.0 m, C at 5.7 / 6.0 / 6.3 m
    assert set(sc["by_range"]) == {"1.50-2.00", "2.00-2.50", "4.00-4.50", "5.50-6.00", "6.00-6.50"}
    assert sc["by_range"]["4.00-4.50"]["n"] == 5
    # below the observation floor nothing is reported
    m2 = M.compute_metrics(rows, S.summary_for([]), P(min_obs_scatter=6))
    assert m2["scatter"]["n_tracks"] == 0 and m2["scalars"]["scatter.along_rms_m"] is None


# ───────────────────────────── duplicate spawns ─────────────────────────────

def test_duplicate_spawns_alive_rule_and_reason_classes():
    rows = rows_of([
        S.obs_line(T0, "A", "created", [0.0, 0.0, 2.0]),                                                   # survivor
        S.view_line(T0 + S_, ["A"]),
        S.obs_line(T0 + S_, "B", "created", [0.2, 0.0, 2.0], n_nearby=1, n_gate_survivors=0),               # gated, A in view
        S.obs_line(T0 + 2 * S_, "C", "created", [2.0, 0.0, 2.0]),                                          # 2 m away: not a duplicate
        S.obs_line(T0 + 3 * S_, "E", "created", [5.0, 0.0, 2.0]),                                          # transient, last seen at +3 s
        S.obs_line(T0 + 30 * S_, "D", "created", [5.1, 0.0, 2.0], n_nearby=0),                              # E is dead by then (TTL 10 s): no duplicate
        S.obs_line(T0 + 4 * S_, "F", "created", [0.3, 0.0, 2.0], n_nearby=2, n_gate_survivors=1, max_cos=0.5),   # low_similarity (of A)
        S.obs_line(T0 + 5 * S_, "G", "created", [0.1, 0.0, 2.0], n_nearby=0),                              # not_in_index (of A)
        S.obs_line(T0 + 6 * S_, "H", "created", [0.15, 0.0, 2.0], n_nearby=2, n_gate_survivors=1, max_cos=0.95),  # other
    ])
    m = M.compute_metrics(rows, S.summary_for([{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}, {"id": "F"}, {"id": "G"}, {"id": "H"}]), P())
    dups = {d["id"]: d for d in m["duplicate_spawns"]}
    assert set(dups) == {"B", "F", "G", "H"}
    assert dups["B"]["reason"] == "gated" and dups["B"]["original_in_view"] is True and dups["B"]["dup_of"] == "A"
    assert dups["F"]["reason"] == "low_similarity" and dups["G"]["reason"] == "not_in_index" and dups["H"]["reason"] == "other"
    s = m["scalars"]
    assert s["duplicates.n"] == 4 and s["duplicates.rate_of_created"] == pytest.approx(4 / 8) and s["duplicates.original_in_view"] == 1
    assert s["duplicates.reason.gated"] == 1 and s["duplicates.reason.low_similarity"] == 1 and s["duplicates.reason.not_in_index"] == 1 and s["duplicates.reason.other"] == 1
    assert s["memory.ids_created"] == 8 and s["memory.transient"] == 1


def test_duplicate_of_a_dead_transient_when_it_is_still_within_ttl():
    rows = rows_of([
        S.obs_line(T0, "E", "created", [5.0, 0.0, 2.0]),
        S.obs_line(T0 + 5 * S_, "D", "created", [5.1, 0.0, 2.0], n_nearby=1, n_gate_survivors=0),   # E last seen 5 s ago (< TTL): alive
    ])
    m = M.compute_metrics(rows, S.summary_for([{"id": "D"}]), P())
    assert [d["id"] for d in m["duplicate_spawns"]] == ["D"] and m["duplicate_spawns"][0]["original_survivor"] is False


# ───────────────────────────── revisits ─────────────────────────────

def _revisit_rows(second_visit: str) -> list:
    rows = rows_of([S.dequeue_line(T0), S.obs_line(T0, "A", "created", [0, 0, 2])])
    for k in (1, 2):
        rows += [S.dequeue_line(T0 + k * S_), S.view_line(T0 + k * S_, ["A"]), S.obs_line(T0 + k * S_, "A", "matched", [0, 0, 2], n_nearby=1, n_gate_survivors=1, max_cos=0.95, cos_sim=0.95)]
    for k in (10, 11, 12):                                        # 8 s later: a second visit
        rows += [S.dequeue_line(T0 + k * S_), S.view_line(T0 + k * S_, ["A"])]
        if second_visit == "matched" and k == 11:
            rows.append(S.obs_line(T0 + k * S_, "A", "matched", [0, 0, 2], n_nearby=1, n_gate_survivors=1, max_cos=0.95, cos_sim=0.95))
        if second_visit == "created" and k == 11:
            rows.append(S.obs_line(T0 + k * S_, "B", "created", [0.1, 0, 2], n_nearby=1, n_gate_survivors=0))
    return rows


@pytest.mark.parametrize("second,want", [("matched", ("reidentified", 1.0)), ("created", ("duplicated", 0.0)), ("nothing", ("missed", 0.0))])
def test_revisits_outcomes_and_lower_bound(second, want):
    m = M.compute_metrics(_revisit_rows(second), S.summary_for([{"id": "A"}]), P())
    r = m["revisits"]
    assert r["clusters_with_revisit"] == 1 and r["n_revisits"] == 1 and r["n_visits"] == 2
    assert r[want[0]] == 1 and r["reid_lower_bound"] == want[1]
    assert m["clusters"][0]["visits"]["n_visits"] == 2 and m["clusters"][0]["visits"]["visit_starts_s"] == [0.0, 10.0]


def test_a_short_gap_is_one_visit():
    m = M.compute_metrics(_revisit_rows("matched"), S.summary_for([{"id": "A"}]), P(revisit_gap_s=20.0))
    assert m["revisits"]["n_visits"] == 1 and m["revisits"]["n_revisits"] == 0 and m["revisits"]["reid_lower_bound"] is None


# ───────────────────────────── worst moments / admission / memory ─────────────────────────────

def test_worst_moments_rank_by_score_and_exist_in_the_rows():
    rows = rows_of([
        S.dequeue_line(T0), S.obs_line(T0, "A", "created", [0, 0, 2]), S.obs_line(T0, "B", "created", [3, 0, 2]), S.obs_line(T0, "C", "created", [6, 0, 2]),
        S.dequeue_line(T0 + S_), S.view_line(T0 + S_, ["A", "B", "C"]),                                 # 3 misses
        S.dequeue_line(T0 + 2 * S_), S.view_line(T0 + 2 * S_, ["A", "B"]), S.obs_line(T0 + 2 * S_, "A", "matched", [0, 0, 2]),   # 1 miss
        S.dequeue_line(T0 + 3 * S_, outcome="frame_rejected", reason="dark"),
    ])
    rows += S.pose_rows(5, hz=1.0, t0_ns=T0)
    m = M.compute_metrics(rows, S.summary_for([{"id": "A"}, {"id": "B"}, {"id": "C"}]), P())
    w = m["worst_moments"]
    assert [x["t_sensor_ns"] for x in w[:2]] == [T0 + S_, T0 + 2 * S_] and w[0]["score"] == 3.0 and w[1]["score"] == 1.0
    kinds = {x["kind"] for x in w}
    assert "frame_rejected" in kinds
    stamps = {r.get("t_sensor_ns") for r in rows}
    assert all(x["t_sensor_ns"] in stamps for x in w)
    assert w[0]["t_rel_s"] == 1.0 and m["t0_sensor_ns"] == T0
    assert m["scalars"]["admission.frame_rejected"] == 1 and m["scalars"]["admission.processed"] == 3


def test_admission_equals_the_frame_outcome_histogram():
    spec = [("enqueued", "processed", "keyframe"), ("enqueued", "gate_rejected", "skip"), ("enqueued", "frame_rejected", "dark"),
            ("dropped:throttle", "", ""), ("dropped:throttle", "", ""), ("dropped:tracking_state", "", ""), ("enqueued", "processed", "parallax")]
    rows = rows_of(S.frame_flow_rows(spec))
    m = M.compute_metrics(rows, None, P())
    a = m["admission"]
    assert a["outcomes"] == dict(outcome_histogram(rows))
    assert a["receiver_lines"] == 7 and a["enqueued"] == 4 and a["throttled"] == 2 and a["tracking_dropped"] == 1
    assert a["processed"] == 2 and a["keyframes_processed"] == 1 and a["gate_rejected"] == 1 and a["frame_rejected"] == 1
    assert a["processed_gap_s"]["n"] == 1 and a["processed_gap_s"]["p50"] == pytest.approx(0.6)


def test_memory_transient_hits_and_view_bins():
    rows = rows_of([S.obs_line(T0, "A", "created", [0, 0, 2]), S.obs_line(T0, "B", "created", [3, 0, 2]), S.obs_line(T0, "C", "created", [6, 0, 2]),
                    S.obs_line(T0 + S_, "A", "matched", [0, 0, 2], matched_without_scoring=True)])
    summ = S.summary_for([{"id": "A", "hits": 7, "confirmed": True, "view_bins": 2}, {"id": "B", "hits": 1}])
    m = M.compute_metrics(rows, summ, P())
    mem = m["memory"]
    assert mem["objects"] == 2 and mem["confirmed"] == 1 and mem["ids_created"] == 3 and mem["transient"] == 1 and mem["survivors_single_hit"] == 1
    assert mem["hits_hist"] == {"1": 1, "2": 0, "3-5": 0, "6-10": 1, "11+": 0} and mem["view_bins_hist"] == {"1": 1, "2": 1}
    assert mem["matched_without_scoring"] == 1
    objs = {o["id"]: o for o in m["objects"]}
    assert objs["C"]["survivor"] is False and objs["A"]["hits"] == 7 and objs["C"]["hits"] is None


def test_empty_ledgers_do_not_crash():
    rows = rows_of(S.pose_rows(3, hz=10.0))
    m = M.compute_metrics(rows, S.summary_for([]), P())
    assert m["scalars"]["clusters.n"] == 0 and m["clusters"] == [] and m["objects"] == [] and m["worst_moments"] == []
    assert m["scalars"]["detection.reid_rate"] is None and m["scalars"]["pose.n_frames"] == 3


# ───────────────────────────── parameters ─────────────────────────────

def test_params_from_cfg_default_to_the_associators_gate():
    p = M.MetricParams.from_cfg({"assoc": {"gate_dist_base_m": 0.7, "cos_min": 0.85}, "object": {"proto_ttl_s": 4.0}})
    assert p.cluster_radius_m == 0.7 and p.cluster_radii_m == (0.35, 0.7, 1.4) and p.cos_min == 0.85 and p.proto_ttl_s == 4.0
    p2 = M.MetricParams.from_cfg({"assoc": {"gate_dist_base_m": 0.7}, "eval": {"metrics": {"cluster_radius_m": 0.3, "cluster_radii_m": [0.1, 0.9], "revisit_gap_s": 2}}})
    assert p2.cluster_radius_m == 0.3 and p2.cluster_radii_m == (0.1, 0.3, 0.9) and p2.revisit_gap_s == 2.0
    with pytest.raises(ValueError):
        M.MetricParams.from_cfg({"eval": {"metrics": {"cluster_radius_m": 0}}})
    with pytest.raises(ValueError):
        M.MetricParams.from_cfg({"eval": {"metrics": {"revisit_gap_s": -1}}})
    from rtsm.cfg import load_config
    pk = M.MetricParams.from_cfg(load_config())
    assert pk.cluster_radius_m == load_config()["assoc"]["gate_dist_base_m"]


# ───────────────────────────── the report ─────────────────────────────

def _one_run(delta: float = 0.0) -> dict:
    rows = rows_of([S.dequeue_line(T0), S.obs_line(T0, "A", "created", [0, 0, 2]), S.obs_line(T0, "B", "created", [3 + delta, 0, 2]),
                    S.dequeue_line(T0 + S_), S.view_line(T0 + S_, ["A", "B"]), S.obs_line(T0 + S_, "A", "matched", [0, 0, 2])])
    rows += S.pose_rows(4, hz=2.0, t0_ns=T0)
    return M.compute_metrics(rows, S.summary_for([{"id": "A"}, {"id": "B"}]), P())


def test_aggregate_floor_spread_and_insufficient_runs():
    agg = RP.aggregate([_one_run(), _one_run(), _one_run()], 0.5)
    assert agg["n_runs"] == 3 and agg["floor_established"] is True
    assert all(v["spread"] == 0 for v in agg["scalars"].values() if v["n_runs"])
    assert agg["scalars"]["detection.reid_rate"]["values"] == [0.5, 0.5, 0.5]
    assert agg["clusters_across_runs"]["n_in_all_runs"] == 2 and agg["clusters_across_runs"]["n_only_in_some"] == 0
    agg2 = RP.aggregate([_one_run(), _one_run(delta=3.0), _one_run()], 0.5)       # B moved 3 m in run 2: one cluster not found there
    assert agg2["clusters_across_runs"]["n_in_all_runs"] == 1 and agg2["clusters_across_runs"]["per_cluster"][1]["in_runs"] == 2
    agg3 = RP.aggregate([_one_run(), _one_run()], 0.5)
    assert agg3["floor_established"] is False
    md = RP.render_markdown(agg3, _one_run(), resolved=None, input_name="x", params=P(), repeats=None)
    assert "Floor not established: 2 run(s)" in md and "insufficient (n=2)" in md


def _write_run_dir(d: Path, rows: list, summary: dict) -> None:
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _synthetic_out_dir(tmp_path: Path, n: int = 3) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    for k in range(1, n + 1):
        rows = rows_of([S.dequeue_line(T0, is_keyframe=True), S.obs_line(T0, "A", "created", [0, 0, 2], frame_seq=1),
                        S.dequeue_line(T0 + S_, gate_shadow="skip"), S.view_line(T0 + S_, ["A"]), S.obs_line(T0 + S_, "A", "matched", [0, 0, 2], frame_seq=2)])
        rows += S.pose_rows(4, hz=2.0, t0_ns=T0)
        _write_run_dir(out / f"run_{k}", rows, S.summary_for([{"id": "A", "hits": 2, "confirmed": True}], run_index=k, wall_s=1.0 + k))
    (out / "resolved.json").write_text(json.dumps({"mode": "dense", "cadence": "representative", "input": "recordings/x", "input_kind": "replay", "clock": "sensor",
                                                   "policy": "lossless", "keyframe_rule": {"kind": "interval", "interval_s": 1.0}, "nonkf_min_interval_s": 0.2,
                                                   "gate_mode": "shadow", "config_fingerprint": "f" * 64, "git_commit": "abc", "rtsm_version": "0", "python": "3"}), encoding="utf-8")
    (out / "repeats.json").write_text(json.dumps({"runs": n, "wall_s": [1.0 + k for k in range(1, n + 1)]}), encoding="utf-8")
    return out


def test_write_report_files_content_and_byte_identical_regeneration(tmp_path):
    out = _synthetic_out_dir(tmp_path)
    mp, rp = RP.write_report(out, {"assoc": {"gate_dist_base_m": 0.5, "cos_min": 0.9}})
    assert mp == out / "metrics.json" and rp == out / "report.md"
    doc = json.loads(mp.read_text(encoding="utf-8"))
    assert doc["n_runs"] == 3 and doc["aggregate"]["floor_established"] is True and len(doc["runs"]) == 3
    assert doc["runs"][0]["run_dir"] == "run_1" and doc["runs"][0]["shadow_mode"] is True and doc["params"]["cluster_radius_m"] == 0.5
    md = rp.read_text(encoding="utf-8")
    assert "representative (dense)" in md and "every 1.0 s of sensor time" in md and "| floor (spread over runs) |" in md
    assert "masked: re-identification rate" in md and "## Worst moments" in md and "## Method notes" in md
    assert "Floor not established" not in md
    b1, b2 = mp.read_bytes(), rp.read_bytes()
    RP.write_report(out, {"assoc": {"gate_dist_base_m": 0.5, "cos_min": 0.9}})
    assert mp.read_bytes() == b1 and rp.read_bytes() == b2


def test_report_cli_and_dispatch(tmp_path, capsys, monkeypatch):
    out = _synthetic_out_dir(tmp_path, n=1)
    assert RP.main([str(out), "--set", "eval.metrics.cluster_radius_m=0.3"]) == 0
    assert "report:" in capsys.readouterr().out
    doc = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert doc["params"]["cluster_radius_m"] == 0.3 and doc["aggregate"]["floor_established"] is False
    with pytest.raises(SystemExit) as ex:
        RP.main([str(tmp_path / "empty_dir_that_does_not_exist")])
    assert ex.value.code == 2
    (tmp_path / "nothing").mkdir()
    with pytest.raises(SystemExit) as ex:
        RP.main([str(tmp_path / "nothing")])
    assert ex.value.code == 2 and "no run_<k> directory" in capsys.readouterr().err
    import rtsm.cli as cli
    monkeypatch.setattr("sys.argv", ["rtsm", "report", "--help"])
    with pytest.raises(SystemExit) as ex:
        cli.main()
    assert ex.value.code == 0 and "out_dir" in capsys.readouterr().out
