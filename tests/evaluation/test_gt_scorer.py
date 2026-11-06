"""The ground-truth scorer on a synthetic sidecar and ledger (no GPU, no models)."""
import json
import math

import pytest

from rtsm.evaluation import gt_scorer as G

S = 1_000_000_000
T0 = 1_700_000_000 * S


def _box(center, extents, q=(0.0, 0.0, 0.0, 1.0)):
    return {"center": list(center), "extents": list(extents), "q_xyzw": list(q)}


OBJECTS = [
    {"id": 0, "prim": "/Root/Table", "class": "table", "bbox3d_world": _box((0, 0, 0.4), (2.0, 1.0, 0.8))},
    {"id": 1, "prim": "/Root/Book", "class": "book", "bbox3d_world": _box((0.5, 0.2, 0.83), (0.3, 0.2, 0.06))},     # on the table, inside its box
    {"id": 2, "prim": "/Root/Chair", "class": "chair", "bbox3d_world": _box((3.0, 0.0, 0.5), (0.5, 0.5, 1.0))},
    {"id": 3, "prim": "/Root/Lamp", "class": "lamp", "bbox3d_world": _box((-3.0, 2.0, 2.5), (0.4, 0.4, 0.3))},   # never big enough in view
    {"id": 4, "prim": "/Root/Looks/Nothing", "class": "table", "bbox3d_world": None},                              # labelled, no box
]


def _frame_line(i, objs):
    return {"frame": i, "ts_ns": T0 + i * S, "T_wc_true": {"t": [0, -3, 1.4], "q_xyzw": [0, 0, 0, 1]},
            "T_wc_published": {"t": [0, -3, 1.4], "q_xyzw": [0, 0, 0, 1]}, "stamp_offset_s": 0.0, "objects": objs}


def _obj(gid, pixels, bbox2d):
    return {"id": gid, "pixels": pixels, "bbox2d": bbox2d, "occlusion": None, "bbox3d_world": None}


@pytest.fixture
def sidecar(tmp_path):
    d = tmp_path / "gt"
    d.mkdir()
    (d / "objects.json").write_text(json.dumps(OBJECTS))
    lines = []
    # frames 0..3: table, book, chair visible; lamp tiny. gap of 8 s. frames 11..13: table, chair visible again (a revisit); book gone.
    for i in range(4):
        lines.append(_frame_line(i, [_obj(0, 50000, [100, 200, 500, 400]), _obj(1, 3000, [300, 250, 360, 290]),
                                     _obj(2, 20000, [520, 150, 630, 420]), _obj(3, 200, [10, 10, 20, 20])]))
    for i in range(11, 14):
        lines.append(_frame_line(i, [_obj(0, 50000, [100, 200, 500, 400]), _obj(2, 20000, [520, 150, 630, 420])]))
    (d / "gt.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    (d / "meta.json").write_text(json.dumps({"schema": 1, "scene": "synthetic", "session_id": "s"}))
    return G.load_sidecar(d)


def _obs(i, oid, outcome, p):
    return {"kind": "obs", "t_sensor_ns": T0 + i * S, "frame_seq": i, "object_id": oid, "outcome": outcome, "p_world": list(p)}


def _frame(i):
    return {"kind": "frame", "t_sensor_ns": T0 + i * S, "frame_seq": i}


def _det(i, boxes):
    return {"kind": "det", "t_sensor_ns": T0 + i * S, "n": len(boxes), "detections": [{"idx": k, "label": "x", "score": 0.5, "bbox_rgb": b} for k, b in enumerate(boxes)]}


def _mem(oid, p, label, confirmed=True, hits=3):
    return {"id": oid, "xyz_world": list(p), "label_primary": label, "confirmed": confirmed, "hits": hits}


def test_box_distance_inside_outside_rotated():
    t = G.TruthObject.from_record(OBJECTS[0])
    assert G.box_distance((0.3, 0.2, 0.5), t) == 0.0
    assert math.isclose(G.box_distance((1.1, 0.0, 0.4), t), 0.1, abs_tol=1e-9)           # 0.1 beyond +x face
    assert math.isclose(G.box_distance((1.3, 0.9, 0.4), t), math.hypot(0.3, 0.4), abs_tol=1e-9)
    # a box rotated 90 deg about z: its 2 m side now runs along y
    rot = G.TruthObject.from_record({"id": 9, "prim": "/r", "class": "c", "bbox3d_world": _box((0, 0, 0), (2.0, 1.0, 1.0), (0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)))})
    assert G.box_distance((0.0, 0.9, 0.0), rot) == 0.0 and math.isclose(G.box_distance((0.9, 0.0, 0.0), rot), 0.4, abs_tol=1e-9)


def test_label_agreement_rules():
    assert G.label_agreement("table", "Table", G.DEFAULT_SYNONYMS) == "exact"
    assert G.label_agreement("book", "books", G.DEFAULT_SYNONYMS) == "exact"
    assert G.label_agreement("table", "desk", G.DEFAULT_SYNONYMS) == "synonym"
    assert G.label_agreement("chair", "office chairs", G.DEFAULT_SYNONYMS) == "synonym"
    assert G.label_agreement("chair", "bed", G.DEFAULT_SYNONYMS) == "none"
    assert G.label_agreement("chair", "", G.DEFAULT_SYNONYMS) == "none"


def test_score_run_end_to_end(sidecar):
    # RTSM: A on the chair (confirmed, "armchair"), B a second object on the chair created in the revisit (a duplicate),
    # C on the book (inside the table's box too: the smaller box must win), D a confirmed phantom in free space,
    # E a transient (no memory entry) ignored. The table is observable but RTSM never built an object on it.
    rows = []
    for i in range(4):
        rows.append(_frame(i))
        rows.append(_obs(i, "A", "created" if i == 0 else "matched", (3.0, 0.05, 0.6)))
        rows.append(_obs(i, "C", "created" if i == 0 else "matched", (0.5, 0.2, 0.86)))
        rows.append(_det(i, [[522, 152, 628, 418], [0, 0, 50, 50]]))                      # a chair hit and a stray box
    for i in range(11, 14):
        rows.append(_frame(i))
        rows.append(_obs(i, "B", "created" if i == 11 else "matched", (2.9, -0.05, 0.5)))
        rows.append(_det(i, [[300, 250, 360, 290]]))                                       # nothing big in view matches this
    rows.append(_obs(2, "D", "created", (0.0, -2.0, 1.0)))
    rows.append(_frame(20))                                                                 # a processed frame the sidecar lacks
    summary = {"memory": {"objects": [_mem("A", (3.0, 0.05, 0.6), "armchair"), _mem("B", (2.9, -0.05, 0.5), "chair", confirmed=False, hits=1),
                                      _mem("C", (0.5, 0.2, 0.86), "notebook"), _mem("D", (0.0, -2.0, 1.0), "plant")]}}
    out = G.score_run(rows, summary, sidecar, G.GtParams(revisit_gap_s=5.0))
    s = out["summary"]
    assert s["frames"] == {"n_processed": 8, "n_processed_in_sidecar": 7, "join_frac": 0.875}
    assert s["objects"]["n"] == 4 and s["objects"]["n_matched"] == 3 and s["objects"]["n_phantom"] == 1 and s["objects"]["n_phantom_confirmed"] == 1
    assert s["objects"]["precision"] == 0.75 and s["objects"]["precision_confirmed"] == pytest.approx(2 / 3)
    by_id = {m["id"]: m for m in out["rtsm_objects"]}
    assert by_id["C"]["gt_class"] == "book" and by_id["C"]["n_candidates"] == 2          # inside both boxes, the book wins
    assert by_id["A"]["gt_class"] == "chair" and by_id["B"]["gt_class"] == "chair" and by_id["D"]["gt_id"] is None
    # truth: table, book, chair observable (lamp too small, the unboxed object cannot be observable); chair and book recalled
    assert s["truth"]["n_labelled"] == 5 and s["truth"]["n_boxed"] == 4 and s["truth"]["n_observable"] == 3
    assert s["truth"]["n_recalled"] == 2 and s["truth"]["recall"] == pytest.approx(2 / 3)
    assert s["truth"]["n_recalled_confirmed"] == 2 and s["truth"]["n_duplicated"] == 1 and s["truth"]["duplicates_extra"] == 1
    assert s["position"]["d_box_median_m"] == 0.0 and s["position"]["n"] == 3
    assert s["labels"] == {"n_pairs": 3, "exact": 1, "exact_frac": pytest.approx(1 / 3), "synonym": 2, "synonym_frac": pytest.approx(2 / 3), "agree_frac": 1.0}
    # revisits against truth: the chair's second visit has A matched? no (A has no obs in 11..13) but B was created there while A existed -> duplicated;
    # the table's second visit -> no RTSM object at all -> not counted (not recalled); the book has one visit only.
    assert s["revisits"] == {"n_revisits": 1, "reidentified": 0, "duplicated": 1, "missed": 0, "first_seen_late": 0, "reid_frac": 0.0}
    # detections: frames 0..3 have 3 big truths each (table, book, chair) and 2 detections, one on the chair -> 4 tp of 12 truths, 8 dets;
    # frames 11..13 have 2 truths and 1 detection on the (absent) book box -> 0 tp of 6 truths, 3 dets
    assert s["detection"] == {"frames": 7, "n_truth": 18, "n_det": 11, "tp": 4, "recall": pytest.approx(4 / 18), "precision": pytest.approx(4 / 11)}
    assert out["detection_by_class"]["chair"] == {"n_truth": 7, "n_hit": 4, "recall": pytest.approx(4 / 7)}
    assert out["confusion"] == []


def test_reidentified_when_the_same_object_matches_in_the_revisit(sidecar):
    rows = [_frame(i) for i in list(range(4)) + list(range(11, 14))]
    rows += [_obs(i, "A", "created" if i == 0 else "matched", (3.0, 0.0, 0.6)) for i in list(range(4)) + [11, 12]]
    summary = {"memory": {"objects": [_mem("A", (3.0, 0.0, 0.6), "chair")]}}
    s = G.score_run(rows, summary, sidecar)["summary"]
    assert s["revisits"]["reidentified"] == 1 and s["revisits"]["n_revisits"] == 1 and s["revisits"]["reid_frac"] == 1.0
    assert s["labels"]["exact"] == 1


def test_scalars_flatten_and_aggregate(sidecar):
    from rtsm.evaluation.metrics import _flatten
    from rtsm.evaluation.report import aggregate
    rows = [_frame(0), _frame(1), _obs(0, "A", "created", (3.0, 0.0, 0.6)), _obs(1, "A", "matched", (3.0, 0.0, 0.6))]
    summary = {"memory": {"objects": [_mem("A", (3.0, 0.0, 0.6), "chair")]}}
    out = G.score_run(rows, summary, sidecar)
    flat = _flatten({"gt": out["summary"]})
    assert flat["gt.objects.precision"] == 1.0 and flat["gt.truth.recall"] == pytest.approx(1 / 3) and "gt.detection.recall" in flat
    agg = aggregate([{"scalars": flat, "clusters": []}, {"scalars": flat, "clusters": []}], 0.5)
    assert agg["scalars"]["gt.objects.precision"]["spread"] == 0.0 and agg["scalars"]["gt.objects.precision"]["n_runs"] == 2


def test_rejects_other_schemas(tmp_path):
    d = tmp_path / "gt"
    d.mkdir()
    (d / "objects.json").write_text("[]")
    (d / "gt.jsonl").write_text("")
    (d / "meta.json").write_text(json.dumps({"schema": 2}))
    with pytest.raises(ValueError):
        G.load_sidecar(d)
    with pytest.raises(FileNotFoundError):
        G.load_sidecar(tmp_path / "nothing")
