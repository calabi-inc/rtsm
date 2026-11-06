"""`rtsm report` / `rtsm eval` with a ground-truth sidecar: the gt block lands in metrics.json
with floors, the report gains its section, the flags parse."""
import json

import synth_ledger as S

from rtsm.evaluation.metrics import MetricParams
from rtsm.evaluation.report import build_parser as report_parser
from rtsm.evaluation.report import write_report
from rtsm.evaluation.runner import build_parser as eval_parser

T0 = 1_000_000_000
N = 10


def _run_dir(root, k, with_phantom):
    d = root / f"run_{k}"
    d.mkdir()
    stamps = [T0 + i * 100_000_000 for i in range(N)]
    rows = S.meta_row() + S.pose_rows(N, hz=10, t0_ns=T0)
    for i, ts in enumerate(stamps):
        rows.append(S.dequeue_line(ts, frame_seq=i, is_keyframe=(i == 0)))
        rows.append(S.obs_line(ts, "A", "created" if i == 0 else "matched", (3.0, 0.0, 0.6), label="chair", frame_seq=i))
        if with_phantom and i == 3:
            rows.append(S.obs_line(ts, "P", "created", (0.0, 5.0, 1.0), label="plant", frame_seq=i))
    objs = [{"id": "A", "xyz_world": [3.0, 0.0, 0.6], "hits": N, "confirmed": True, "label_primary": "chair"}]
    if with_phantom:
        objs.append({"id": "P", "xyz_world": [0.0, 5.0, 1.0], "hits": 1, "confirmed": False, "label_primary": "plant"})
    (d / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    (d / "summary.json").write_text(json.dumps(S.summary_for(objs, run_index=k)), encoding="utf-8")
    return stamps


def _sidecar(root, stamps):
    g = root / "gt"
    g.mkdir()
    (g / "objects.json").write_text(json.dumps([
        {"id": 0, "prim": "/Root/Chair", "class": "chair", "bbox3d_world": {"center": [3.0, 0.0, 0.5], "extents": [0.5, 0.5, 1.0], "q_xyzw": [0, 0, 0, 1]}},
        {"id": 1, "prim": "/Root/Table", "class": "table", "bbox3d_world": {"center": [0.0, 0.0, 0.4], "extents": [1.5, 0.8, 0.8], "q_xyzw": [0, 0, 0, 1]}},
    ]))
    lines = [{"frame": i, "ts_ns": ts, "T_wc_true": {"t": [0, 0, 0], "q_xyzw": [0, 0, 0, 1]}, "T_wc_published": {"t": [0, 0, 0], "q_xyzw": [0, 0, 0, 1]},
              "stamp_offset_s": 0.0, "objects": [{"id": 0, "pixels": 6000, "bbox2d": [1, 1, 5, 5], "occlusion": None, "bbox3d_world": None},
                                                 {"id": 1, "pixels": 9000, "bbox2d": [6, 2, 12, 8], "occlusion": None, "bbox3d_world": None}]}
             for i, ts in enumerate(stamps)]
    (g / "gt.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    (g / "meta.json").write_text(json.dumps({"schema": 1, "scene": "synthetic", "session_id": "s"}), encoding="utf-8")
    return g


def test_write_report_with_gt(tmp_path):
    out = tmp_path / "eval"
    out.mkdir()
    stamps = _run_dir(out, 1, with_phantom=False)
    _run_dir(out, 2, with_phantom=True)
    gt = _sidecar(tmp_path, stamps)
    (out / "resolved.json").write_text(json.dumps({"input": "synthetic.bag", "gt_dir": str(gt)}), encoding="utf-8")
    metrics_path, report_path = write_report(out, params=MetricParams())          # the sidecar comes from resolved.json
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert m["gt_dir"] == str(gt) and m["runs"][0]["gt"]["summary"]["objects"]["precision"] == 1.0
    assert m["runs"][1]["gt"]["summary"]["objects"]["precision"] == 0.5 and m["runs"][1]["gt"]["summary"]["objects"]["n_phantom"] == 1
    agg = m["aggregate"]["scalars"]
    assert agg["gt.objects.precision"]["values"] == [1.0, 0.5] and agg["gt.objects.precision"]["spread"] == 0.5
    assert agg["gt.truth.recall"]["value"] == 0.5 and agg["gt.truth.n_observable"]["value"] == 2        # the table is observable, never built
    assert agg["gt.labels.exact_frac"]["value"] == 1.0 and agg["gt.frames.join_frac"]["value"] == 1.0
    doc = report_path.read_text(encoding="utf-8")
    assert "## Ground truth (simulator sidecar)" in doc and "recall: observable true objects with an RTSM object" in doc
    # an explicit --gt wins over resolved.json, and a wrong one is refused
    write_report(out, params=MetricParams(), gt_dir=str(gt))
    import pytest
    with pytest.raises(FileNotFoundError):
        write_report(out, params=MetricParams(), gt_dir=str(tmp_path / "missing"))


def test_write_report_without_gt_is_unchanged(tmp_path):
    out = tmp_path / "eval"
    out.mkdir()
    _run_dir(out, 1, with_phantom=False)
    metrics_path, report_path = write_report(out, params=MetricParams())
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert m["gt_dir"] is None and "gt" not in m["runs"][0] and not any(k.startswith("gt.") for k in m["aggregate"]["scalars"])
    assert "## Ground truth" not in report_path.read_text(encoding="utf-8")


def test_flags_parse():
    a = eval_parser().parse_args(["some.bag", "--gt", "truth_dir"])
    assert a.gt == "truth_dir"
    r = report_parser().parse_args(["out_dir", "--gt", "truth_dir", "--gt-margin", "0.25", "--gt-min-pixels", "500"])
    assert r.gt == "truth_dir" and r.gt_margin == 0.25 and r.gt_min_pixels == 500
