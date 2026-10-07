"""
P3 task 2 -- the eval runner's plumbing on CPU: mode resolution, the per-run
config isolation, termination (source done + queue drained), run-directory
layout, summary / repeats records, the CLI. The pipeline is a stub that pops
the queue and grows a fake memory; the real engine is gated on the GPU
(gate G3-2, 2026-09-27; record kept locally). P3 task 3 adds
the every_frame cadence and the report written after the repeats.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")

from rtsm.cfg import load_config
from rtsm.evaluation import runner as R
from rtsm.evaluation.event_log import DQ_PROCESSED, DequeueEvent
from rtsm.io.ingest_lanes import LaneConfig, make_ingest_queue

REPO = Path(__file__).resolve().parents[1]


# ───────────────────────────── stubs ─────────────────────────────

@dataclass
class FakeObject:
    id: str
    xyz_world: np.ndarray
    hits: int = 1
    confirmed: bool = False
    label_primary: str = "thing"
    view_bins: dict = field(default_factory=dict)
    stability: float = 0.5
    created_wall_utc: float = 0.0
    created_mono: float = 0.0
    last_seen_mono: float = 0.0
    image_crops: list = field(default_factory=list)


class FakeWM:
    def __init__(self) -> None:
        self.objs = []
        self.poses = 0
        self.flushed = 0

    def iter_objects(self):
        return list(self.objs)

    def stats(self):
        return {"objects": len(self.objs), "confirmed": sum(o.confirmed for o in self.objs), "robot_pose": {"writes_accepted": self.poses}}

    def update_robot_pose(self, *a, **k):
        self.poses += 1

    def collect_ready_for_upsert(self, force_all=False):
        return [{"id": o.id} for o in self.objs] if force_all else []


class FakeVectors:
    def __init__(self):
        self.upserted = 0
        self.closed = False

    def upsert_batch(self, rows):
        self.upserted += len(rows)

    def close(self):
        self.closed = True


class FakeAnalytics:
    def __init__(self):
        self.started = self.stopped = False
        self.latency = None
        self.seg = None

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakePipeline:
    """Pops one packet per step; every processed frame becomes one object
    (deterministic: the fingerprint depends only on the input)."""

    def __init__(self, q, wm, event_log, gate_mode):
        self.q, self.wm, self.log, self.gate_mode = q, wm, event_log, gate_mode
        self.steps = 0

    def run_one_step(self):
        self.steps += 1
        pkt = self.q.get(timeout=0.0)
        if pkt is None:
            time.sleep(0.002)
            return
        i = len(self.wm.objs)
        self.wm.objs.append(FakeObject(id=f"o{i}", xyz_world=np.array([i * 0.1, 0.0, 1.0], dtype=np.float32), hits=1 + (i % 3), confirmed=(i % 2 == 0)))
        shadow = "ttl" if (self.gate_mode == "shadow" and not pkt.is_keyframe) else None
        self.log.write(DequeueEvent(timestamp=time.monotonic(), frame_seq=pkt.time.seq, t_sensor_ns=pkt.time.t_sensor_ns,
                                    is_keyframe=bool(pkt.is_keyframe), queue_wait_s=0.0, queue_depth=self.q.qsize(),
                                    outcome=DQ_PROCESSED, reason="keyframe" if pkt.is_keyframe else "ok", gate_shadow=shadow))


@dataclass
class FakeRuntime:
    ingest_q: object
    wm: FakeWM
    vectors: FakeVectors
    analytics: FakeAnalytics
    pipeline: FakePipeline
    event_log: object


def fake_runtime_factory(cfg, models, *, clock, lane_cfg, event_log, up_axis_default):
    q = make_ingest_queue(lane_cfg)
    wm = FakeWM()
    return FakeRuntime(ingest_q=q, wm=wm, vectors=FakeVectors(), analytics=FakeAnalytics(),
                       pipeline=FakePipeline(q, wm, event_log, (cfg.get("eval") or {}).get("gate_mode")), event_log=event_log)


@pytest.fixture
def bag(tmp_path):
    from test_bag_reader import simple_bag_messages, write_bag
    return write_bag(tmp_path / "bag", simple_bag_messages(12, seq_topic=True))


@pytest.fixture
def cfg():
    return load_config()


# ───────────────────────────── resolution + config isolation ─────────────────────────────

def test_resolve_both_modes(cfg, bag):
    r = R.resolve_eval(cfg, R.EvalOptions(input=str(bag)))
    assert (r.mode, r.input_kind, r.clock, r.policy, r.gate_mode) == ("as_deployed", "bag", "sensor", "lossless", "enforce")
    assert r.keyframe_rule["kind"] == "every_n" and r.keyframe_rule["n"] == cfg["ingest"]["keyframe_every_n"]
    assert r.nonkf_min_interval_s == cfg["ingest"]["nonkf_min_interval_s"] and r.up_axis_default == "z"   # a synthetic bag is not a Lens one
    assert len(r.config_fingerprint) == 64 and r.ledgers is True
    d = R.resolve_eval(cfg, R.EvalOptions(input=str(bag), mode="dense"))
    assert d.gate_mode == "shadow" and d.keyframe_rule == {"kind": "interval", "interval_s": 1.0} and d.nonkf_min_interval_s == pytest.approx(0.2)
    cfg2 = json.loads(json.dumps(cfg)); cfg2["eval"]["gate_mode"] = "enforce"; cfg2["eval"]["process_rate_hz"] = 10
    d2 = R.resolve_eval(cfg2, R.EvalOptions(input=str(bag), mode="dense"))
    assert d2.gate_mode == "enforce" and d2.nonkf_min_interval_s == pytest.approx(0.1)
    with pytest.raises(ValueError):
        R.resolve_eval(cfg, R.EvalOptions(input=str(bag), mode="sparse"))
    assert R.input_kind(REPO / "recordings" / "demo_clip") == "replay" and R.input_kind(bag) == "bag"
    # every_frame (task 3) = dense without the throttle; the cadence is recorded
    e = R.resolve_eval(cfg, R.EvalOptions(input=str(bag), mode="every_frame"))
    assert (e.cadence, e.gate_mode, e.nonkf_min_interval_s, e.keyframe_rule) == ("exhaustive", "shadow", 0.0, {"kind": "interval", "interval_s": 1.0})
    assert r.cadence == "deployed" and d.cadence == "representative"
    cfg3 = json.loads(json.dumps(cfg)); cfg3["eval"]["keyframe_interval_s"] = 0
    for mode in ("dense", "every_frame"):
        with pytest.raises(ValueError, match="every_frame"):
            R.resolve_eval(cfg3, R.EvalOptions(input=str(bag), mode=mode))


def test_configure_run_isolates_paths_and_never_touches_the_input(cfg, bag, tmp_path):
    before = json.dumps(cfg, sort_keys=True)
    r = R.resolve_eval(cfg, R.EvalOptions(input=str(bag), mode="dense"))
    c = R.configure_run(cfg, r, tmp_path / "run_1")
    assert json.dumps(cfg, sort_keys=True) == before
    assert c["vectors"]["faiss"]["index_path"] == str(tmp_path / "run_1" / "faiss" / "index.flatip")
    assert c["diagnostics"]["enabled"] is True and c["diagnostics"]["ledgers"] is True and c["diagnostics"]["event_log_path"].endswith("events.jsonl")
    assert c["eval"]["gate_mode"] == "shadow" and c["visualization"]["enable"] is False and c["io"]["receiver"] == "bag"


def test_fingerprint_is_the_gate_scripts_definition():
    objs = [{"label_primary": "cup", "xyz_world": [0.12345, 1.0, 2.0], "hits": 3, "confirmed": True},
            {"label_primary": "cup", "xyz_world": [0.12345, 1.0, 2.0], "hits": 3, "confirmed": True},
            {"label_primary": "box", "xyz_world": [1.0, 1.0, 1.0], "hits": 1, "confirmed": False}]
    from collections import Counter
    ms = Counter((o["label_primary"], tuple(round(float(v), 3) for v in o["xyz_world"]), int(o["hits"]), bool(o["confirmed"])) for o in objs)
    want = hashlib.sha256(json.dumps(sorted(map(list, ms.elements())), default=str).encode()).hexdigest()[:16]
    assert R.fingerprint(objs) == want
    assert R.fingerprint(list(reversed(objs))) == want                           # order-free
    assert R.fingerprint(objs[:1]) != want


# ───────────────────────────── the run ─────────────────────────────

def test_run_eval_terminates_writes_records_and_isolates_repeats(cfg, bag, tmp_path):
    packaged = REPO / "rtsm" / "cfg" / "rtsm.yaml"
    yaml_before = packaged.read_bytes()
    opts = R.EvalOptions(input=str(bag), repeats=2, out=str(tmp_path / "out"))
    result = R.run_eval(cfg, opts, models=object(), runtime_factory=fake_runtime_factory)
    assert packaged.read_bytes() == yaml_before
    assert (tmp_path / "out" / "resolved.json").is_file() and (tmp_path / "out" / "bag_probe.json").is_file()
    assert len(result.runs) == 2
    # the report (task 3) is written after the repeats, over the run directories
    assert result.metrics_path == tmp_path / "out" / "metrics.json" and result.report_path == tmp_path / "out" / "report.md"
    doc = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert doc["n_runs"] == 2 and doc["aggregate"]["floor_established"] is False and doc["runs"][0]["scalars"]["admission.processed"] == 4
    assert doc["runs"][0]["scalars"]["pose.n_frames"] == 12 and doc["runs"][0]["clusters"] == []      # the stub creates no obs lines
    assert "Floor not established: 2 run(s)" in result.report_path.read_text(encoding="utf-8")
    for k, s in enumerate(result.runs, 1):
        rd = tmp_path / "out" / f"run_{k}"
        assert s["run_dir"] == str(rd) and (rd / "summary.json").is_file() and (rd / "events.jsonl").is_file()
        assert s["aborted"] is None and s["mode"] == "as_deployed" and s["resolved"]["gate_mode"] == "enforce"
        # keyframe at frame 1 (count rule), throttle 0.5 s on the 100 ms synthetic cadence: non-KFs at 0.1, 0.6, 1.1 s -> 4 admitted, 8 throttled
        assert s["frames"]["receiver"]["enqueued"] == 4 and s["frames"]["receiver"]["dropped:throttle"] == 8
        assert s["frames"]["processed"] == 4 and s["frames"]["kinds"]["dequeue"] == 4 and s["frames"]["kinds"]["pose"] == 12
        assert s["memory"]["objects_count"] == 4 and s["memory"]["confirmed_count"] == 2 and s["flushed_to_vectors"] == 4
        assert s["source"]["yielded"] == 12 and s["frames"]["gate_shadow"] == {}
        assert json.loads((rd / "summary.json").read_text(encoding="utf-8"))["memory"]["fingerprint"] == s["memory"]["fingerprint"]
    rep = json.loads((tmp_path / "out" / "repeats.json").read_text(encoding="utf-8"))
    assert rep["runs"] == 2 and rep["identical_fingerprints"] is True and rep["objects"] == [4, 4]
    assert result.repeats["fingerprints"][0] == result.repeats["fingerprints"][1]
    # the two runs' vector stores are different files
    assert result.runs[0]["resolved"] == result.runs[1]["resolved"]
    c1 = R.configure_run(cfg, result.resolved, tmp_path / "out" / "run_1")["vectors"]["faiss"]["index_path"]
    c2 = R.configure_run(cfg, result.resolved, tmp_path / "out" / "run_2")["vectors"]["faiss"]["index_path"]
    assert c1 != c2


def test_dense_mode_shadow_gate_interval_keyframes_and_max_frames(cfg, bag, tmp_path):
    opts = R.EvalOptions(input=str(bag), mode="dense", repeats=1, out=str(tmp_path / "out"), max_frames=8)
    result = R.run_eval(cfg, opts, models=object(), runtime_factory=fake_runtime_factory)
    s = result.runs[0]
    # interval 1.0 s on a 100 ms cadence over 8 frames -> 1 keyframe (t=0); throttle 0.2 s -> non-KFs at 0.1, 0.3, 0.5, 0.7 s
    assert s["frames"]["receiver"]["enqueued"] == 5 and s["frames"]["processed"] == 5
    assert s["frames"]["gate_shadow"] == {"ttl": 4}                                   # the stub shadows every non-keyframe
    assert s["resolved"]["keyframe_rule"] == {"kind": "interval", "interval_s": 1.0} and s["source"]["yielded"] == 8
    rows = [json.loads(l) for l in (tmp_path / "out" / "run_1" / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["eval_mode"] == "dense" and rows[0]["gate_mode"] == "shadow" and rows[0]["keyframe_rule"]["kind"] == "interval"
    assert sum(1 for r in rows if r.get("kind") == "dequeue" and r.get("gate_shadow")) == 4


def test_every_frame_mode_admits_every_frame_and_no_report(cfg, bag, tmp_path):
    opts = R.EvalOptions(input=str(bag), mode="every_frame", repeats=1, out=str(tmp_path / "out"), report=False)
    result = R.run_eval(cfg, opts, models=object(), runtime_factory=fake_runtime_factory)
    s = result.runs[0]
    # 12 frames at 100 ms: interval keyframes at 0 and 1.0 s, no throttle -> every frame admitted and processed
    assert s["frames"]["receiver"] == {"enqueued": 12} and s["frames"]["processed"] == 12
    assert s["resolved"]["cadence"] == "exhaustive" and s["resolved"]["nonkf_min_interval_s"] == 0.0 and s["resolved"]["gate_mode"] == "shadow"
    assert s["frames"]["gate_shadow"] == {"ttl": 10}                                  # the stub shadows every non-keyframe
    rows = [json.loads(l) for l in (tmp_path / "out" / "run_1" / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["eval_mode"] == "every_frame" and rows[0]["nonkf_min_interval_s"] == 0.0
    assert sum(1 for r in rows if r.get("kind") == "receiver" and r.get("is_keyframe")) == 2
    assert result.report_path is None and not (tmp_path / "out" / "metrics.json").exists()
    # rtsm report writes it later from the same directory
    from rtsm.evaluation.report import write_report
    mp, rp = write_report(tmp_path / "out", cfg)
    assert mp.is_file() and "exhaustive (every_frame)" in rp.read_text(encoding="utf-8")


def test_max_wall_abort_is_recorded(cfg, bag, tmp_path):
    class SlowPipeline(FakePipeline):
        def run_one_step(self):
            time.sleep(0.05)
            super().run_one_step()

    def slow_factory(cfg, models, *, clock, lane_cfg, event_log, up_axis_default):
        rt = fake_runtime_factory(cfg, models, clock=clock, lane_cfg=lane_cfg, event_log=event_log, up_axis_default=up_axis_default)
        rt.pipeline = SlowPipeline(rt.ingest_q, rt.wm, event_log, None)
        return rt

    opts = R.EvalOptions(input=str(bag), out=str(tmp_path / "out"), max_wall_s=0.05)
    result = R.run_eval(cfg, opts, models=object(), runtime_factory=slow_factory)
    assert result.runs[0]["aborted"] == "max_wall_s" and result.repeats["aborted"] == ["max_wall_s"]


def test_refused_bag_stops_before_models(cfg, tmp_path):
    from test_bag_reader import simple_bag_messages, write_bag
    nod = write_bag(tmp_path / "nodepth", [m for m in simple_bag_messages(3) if "depth" not in m[0]])
    calls = []
    with pytest.raises(ValueError, match="no_depth_topic"):
        R.run_eval(cfg, R.EvalOptions(input=str(nod), out=str(tmp_path / "out")), load_models=lambda c: calls.append(1))
    assert calls == []


# ───────────────────────────── CLI ─────────────────────────────

def test_cli_main_and_errors(cfg, bag, tmp_path, monkeypatch, capsys):
    seen = {}

    def fake_run_eval(cfg, opts):
        seen["opts"] = opts
        r = R.resolve_eval(cfg, opts)
        res = R.EvalResult(out_dir=tmp_path / "o", resolved=r)
        res.runs.append({"memory": {"fingerprint": "abc", "objects_count": 1, "confirmed_count": 0}, "frames": {"processed": 1},
                         "wall_s": 0.1, "aborted": None, "run_dir": "x"})
        return res

    monkeypatch.setattr(R, "run_eval", fake_run_eval)
    assert R.main([str(bag), "--mode", "dense", "--repeats", "2", "--max-frames", "5", "--set", "eval.process_rate_hz=4"]) == 0
    out = capsys.readouterr().out
    assert seen["opts"].mode == "dense" and seen["opts"].repeats == 2 and seen["opts"].max_frames == 5
    assert "keyframes={'kind': 'interval'" in out and '"identical_fingerprints": true' in out and "(representative)" in out
    assert seen["opts"].report is True
    assert R.main([str(bag), "--no-report"]) == 0 and seen["opts"].report is False
    with pytest.raises(SystemExit) as ex:
        R.main([str(tmp_path / "missing.bag")])
    assert ex.value.code == 2 and "input not found" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ex:
        R.main([str(bag), "--repeats", "0"])
    assert ex.value.code == 2


def test_cli_dispatch_reaches_the_runner(monkeypatch, capsys):
    import rtsm.cli as cli
    monkeypatch.setattr("sys.argv", ["rtsm", "eval", "--help"])
    with pytest.raises(SystemExit) as ex:
        cli.main()
    assert ex.value.code == 0 and "--mode" in capsys.readouterr().out


def test_save_crops_writes_snapshots_and_index(cfg, bag, tmp_path):
    """--save-crops writes run_N/crops/<id>/<k>.jpg + index.json from the objects' image_crops;
    the default run writes no crops directory; resolved carries the dirty flag."""
    jpeg_a, jpeg_b = b"\xff\xd8\xff\xe0" + b"a" * 16, b"\xff\xd8\xff\xe0" + b"b" * 16

    def crops_factory(cfg_, models, *, clock, lane_cfg, event_log, up_axis_default):
        rt = fake_runtime_factory(cfg_, models, clock=clock, lane_cfg=lane_cfg, event_log=event_log, up_axis_default=up_axis_default)
        orig = rt.wm.iter_objects

        def with_crops():
            objs = orig()
            for i, o in enumerate(objs):
                o.image_crops = [jpeg_a, jpeg_b] if i % 2 == 0 else []     # half the objects carry two snapshots
            return objs

        rt.wm.iter_objects = with_crops
        return rt

    opts = R.EvalOptions(input=str(bag), repeats=1, out=str(tmp_path / "out"), save_crops=True, report=False)
    result = R.run_eval(cfg, opts, models=object(), runtime_factory=crops_factory)
    rd = tmp_path / "out" / "run_1"
    idx = json.loads((rd / "crops" / "index.json").read_text(encoding="utf-8"))
    assert idx["schema"] == 1 and idx["n_objects"] == 2 and idx["n_files"] == 4      # 4 objects, 2 with crops, 2 crops each
    for oid, paths in idx["objects"].items():
        assert paths == [f"{oid}/000.jpg", f"{oid}/001.jpg"]
        assert (rd / "crops" / paths[0]).read_bytes() == jpeg_a and (rd / "crops" / paths[1]).read_bytes() == jpeg_b
    s = result.runs[0]
    assert s["crops"] == {"path": "crops", "objects": 2, "files": 4}
    assert s["resolved"]["git_dirty"] in (True, False, None) and "tree_digest" in s["resolved"]
    # default: no crops directory, summary says None
    opts2 = R.EvalOptions(input=str(bag), repeats=1, out=str(tmp_path / "out2"), report=False)
    result2 = R.run_eval(cfg, opts2, models=object(), runtime_factory=crops_factory)
    assert not (tmp_path / "out2" / "run_1" / "crops").exists() and result2.runs[0]["crops"] is None


def test_git_state_outside_a_checkout(tmp_path):
    assert R._git_state(tmp_path) == (None, None)


def test_model_files_and_hub_ids_in_provenance(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"weights" * 100)
    cfg = {"segmentation": {"backend": "dual",
                            "fastsam": {"model_path": "a.pt"},                      # relative to the working directory
                            "yoloe": {"model_path": str(tmp_path / "missing.pt")},  # absent: a library would auto-download
                            "sam2": {"model_id": "facebook/sam2.1-hiera-small"},
                            "grounded_sam2": {"gdino_model_id": "IDEA-Research/grounding-dino-tiny", "sam2_model_id": None}},
           "clip": {"pretrained": "webli"}}
    files = R._model_files(cfg, root=tmp_path)
    assert set(files) == {"segmentation.fastsam.model_path", "segmentation.yoloe.model_path"}
    a = files["segmentation.fastsam.model_path"]
    assert a["exists"] is True and a["bytes"] == 700 and a["sha256"] == hashlib.sha256(b"weights" * 100).hexdigest() and a["path"] == "a.pt"
    assert files["segmentation.yoloe.model_path"] == {"path": str(tmp_path / "missing.pt"), "exists": False}
    assert R._hf_models(cfg) == {"segmentation.sam2.model_id": "facebook/sam2.1-hiera-small",
                                 "segmentation.grounded_sam2.gdino_model_id": "IDEA-Research/grounding-dino-tiny"}
