"""Memory events (ledger kind mem, B2): the working memory hands one record per proto expiry, index eviction and pose
correction to its event sink, with the object's state before the event; the pipeline writes them as `mem` lines with the
processed frame current at the event."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rtsm.core.pipeline import Pipeline
from rtsm.evaluation.event_log import KIND_MEM, LEDGER_KINDS, LEDGER_SCHEMA, MEM_ACTIONS, EventLogWriter, MemoryEvent
from rtsm.io.ingest_queue import IngestQueue
from rtsm.stores.proximity_index import GridSpec, ProximityIndex
from rtsm.stores.working_memory import WorkingMemory

DIM = 8


class FakeClock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def now_mono(self) -> float:
        return self.t


def _emb(seed: int = 1) -> np.ndarray:
    v = np.arange(1, DIM + 1, dtype=np.float32) * seed
    return v / np.linalg.norm(v)


def _wm(cfg=None, *, index=None, clock=None, sink=None):
    wm = WorkingMemory({"object": {"proto_ttl_s": 10.0, "pose": {"demote_thresh_m": 0.5}}, **(cfg or {})}, index=index, clock=clock or FakeClock())
    wm.event_sink = sink
    return wm


def test_constants_and_schema():
    assert KIND_MEM == "mem" and KIND_MEM in LEDGER_KINDS and LEDGER_SCHEMA == 3 and MEM_ACTIONS == ("expire", "evict", "correct")


def test_proto_expiry_emits_the_state_before_deletion():
    clock, got = FakeClock(1000.0), []
    wm = _wm(clock=clock, sink=got.append)
    oid = wm.create_object(np.array([1.0, 2.0, 3.0], dtype=np.float32), _emb(), label_topk=[("cup", 0.7)], frame_id="ws_4")
    clock.t = 1005.0
    assert wm.expire_timeouts() == [] and got == []                       # not yet
    clock.t = 1011.0
    assert wm.expire_timeouts() == [oid] and wm.get(oid) is None
    assert len(got) == 1
    r = got[0]
    assert r["action"] == "expire" and r["reason"] == "proto_ttl" and r["object_id"] == oid and r["confirmed"] is False
    assert r["p_before"] == [1.0, 2.0, 3.0] and r["frame_id"] == "ws_4" and r["label_primary"] == "cup" and r["hits"] >= 1
    assert r["age_s"] == pytest.approx(11.0) and r["lifetime_s"] == pytest.approx(11.0) and "p_after" not in r


def test_index_eviction_emits_the_victim_and_the_object_stays():
    got = []
    grid = GridSpec(cell_m=0.25, use_3d=False, up_axis="y")
    pi = ProximityIndex(grid, per_cell_cap=1, neighbors_max=16)
    wm = _wm(index=pi, sink=got.append)
    a = wm.create_object(np.array([0.1, 0.0, 0.1], dtype=np.float32), _emb(1))
    assert got == []
    b = wm.create_object(np.array([0.12, 0.0, 0.1], dtype=np.float32), _emb(2))      # same cell: the cap of 1 evicts one id
    assert len(got) == 1 and got[0]["action"] == "evict" and got[0]["reason"] == "cell_overflow"
    victim = got[0]["object_id"]
    assert victim in (a, b) and wm.get(victim) is not None                          # the index forgets it, the memory keeps it
    assert set(pi.nearby_ids(np.array([0.1, 0.0, 0.1], dtype=np.float32))) == {a, b} - {victim}
    assert "lifetime_s" in got[0] and got[0]["p_before"][0] == pytest.approx(0.1 if victim == a else 0.12)


def test_pose_corrections_emit_before_and_after_with_demotion():
    got = []
    wm = _wm(sink=got.append)
    a = wm.create_object(np.array([1.0, 0.0, 0.0], dtype=np.float32), _emb(1), frame_id="ws_1")
    b = wm.create_object(np.array([5.0, 0.0, 0.0], dtype=np.float32), _emb(2), frame_id="ws_9")
    eye = np.eye(3, dtype=np.float32)
    small = {"ws_1": (np.array([0.9, 0.0, 0.0], dtype=np.float32), eye, np.array([0.1, 0.0, 0.0], dtype=np.float32))}
    assert wm.apply_pose_corrections(small) == 2                                      # a direct, b by the nearest camera's delta
    by_id = {r["object_id"]: r for r in got}
    ra, rb = by_id[a], by_id[b]
    assert ra["action"] == "correct" and ra["reason"] == "direct" and ra["frame_id"] == "ws_1" and ra["demoted"] is False
    assert ra["p_before"] == [1.0, 0.0, 0.0] and ra["p_after"] == pytest.approx([1.1, 0.0, 0.0]) and ra["shift_m"] == pytest.approx(0.1, abs=1e-3)
    assert rb["reason"] == "fallback" and rb["frame_id"] == "ws_1" and rb["p_after"] == pytest.approx([5.1, 0.0, 0.0])
    assert wm.get(a).xyz_world.tolist() == pytest.approx([1.1, 0.0, 0.0])
    got.clear()
    big = {"ws_1": (np.array([0.9, 0.0, 0.0], dtype=np.float32), eye, np.array([2.0, 0.0, 0.0], dtype=np.float32))}
    wm.apply_pose_corrections(big)
    ra = {r["object_id"]: r for r in got}[a]
    assert ra["demoted"] is True and ra["shift_m"] == pytest.approx(2.0, abs=1e-3) and wm.get(a).confirmed is False
    assert wm.last_correction["demoted"] == [a, b]


def test_sink_errors_never_reach_the_memory():
    clock = FakeClock(1000.0)

    def boom(_rec):
        raise RuntimeError("sink down")

    wm = _wm(clock=clock, sink=boom)
    oid = wm.create_object(np.array([0.0, 0.0, 1.0], dtype=np.float32), _emb())
    clock.t = 1020.0
    assert wm.expire_timeouts() == [oid]


def test_pipeline_writes_mem_lines_with_the_current_frame(tmp_path: Path):
    clock = FakeClock(1000.0)
    wm = _wm(clock=clock)
    w = EventLogWriter(enabled=True, configured_path=str(tmp_path / "e.jsonl"), ledgers=True)
    pipe = Pipeline(cfg={}, segmenter=None, clip=None, working_mem=wm, proximity_index=None, associator=None,
                    ingest_gate=None, ingest_q=IngestQueue(), sweep_cache=None, event_log=w)
    assert wm.event_sink == pipe._mem_event
    oid = wm.create_object(np.array([0.0, 0.0, 2.0], dtype=np.float32), _emb(), frame_id="ws_3")
    pipe._cur_frame = (123_000_000, 7)
    clock.t = 1011.0
    wm.expire_timeouts()
    w.close()
    rows = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["ledgers"]["schema"] == 3
    mem = [r for r in rows if r["kind"] == "mem"]
    assert len(mem) == 1 and mem[0]["object_id"] == oid and mem[0]["action"] == "expire" and mem[0]["t_sensor_ns"] == 123_000_000 and mem[0]["frame_seq"] == 7
    assert mem[0]["p_before"] == [0.0, 0.0, 2.0] and mem[0]["p_after"] is None and mem[0]["demoted"] is False and mem[0]["frame_id"] == "ws_3"
    # ledgers off: the sink is wired but writes nothing
    w2 = EventLogWriter(enabled=True, configured_path=str(tmp_path / "f.jsonl"), ledgers=False)
    wm2 = _wm(clock=clock)
    Pipeline(cfg={}, segmenter=None, clip=None, working_mem=wm2, proximity_index=None, associator=None,
             ingest_gate=None, ingest_q=IngestQueue(), sweep_cache=None, event_log=w2)
    wm2.create_object(np.array([0.0, 0.0, 2.0], dtype=np.float32), _emb())
    clock.t = 1030.0
    wm2.expire_timeouts()
    w2.close()
    assert [r["kind"] for r in (json.loads(l) for l in (tmp_path / "f.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())] == ["meta"]


def test_event_dataclass_defaults():
    e = MemoryEvent(timestamp=1.0, action="expire", object_id="x", reason="proto_ttl", t_sensor_ns=None, frame_seq=None,
                    confirmed=False, hits=1, stability=0.1, label_primary=None, p_before=[0.0, 0.0, 1.0])
    assert e.kind == "mem" and e.p_after is None and e.demoted is False and e.age_s is None
