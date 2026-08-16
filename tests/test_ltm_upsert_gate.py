"""
Tests for the WM → vector-store upsert path and its failure visibility.

Covers the 2026-08-15 half-dead-retrieval boots: objects confirmed normally
(object.require_view_bins loosened to 1 for stationary cameras) but the LTM
scheduler's separate view-diversity gate (ltm_min_view_bins, old default 2)
silently starved the FAISS index — /search/semantic returned <=1 hit while
WorkingMemory held 49 confirmed objects. No exception ever fired, so nothing
was logged and /healthz stayed "ok".

- ltm_min_view_bins now defaults to object.require_view_bins (confirmable ⇒
  searchable); an explicit stricter override still works but warns at init.
- mark_upsert_failed() rolls back collect bookkeeping so a failed store write
  retries on the next flush instead of vanishing until the force period.
- FaissClient records failures for /healthz, creates missing parent dirs on
  save (persistence silently never worked before), rejects torn persisted
  state instead of half-loading it, and rebuilds the index on delete().
- /healthz reports "degraded" when upserts fail or when confirmed objects far
  outnumber indexed vectors.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.api.server import create_app
from rtsm.stores.vectors.faiss_client import FaissClient
from rtsm.stores.working_memory import WorkingMemory, ObjectState

DIM = 8

WM_CFG = {
    "object": {
        "proto_ttl_s": 10.0,
        "promote_hits": 2,
        "stability_promote": 0.55,
        "require_view_bins": 1,  # matches rtsm.yaml stationary-camera loosening
    },
    # no "ltm" section — exercises the default (must follow require_view_bins)
}

CENTER_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # object at image center
OFFSIDE_DIR = np.array([0.9, 0.0, 0.5], dtype=np.float32)  # far off-center (other az bin)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


class Obs:
    """Minimal observation envelope for WorkingMemory.update_object."""

    def __init__(self, p, e, vdir):
        self.p_world = p
        self.emb_vis = e
        self.view_dir_cam = vdir
        self.centroid_px = (100.0, 100.0)
        self.depth_valid = 1.0
        self.quality = 1.0
        self.label_topk = [("cup", 0.9)]
        self.crop = None
        self.frame_id = None


def _confirmed_object(wm: WorkingMemory, emb: np.ndarray, view_dir: np.ndarray,
                      p=(1.0, 0.0, 2.0), updates: int = 10) -> str:
    p = np.array(p, dtype=np.float32)
    oid = wm.create_object(p, emb, view_dir_cam=view_dir)
    for _ in range(updates):
        wm.update_object(oid, Obs(p, emb, view_dir))
        wm.maybe_promote(oid)
    assert wm.get(oid).confirmed
    return oid


# ─────────────────── LTM view-bin gate (root cause) ───────────────────


class TestLtmViewBinGate:
    def test_single_view_bin_confirmed_object_is_collected(self):
        """A stationary camera confirms objects with one view bin; they must
        still reach the vector store (this starved on 2026-08-15)."""
        wm = WorkingMemory(WM_CFG)
        assert wm.ltm_min_view_bins == 1
        oid = _confirmed_object(wm, _unit(np.arange(1, DIM + 1)), CENTER_DIR)
        assert len(wm.get(oid).view_bins) == 1

        ready = wm.collect_ready_for_upsert()
        assert [p["object_id"] for p in ready] == [oid]

    def test_gate_default_follows_require_view_bins(self):
        wm_default = WorkingMemory({})  # require_view_bins default 2
        assert wm_default.ltm_min_view_bins == wm_default.require_view_bins == 2

    def test_explicit_stricter_gate_still_respected(self):
        """An explicit ltm_min_view_bins override wins (and warns at init);
        single-bin objects then stay out of the store by configured choice."""
        cfg = dict(WM_CFG)
        cfg["ltm"] = {"ltm_min_view_bins": 2}
        wm = WorkingMemory(cfg)
        _confirmed_object(wm, _unit(np.arange(1, DIM + 1)), CENTER_DIR)
        assert wm.collect_ready_for_upsert() == []
        # ...and the starvation is visible in stats
        assert wm.stats()["ltm_never_upserted"] == 1

    def test_two_view_bins_pass_a_stricter_gate(self):
        cfg = dict(WM_CFG)
        cfg["ltm"] = {"ltm_min_view_bins": 2}
        wm = WorkingMemory(cfg)
        emb = _unit(np.arange(1, DIM + 1))
        p = np.array([1.0, 0.0, 2.0], dtype=np.float32)
        oid = wm.create_object(p, emb, view_dir_cam=CENTER_DIR)
        for vdir in (CENTER_DIR, OFFSIDE_DIR, CENTER_DIR, OFFSIDE_DIR):
            wm.update_object(oid, Obs(p, emb, vdir))
            wm.maybe_promote(oid)
        assert len(wm.get(oid).view_bins) == 2
        assert [x["object_id"] for x in wm.collect_ready_for_upsert()] == [oid]


# ─────────────────── failed-flush rollback ───────────────────


class TestMarkUpsertFailed:
    def test_requeues_for_immediate_retry(self):
        wm = WorkingMemory(WM_CFG)
        oid = _confirmed_object(wm, _unit(np.arange(1, DIM + 1)), CENTER_DIR)

        first = wm.collect_ready_for_upsert()
        assert len(first) == 1
        # Without a rollback the object looks freshly upserted: unchanged
        # embedding + min_period means the next collect returns nothing.
        assert wm.collect_ready_for_upsert() == []

        assert wm.mark_upsert_failed([oid]) == 1
        retry = wm.collect_ready_for_upsert()
        assert [p["object_id"] for p in retry] == [oid]

    def test_unknown_ids_are_ignored(self):
        wm = WorkingMemory(WM_CFG)
        assert wm.mark_upsert_failed(["nope"]) == 0


# ─────────────────── FaissClient hardening ───────────────────


def _vec_cfg(dim: int = DIM, index_path=None) -> dict:
    cfg = {"vectors": {"enable": True, "backend": "faiss", "dim": dim}}
    if index_path is not None:
        cfg["vectors"]["faiss"] = {"index_path": str(index_path)}
    return cfg


class TestFaissClientHardening:
    def test_upsert_failure_is_recorded_and_raised(self):
        vc = FaissClient(_vec_cfg())
        with pytest.raises(ValueError):
            vc.upsert_batch([{"object_id": "bad", "emb": np.ones(DIM + 1, np.float32)}])
        st = vc.stats()
        assert st["consecutive_failures"] == 1
        assert st["total_failures"] == 1
        assert "Embedding dim" in st["last_error"]

        vc.upsert_batch([{"object_id": "ok", "emb": _unit(np.ones(DIM))}])
        st = vc.stats()
        assert st["consecutive_failures"] == 0
        assert st["total_failures"] == 1  # cumulative history kept
        assert st["count"] == 1

    def test_save_creates_missing_parent_dirs_and_roundtrips(self, tmp_path):
        """model_store/faiss/ never existed, so every auto-save failed silently
        and each boot started with an empty index."""
        index_path = tmp_path / "nested" / "faiss" / "index.flatip"
        vc = FaissClient(_vec_cfg(index_path=index_path))
        vc.upsert_batch([
            {"object_id": "a", "emb": _unit(np.eye(DIM, dtype=np.float32)[0])},
            {"object_id": "b", "emb": _unit(np.eye(DIM, dtype=np.float32)[1])},
        ])
        assert index_path.exists()
        assert vc.stats()["persist_error"] is None

        vc2 = FaissClient(_vec_cfg(index_path=index_path))
        assert vc2.stats()["count"] == 2
        hits = vc2.search(np.eye(DIM, dtype=np.float32)[0], top_k=2)
        assert hits[0][0] == "a"

    def test_torn_persisted_state_is_discarded_not_half_loaded(self, tmp_path):
        index_path = tmp_path / "faiss" / "index.flatip"
        vc = FaissClient(_vec_cfg(index_path=index_path))
        vc.upsert_batch([
            {"object_id": "a", "emb": _unit(np.eye(DIM, dtype=np.float32)[0])},
            {"object_id": "b", "emb": _unit(np.eye(DIM, dtype=np.float32)[1])},
        ])
        # Simulate a torn write: ids file loses a line
        (index_path.parent / "index.flatip.ids").write_text("a\n", encoding="utf-8")

        vc2 = FaissClient(_vec_cfg(index_path=index_path))
        assert vc2.stats()["count"] == 0  # fresh start, no crash, no ghost rows

    def test_dim_change_discards_persisted_index(self, tmp_path):
        index_path = tmp_path / "faiss" / "index.flatip"
        vc = FaissClient(_vec_cfg(dim=DIM, index_path=index_path))
        vc.upsert_batch([{"object_id": "a", "emb": _unit(np.ones(DIM))}])

        vc2 = FaissClient(_vec_cfg(dim=DIM * 2, index_path=index_path))
        assert vc2.stats()["count"] == 0
        assert vc2.stats()["dim"] == DIM * 2

    def test_delete_keeps_survivors_searchable(self):
        """delete() used to reset the index and never re-add survivors."""
        vc = FaissClient(_vec_cfg())
        vc.upsert_batch([
            {"object_id": "a", "emb": _unit(np.eye(DIM, dtype=np.float32)[0])},
            {"object_id": "b", "emb": _unit(np.eye(DIM, dtype=np.float32)[1])},
        ])
        vc.delete(["a"])
        hits = vc.search(np.eye(DIM, dtype=np.float32)[1], top_k=5)
        assert [h[0] for h in hits] == ["b"]


# ─────────────────── pipeline flush wiring ───────────────────


class TestPipelineFlush:
    """End-to-end: _maybe_flush_vectors must retry failed batches and keep
    the pipeline thread alive when collection itself throws."""

    @staticmethod
    def _make_pipe(wm, vectors):
        from rtsm.core.pipeline import Pipeline
        pipe = Pipeline.__new__(Pipeline)  # skip heavy __init__ (models)
        pipe.cfg = {"vectors": {"flush_period_s": 0.0}}
        pipe.working_mem = wm
        pipe.vectors = vectors
        pipe._last_flush_ts = 0.0
        pipe._vector_flush_failures = 0
        return pipe

    def test_failed_flush_requeues_and_recovers(self):
        wm = WorkingMemory(WM_CFG)
        oid = _confirmed_object(wm, _unit(np.arange(1, DIM + 1)), CENTER_DIR)

        class FlakyStore:
            def __init__(self):
                self.fail = True
                self.upserted = []

            def upsert_batch(self, records):
                if self.fail:
                    raise RuntimeError("simulated index-add failure")
                self.upserted.extend(r["object_id"] for r in records)

            def stats(self):
                return {"count": len(self.upserted)}

        store = FlakyStore()
        pipe = self._make_pipe(wm, store)

        pipe._maybe_flush_vectors()  # fails → must requeue, not raise
        assert pipe._vector_flush_failures == 1
        assert store.upserted == []

        store.fail = False
        pipe._last_flush_ts = 0.0
        pipe._maybe_flush_vectors()  # retry succeeds
        assert store.upserted == [oid]

    def test_collect_exception_does_not_kill_pipeline(self):
        class BrokenWM:
            def collect_ready_for_upsert(self):
                raise RuntimeError("boom")

        pipe = self._make_pipe(BrokenWM(), FaissClient(_vec_cfg()))
        pipe._maybe_flush_vectors()  # must swallow + log, not propagate


# ─────────────────── /healthz degradation ───────────────────


def _make_confirmed_state(oid: str) -> ObjectState:
    now = 1000.0
    return ObjectState(
        id=oid,
        xyz_world=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        cov_world=np.array([0.01, 0.01, 0.01], dtype=np.float32),
        emb_mean=np.zeros(DIM, dtype=np.float32),
        emb_gallery=np.zeros((0, DIM), dtype=np.float16),
        view_bins={},
        label_scores={"mug": 1.0},
        label_primary="mug",
        stability=0.8,
        hits=5,
        confirmed=True,
        created_mono=now,
        created_wall_utc=now,
        last_seen_mono=now,
        last_seen_wall_utc=now,
        last_seen_px=None,
        last_upsert_wall_utc=0.0,
        last_upsert_mono=0.0,
        last_upsert_emb=None,
        last_upsert_xyz=None,
        image_crops=[],
        last_update_frame_id=None,
        _dim=DIM,
    )


def _make_app(wm: WorkingMemory, vectors: FaissClient) -> TestClient:
    app = create_app(
        working_memory=wm,
        vectors=vectors,
        registry=CollectorRegistry(),
    )
    return TestClient(app)


class TestHealthzDegradation:
    def test_ok_when_index_tracks_working_memory(self):
        wm = WorkingMemory(cfg={})
        vc = FaissClient(_vec_cfg())
        client = _make_app(wm, vc)
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert "reasons" not in body

    def test_degraded_on_semantic_index_lag(self):
        """The observed failure shape: WM full of confirmed objects, index
        (nearly) empty — /healthz must say so instead of 'ok'."""
        wm = WorkingMemory(cfg={})
        for i in range(12):
            o = _make_confirmed_state(f"obj_{i}")
            wm._map[o.id] = o
        vc = FaissClient(_vec_cfg())
        vc.upsert_batch([{"object_id": "obj_0", "emb": _unit(np.ones(DIM))}])

        body = _make_app(wm, vc).get("/healthz").json()
        assert body["status"] == "degraded"
        assert any("semantic_index_lag" in r for r in body["reasons"])
        assert body["semantic_index"] == {"confirmed": 12, "indexed": 1, "lag": 11}

    def test_degraded_while_upserts_failing(self):
        wm = WorkingMemory(cfg={})
        vc = FaissClient(_vec_cfg())
        with pytest.raises(ValueError):
            vc.upsert_batch([{"object_id": "bad", "emb": np.ones(DIM + 3, np.float32)}])

        body = _make_app(wm, vc).get("/healthz").json()
        assert body["status"] == "degraded"
        assert any("vector_upserts_failing" in r for r in body["reasons"])

    def test_stats_detailed_exposes_vector_store(self):
        wm = WorkingMemory(cfg={})
        vc = FaissClient(_vec_cfg())
        body = _make_app(wm, vc).get("/stats/detailed").json()
        assert body["vectors"]["backend"] == "faiss"
        assert body["vectors"]["count"] == 0
