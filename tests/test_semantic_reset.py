"""
Tests for /reset ↔ vector store coupling and semantic-search ghost handling.

Covers:
- FaissClient.clear() empties the index (search returns nothing) and reports
  the cleared count.
- POST /reset clears the FAISS vector store, so /search/semantic returns no
  stale ids ("ghost" objects) from the previous map.
- A vector-store id with no WorkingMemory object must report confirmed=False
  (not True) — ghosts were crowding top-K slots advertising confirmed=True.
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


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v + [0.0] * (DIM - len(v)), dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-12)


# Query embedding and two stored embeddings with known cosine scores:
# "obj_real" scores 1.0 against the query, "obj_ghost" ~0.707.
QUERY_EMB = _unit([1.0])
REAL_EMB = _unit([1.0])
GHOST_EMB = _unit([1.0, 1.0])


class FakeClipAdapter:
    """Stands in for the CLIP adapter — returns a fixed query embedding."""

    def encode_text(self, text: str) -> np.ndarray:
        return QUERY_EMB


def _make_object(oid: str, xyz: list[float], label: str = "object") -> ObjectState:
    now = 1000.0
    return ObjectState(
        id=oid,
        xyz_world=np.array(xyz, dtype=np.float32),
        cov_world=np.array([0.01, 0.01, 0.01], dtype=np.float32),
        emb_mean=np.zeros(512, dtype=np.float32),
        emb_gallery=np.zeros((0, 512), dtype=np.float16),
        view_bins={},
        label_scores={label: 1.0},
        label_primary=label,
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
        _dim=512,
    )


def _make_vectors() -> FaissClient:
    vc = FaissClient({"vectors": {"enable": True, "backend": "faiss", "dim": DIM}})
    vc.upsert_batch([
        {"object_id": "obj_real", "emb": REAL_EMB},
        {"object_id": "obj_ghost", "emb": GHOST_EMB},
    ])
    return vc


@pytest.fixture
def client_and_vectors():
    """TestClient whose vector store holds a real object and a ghost id
    (present in FAISS but absent from WorkingMemory)."""
    wm = WorkingMemory(cfg={})
    obj = _make_object("obj_real", [1.0, 0.0, 1.0], "mug")
    wm._map[obj.id] = obj
    vectors = _make_vectors()
    app = create_app(
        working_memory=wm,
        clip_adapter=FakeClipAdapter(),
        vectors=vectors,
        registry=CollectorRegistry(),
    )
    return TestClient(app), vectors


# ─────────────────── FaissClient.clear ───────────────────


class TestFaissClientClear:
    def test_clear_empties_index_and_reports_count(self):
        vc = _make_vectors()
        assert len(vc.search(QUERY_EMB, top_k=5)) == 2
        assert vc.clear() == {"vectors_cleared": 2}
        assert vc.search(QUERY_EMB, top_k=5) == []

    def test_clear_when_already_empty(self):
        vc = FaissClient({"vectors": {"enable": True, "backend": "faiss", "dim": DIM}})
        assert vc.clear() == {"vectors_cleared": 0}

    def test_upsert_after_clear_works(self):
        vc = _make_vectors()
        vc.clear()
        vc.upsert_batch([{"object_id": "obj_new", "emb": REAL_EMB}])
        assert [oid for oid, _ in vc.search(QUERY_EMB, top_k=5)] == ["obj_new"]


# ─────────────────── ghost entries in /search/semantic ───────────────────


class TestSemanticGhostEntries:
    def test_missing_wm_object_reports_unconfirmed(self, client_and_vectors):
        """A FAISS id whose WM lookup fails is stale — it must not advertise
        confirmed=True (ghosts were outranking real objects in top-K)."""
        client, _ = client_and_vectors
        resp = client.get("/search/semantic", params={"query": "mug"})
        assert resp.status_code == 200
        by_id = {r["id"]: r for r in resp.json()["results"]}
        assert by_id["obj_ghost"]["confirmed"] is False
        assert by_id["obj_ghost"]["stability"] == 0.0
        assert by_id["obj_ghost"]["xyz_world"] is None
        # The real object keeps its WM-backed fields
        assert by_id["obj_real"]["confirmed"] is True
        assert by_id["obj_real"]["xyz_world"] == [1.0, 0.0, 1.0]


# ─────────────────── /reset clears the vector store ───────────────────


class TestResetClearsVectors:
    def test_reset_reports_vector_store_cleared(self, client_and_vectors):
        client, _ = client_and_vectors
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["cleared"]["vector_store"] == {"vectors_cleared": 2}

    def test_semantic_search_empty_after_reset(self, client_and_vectors):
        """The live bug: after /reset, /search/semantic kept returning
        old-map ids. The reset must leave no stale ids behind."""
        client, _ = client_and_vectors
        assert len(client.get("/search/semantic", params={"query": "mug"}).json()["results"]) == 2
        client.post("/reset")
        resp = client.get("/search/semantic", params={"query": "mug"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []

    def test_reset_without_vectors_configured(self):
        """No vector store wired (e.g. semantic search disabled) — reset
        must succeed and simply omit the vector_store entry."""
        app = create_app(working_memory=WorkingMemory(cfg={}), registry=CollectorRegistry())
        resp = TestClient(app).post("/reset")
        assert resp.status_code == 200
        assert "vector_store" not in resp.json()["cleared"]
