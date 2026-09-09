"""A failed post-reset FAISS save must be visible, not just logged.

PR #24 (main 034bc43) made ``FaissClient.clear()`` warn that a failed save
after ``/reset`` leaves the pre-reset vectors on disk (a restart resurrects
them as ghosts).  demo2 (ab2999c) added ``stats()["persist_error"]`` and the
``/healthz`` reason ``vector_index_persist_failing`` -- but only the upsert
path wrote ``_last_save_error``; ``clear()`` did not, so a failed post-reset
persist still left ``/healthz`` at ``ok``.  The reconcile merges the two:
``clear()`` records the failure and clears it on the next successful save.

CPU-only (faiss-cpu, fastapi TestClient); no models.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.api.server import create_app
from rtsm.stores.vectors.faiss_client import FaissClient
from rtsm.stores.working_memory import WorkingMemory

DIM = 8


def _cfg(index_path) -> dict:
    return {
        "vectors": {
            "enable": True,
            "backend": "faiss",
            "dim": DIM,
            "faiss": {"index_path": str(index_path)},
        }
    }


def _healthz(vc: FaissClient) -> dict:
    app = create_app(
        working_memory=WorkingMemory(cfg={}),
        vectors=vc,
        registry=CollectorRegistry(),
    )
    return TestClient(app).get("/healthz").json()


def test_failed_post_reset_save_is_recorded_then_cleared(tmp_path, monkeypatch):
    vc = FaissClient(_cfg(tmp_path / "faiss" / "index.flatip"))
    vc.upsert_batch([{"object_id": "a", "emb": np.ones(DIM, np.float32) / np.sqrt(DIM)}])
    assert vc.stats()["count"] == 1
    assert vc.stats()["persist_error"] is None  # save() created the dirs

    def _boom(self, path):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(FaissClient, "save", _boom)
    out = vc.clear()  # /reset stays non-fatal ...
    assert out == {"vectors_cleared": 1}
    st = vc.stats()
    assert st["count"] == 0
    assert st["persist_error"] and "disk full" in st["persist_error"]  # ... but is visible
    body = _healthz(vc)
    assert body["status"] == "degraded"
    assert any("vector_index_persist_failing" in r for r in body["reasons"])

    monkeypatch.undo()  # the disk is back
    vc.clear()
    assert vc.stats()["persist_error"] is None
    body = _healthz(vc)
    assert body["status"] == "ok"
    assert "reasons" not in body
