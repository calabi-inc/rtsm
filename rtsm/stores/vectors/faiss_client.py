from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FaissClient:
    """
    Minimal FAISS-backed vector store for MVP.

    Interface mirrors expected methods used by the pipeline's LTM hook:
    - upsert_batch(records)
    - search(emb, top_k)
    - delete(ids)
    - clear()
    - save(path)
    - load(path)
    - stats()
    - close()

    Records are dicts produced by WorkingMemory.collect_ready_for_upsert():
        {
            "object_id": str,
            "emb": np.ndarray (float32, L2-normalized),
            ... other metadata we keep in a side map ...
        }

    Notes:
    - Uses IndexFlatIP over L2-normalized embeddings => cosine similarity == inner product.
    - Maintains an in-memory ID→row mapping and arrays; rebuild is O(N) on upsert.
      This is acceptable for MVP scale. Optimize later with IDMap or IVF if needed.
    - A single lock guards index + row maps: the pipeline thread upserts while
      the API thread searches, and a reset()/add() rebuild mid-search is unsafe.
    - Failures are tracked (last_error / consecutive_failures) and surfaced via
      stats() so /healthz can report semantic-search degradation instead of the
      index silently starving (2026-08-15: boots where every query returned <=1
      hit while WM held 49 confirmed objects).
    """

    def __init__(self, cfg: Dict[str, Any]):
        if faiss is None:
            raise ImportError(
                "faiss is not installed. Please install faiss-cpu for MVP usage."
            )
        vcfg = cfg.get("vectors", {})
        self.enabled: bool = bool(vcfg.get("enable", True))
        self.dim: int = int(vcfg.get("dim", 512))
        self.persistent_path: Optional[str] = vcfg.get("faiss", {}).get("index_path")
        self._index: Optional[Any] = None
        self._id_to_row: Dict[str, int] = {}
        self._row_to_id: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()

        # Health / telemetry — surfaced via stats() → /healthz, /stats/detailed
        self.last_error: Optional[str] = None
        self.total_failures: int = 0
        self.consecutive_failures: int = 0
        self._last_save_error: Optional[str] = None

        # eager load if path exists (a bad persisted state must not brick the
        # boot: discard it loudly and start fresh)
        if self.persistent_path and os.path.exists(self.persistent_path):
            try:
                self.load(self.persistent_path)
                logger.info(
                    f"[FAISS] loaded persisted index from {self.persistent_path} "
                    f"({len(self._row_to_id)} vectors, dim={self.dim})"
                )
            except Exception as e:
                logger.error(
                    f"[FAISS] failed to load persisted index {self.persistent_path} "
                    f"({type(e).__name__}: {e}); starting with an empty index"
                )
                self._index = None
                self._id_to_row = {}
                self._row_to_id = []
                self._metadata = {}
                self._embeddings = {}
                self._ensure_index()
        else:
            self._ensure_index()

    # ---------- public API ----------
    def upsert_batch(self, records: Iterable[Dict[str, Any]]) -> None:
        """Insert or update vectors by object_id. Rebuilds the FAISS index.
        Accepts an iterable of payload dicts as produced by WorkingMemory.
        Raises on failure (caller decides retry policy); the failure is also
        recorded for stats()/healthz.
        """
        try:
            with self._lock:
                self._upsert_batch_locked(records)
        except Exception as e:
            self.total_failures += 1
            self.consecutive_failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            raise
        self.consecutive_failures = 0

    def _upsert_batch_locked(self, records: Iterable[Dict[str, Any]]) -> None:
        to_add: List[Tuple[str, np.ndarray, Dict[str, Any]]] = []
        for rec in records:
            oid = str(rec["object_id"])  # required
            emb = np.asarray(rec["emb"], dtype=np.float32)
            if emb.ndim != 1:
                emb = emb.reshape(-1)
            if emb.shape[0] != self.dim:
                raise ValueError(
                    f"Embedding dim {emb.shape[0]} != configured dim {self.dim} "
                    f"(object_id={oid})"
                )
            # Store normalized inputs (should already be L2-normalized)
            n = float(np.linalg.norm(emb) + 1e-12)
            emb = (emb / n).astype(np.float32)
            meta = {k: v for k, v in rec.items() if k not in ("emb",)}
            to_add.append((oid, emb, meta))

        # Merge/update metadata and materialize dense arrays
        for oid, emb, meta in to_add:
            self._metadata[oid] = meta
            self._embeddings[oid] = emb

        # Rebuild arrays from shadow stores
        ids_sorted = sorted(self._embeddings.keys())  # deterministic rebuild
        embs = np.zeros((len(ids_sorted), self.dim), dtype=np.float32)
        for row, oid in enumerate(ids_sorted):
            embs[row] = self._embeddings[oid]

        self._row_to_id = ids_sorted
        self._id_to_row = {oid: i for i, oid in enumerate(ids_sorted)}

        self._ensure_index()
        assert self._index is not None
        self._index.reset()
        if len(embs) > 0:
            self._index.add(embs)

        # auto-save if configured. A persistence failure must not fail the
        # in-memory upsert, but it must not be silent either — a dead save
        # path means every boot starts with an empty index.
        if self.persistent_path:
            try:
                self.save(self.persistent_path)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if msg != self._last_save_error:  # don't spam per-flush
                    logger.warning(
                        f"[FAISS] index save to {self.persistent_path} failed: {msg}"
                    )
                self._last_save_error = msg
            else:
                if self._last_save_error is not None:
                    logger.info(
                        f"[FAISS] index save to {self.persistent_path} recovered"
                    )
                self._last_save_error = None

    def search(self, emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top_k (object_id, score) by inner product (cosine)"""
        with self._lock:
            if self._index is None or len(self._row_to_id) == 0:
                return []
            q = np.asarray(emb, dtype=np.float32).reshape(1, -1)
            if q.shape[1] != self.dim:
                raise ValueError(f"Query dim {q.shape[1]} != configured dim {self.dim}")
            # normalize to keep cosine semantics
            n = float(np.linalg.norm(q) + 1e-12)
            q = (q / n).astype(np.float32)
            D, I = self._index.search(q, min(top_k, len(self._row_to_id)))
            out: List[Tuple[str, float]] = []
            for d, i in zip(D[0].tolist(), I[0].tolist()):
                if i == -1 or i >= len(self._row_to_id):
                    continue
                out.append((self._row_to_id[int(i)], float(d)))
            return out

    def delete(self, ids: Iterable[str]) -> None:
        """Remove ids and rebuild index from the remaining embeddings."""
        with self._lock:
            id_set = set(str(x) for x in ids)
            for oid in list(self._metadata.keys()):
                if oid in id_set:
                    del self._metadata[oid]
            for oid in list(self._embeddings.keys()):
                if oid in id_set:
                    del self._embeddings[oid]
            ids_sorted = sorted(self._embeddings.keys())
            self._row_to_id = ids_sorted
            self._id_to_row = {oid: i for i, oid in enumerate(ids_sorted)}
            self._ensure_index()
            self._index.reset()
            if ids_sorted:
                embs = np.vstack([self._embeddings[oid] for oid in ids_sorted]).astype(np.float32)
                self._index.add(embs)
            if self.persistent_path:
                try:
                    self.save(self.persistent_path)
                except Exception as e:
                    logger.warning(
                        f"[FAISS] index save after delete failed: {type(e).__name__}: {e}"
                    )

    def clear(self) -> Dict[str, int]:
        """
        Remove all vectors and reset the index (called on /reset).

        Returns dict with count of what was cleared.
        """
        with self._lock:
            vec_count = len(self._row_to_id)
            self._metadata.clear()
            self._embeddings.clear()
            self._row_to_id = []
            self._id_to_row = {}
            self._ensure_index()
            self._index.reset()
            # Persist the empty state, else a restart reloads the old vectors
            if self.persistent_path:
                try:
                    self.save(self.persistent_path)
                except Exception as e:
                    logger.warning(
                        f"[FAISS] index save after clear failed: {type(e).__name__}: {e}"
                    )
            return {"vectors_cleared": vec_count}

    def save(self, path: str) -> None:
        with self._lock:
            if self._index is None:
                self._ensure_index()
            # faiss.write_index does not create parent directories; a missing
            # dir made every save fail silently before (index never persisted).
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Persist both index and id list
            faiss.write_index(self._index, path)
            with open(path + ".ids", "w", encoding="utf-8") as f:
                for oid in self._row_to_id:
                    f.write(oid + "\n")
            # Persist embeddings aligned with ids list
            if self._row_to_id:
                embs = np.vstack([self._embeddings[oid] for oid in self._row_to_id]).astype(np.float32)
                np.save(path + ".embs.npy", embs)

    def load(self, path: str) -> None:
        """Load persisted index + ids + embeddings. Raises if the persisted
        trio is missing pieces or inconsistent (torn write, dim change) —
        callers should treat that as "start fresh", not partially load.
        """
        with self._lock:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            index = faiss.read_index(path)
            if int(index.d) != self.dim:
                raise ValueError(
                    f"persisted index dim {int(index.d)} != configured dim {self.dim} "
                    f"(embedding model changed?)"
                )
            ids_path = path + ".ids"
            row_to_id: List[str] = []
            if os.path.exists(ids_path):
                with open(ids_path, "r", encoding="utf-8") as f:
                    row_to_id = [line.strip() for line in f if line.strip()]
            if len(row_to_id) != int(index.ntotal):
                raise ValueError(
                    f"persisted ids ({len(row_to_id)}) != index vectors "
                    f"({int(index.ntotal)}) — torn write?"
                )
            # Embeddings shadow store must align too: upsert_batch rebuilds the
            # whole index from it, so un-shadowed vectors would silently vanish
            # on the first flush.
            embeddings: Dict[str, np.ndarray] = {}
            if row_to_id:
                embs_path = path + ".embs.npy"
                if not os.path.exists(embs_path):
                    raise ValueError(f"missing embeddings shadow file {embs_path}")
                embs = np.load(embs_path).astype(np.float32)
                if embs.ndim != 2 or embs.shape[0] != len(row_to_id) or embs.shape[1] != self.dim:
                    raise ValueError(
                        f"embeddings shadow shape {embs.shape} inconsistent with "
                        f"{len(row_to_id)} ids of dim {self.dim}"
                    )
                for oid, vec in zip(row_to_id, embs):
                    embeddings[oid] = vec
            # Commit only after full validation
            self._index = index
            self._row_to_id = row_to_id
            self._id_to_row = {oid: i for i, oid in enumerate(row_to_id)}
            self._embeddings = embeddings

    def stats(self) -> Dict[str, Any]:
        """Health snapshot for /healthz and /stats/detailed."""
        with self._lock:
            return {
                "backend": "faiss",
                "count": len(self._row_to_id),
                "index_ntotal": int(self._index.ntotal) if self._index is not None else 0,
                "dim": self.dim,
                "last_error": self.last_error,
                "total_failures": self.total_failures,
                "consecutive_failures": self.consecutive_failures,
                "persist_path": self.persistent_path,
                "persist_error": self._last_save_error,
            }

    def close(self) -> None:
        with self._lock:
            self._index = None

    # ---------- helpers ----------
    def _ensure_index(self) -> None:
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.dim)
