from __future__ import annotations

import time
import threading
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass

import base64
import numpy as np
from fastapi import FastAPI, Response, HTTPException
from prometheus_client import Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST, REGISTRY


@dataclass
class ResetComponents:
    """Components that can be reset without restarting RTSM."""
    sweep_cache: Any = None
    frame_window: Any = None
    vis_server: Any = None  # VisualizationServer with registry


def create_app(
    *,
    working_memory: Any,
    clip_adapter: Optional[Any] = None,
    vectors: Optional[Any] = None,
    extra_stats_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    registry: Optional[CollectorRegistry] = None,
    reset_components: Optional[ResetComponents] = None,
) -> FastAPI:
    """
    Build a FastAPI app exposing:
      - /healthz: liveness
      - /readyz: readiness (trivial true for now)
      - /stats: JSON snapshot (WorkingMemory.stats() + optional extra stats)
      - /metrics: Prometheus metrics (mounted ASGI app)

    The app expects a `working_memory` with a `stats()` method.
    """
    app = FastAPI(title="RTSM API — Real-Time Spatio-Semantic Memory", version="1.0.0")

    # ---------------- Prometheus metrics ----------------
    # Create a few dynamic gauges that read values from WorkingMemory on scrape.
    # Default to the global REGISTRY when a custom registry isn't provided.
    reg = registry or REGISTRY
    objects_gauge = Gauge(
        "rtsm_working_objects",
        "Total objects in WorkingMemory",
        registry=reg,
    )
    confirmed_gauge = Gauge(
        "rtsm_confirmed_objects",
        "Confirmed objects in WorkingMemory",
        registry=reg,
    )
    upserts_total_gauge = Gauge(
        "rtsm_upserts_total",
        "Total upserts emitted by WorkingMemory",
        registry=reg,
    )

    def _wm_stat_val(key: str) -> Callable[[], float]:
        def _f() -> float:
            try:
                st = working_memory.stats()
                v = float(st.get(key, 0.0))
                return v
            except Exception:
                return 0.0
        return _f

    objects_gauge.set_function(_wm_stat_val("objects"))
    confirmed_gauge.set_function(_wm_stat_val("confirmed"))
    upserts_total_gauge.set_function(_wm_stat_val("upserts_total"))

    # Expose metrics at /metrics directly (avoid nested /metrics/metrics when mounting)
    @app.get("/metrics")
    def metrics() -> Response:
        data = generate_latest(registry=reg)
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    # ---------------- Routes ----------------
    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> Dict[str, str]:
        # TODO: add checks for external deps (Milvus, FAISS, ZMQ subscriber)
        return {"status": "ready"}

    @app.get("/stats")
    def stats() -> Dict[str, Any]:
        base = {}
        try:
            base = dict(working_memory.stats())
        except Exception:
            base = {}
        if extra_stats_provider is not None:
            try:
                extra = extra_stats_provider() or {}
                base.update(extra)
            except Exception:
                pass
        return base

    # ---- Object debug endpoints ----
    def _obj_summary(o: Any) -> Dict[str, Any]:
        try:
            return {
                "id": getattr(o, "id", None),
                "xyz_world": getattr(o, "xyz_world", None).tolist() if getattr(o, "xyz_world", None) is not None else None,
                "created_wall_utc": float(getattr(o, "created_wall_utc", 0.0)),
                "created_mono": float(getattr(o, "created_mono", 0.0)),
                "stability": float(getattr(o, "stability", 0.0)),
                "hits": int(getattr(o, "hits", 0)),
                "confirmed": bool(getattr(o, "confirmed", False)),
                "label_primary": getattr(o, "label_primary", None),
                "view_bins": len(getattr(o, "view_bins", {}) or {}),
                "last_seen_mono": float(getattr(o, "last_seen_mono", 0.0)),
            }
        except Exception:
            return {"id": getattr(o, "id", None)}

    def _obj_detail(o: Any, *, include_vectors: bool = False) -> Dict[str, Any]:
        d = _obj_summary(o)
        try:
            d.update({
                "cov_world": getattr(o, "cov_world", None).tolist() if getattr(o, "cov_world", None) is not None else None,
                "label_scores": dict(getattr(o, "label_scores", {}) or {}),
                "last_seen_wall_utc": float(getattr(o, "last_seen_wall_utc", 0.0)),
                "last_seen_px": list(getattr(o, "last_seen_px", [])) if getattr(o, "last_seen_px", None) is not None else None,
                "upsert": {
                    "last_upsert_wall_utc": float(getattr(o, "last_upsert_wall_utc", 0.0)),
                    "last_upsert_mono": float(getattr(o, "last_upsert_mono", 0.0)),
                },
                "view_bins_keys": list((getattr(o, "view_bins", {}) or {}).keys()),
            })
            if include_vectors:
                emb_mean = getattr(o, "emb_mean", None)
                d["emb_mean"] = emb_mean.tolist() if emb_mean is not None else None
                emb_gallery = getattr(o, "emb_gallery", None)
                if emb_gallery is not None:
                    try:
                        d["emb_gallery_shape"] = list(emb_gallery.shape)
                        # Avoid dumping entire gallery by default; include if requested
                        d["emb_gallery"] = emb_gallery.astype(float).tolist()
                    except Exception:
                        d["emb_gallery"] = None
        except Exception:
            pass
        return d

    @app.get("/objects")
    def list_objects(include_vectors: bool = False) -> Dict[str, Any]:
        try:
            objs: List[Any] = working_memory.iter_objects()
        except Exception:
            objs = []
        return {
            "count": len(objs),
            "objects": [(_obj_detail(o, include_vectors=include_vectors) if include_vectors else _obj_summary(o)) for o in objs],
        }

    @app.get("/objects/{oid}")
    def get_object(oid: str, include_vectors: bool = False) -> Dict[str, Any]:
        try:
            o = working_memory.get(oid)
        except Exception:
            o = None
        if o is None:
            return {"error": "not_found", "id": oid}
        return _obj_detail(o, include_vectors=include_vectors)

    # ---- Snapshot gallery endpoints ----
    @app.get("/objects/{oid}/snapshots")
    def get_object_snapshots(oid: str, index: Optional[int] = None) -> Dict[str, Any]:
        """
        Get image crop gallery for an object.

        Args:
            oid: Object ID
            index: Optional specific index (0 = most recent, -1 = oldest)

        Returns:
            List of base64-encoded JPEG images (most recent first)
        """
        try:
            o = working_memory.get(oid)
        except Exception:
            o = None
        if o is None:
            raise HTTPException(status_code=404, detail=f"Object {oid} not found")

        crops = getattr(o, 'image_crops', []) or []
        if not crops:
            return {"id": oid, "count": 0, "snapshots": []}

        # Reverse order so index 0 is most recent
        crops_reversed = list(reversed(crops))

        if index is not None:
            if index < 0 or index >= len(crops_reversed):
                raise HTTPException(status_code=404, detail=f"Snapshot index {index} out of range (have {len(crops_reversed)})")
            jpeg_bytes = crops_reversed[index]
            b64 = base64.b64encode(jpeg_bytes).decode('ascii')
            return {
                "id": oid,
                "index": index,
                "total": len(crops_reversed),
                "snapshot": f"data:image/jpeg;base64,{b64}",
            }

        # Return all snapshots
        snapshots = []
        for i, jpeg_bytes in enumerate(crops_reversed):
            b64 = base64.b64encode(jpeg_bytes).decode('ascii')
            snapshots.append({
                "index": i,
                "data": f"data:image/jpeg;base64,{b64}",
                "size_bytes": len(jpeg_bytes),
            })

        return {
            "id": oid,
            "count": len(snapshots),
            "snapshots": snapshots,
        }

    @app.get("/objects/{oid}/snapshots/{index}/image")
    def get_object_snapshot_image(oid: str, index: int) -> Response:
        """Get raw JPEG image for a specific snapshot."""
        try:
            o = working_memory.get(oid)
        except Exception:
            o = None
        if o is None:
            raise HTTPException(status_code=404, detail=f"Object {oid} not found")

        crops = getattr(o, 'image_crops', []) or []
        if not crops:
            raise HTTPException(status_code=404, detail=f"Object {oid} has no snapshots")

        crops_reversed = list(reversed(crops))
        if index < 0 or index >= len(crops_reversed):
            raise HTTPException(status_code=404, detail=f"Snapshot index {index} out of range")

        return Response(content=crops_reversed[index], media_type="image/jpeg")

    # ---- Object debug endpoint ----
    @app.get("/objects/{oid}/debug")
    def get_object_debug(oid: str) -> Dict[str, Any]:
        """Get detailed diagnostic information for an object."""
        try:
            o = working_memory.get(oid)
        except Exception:
            o = None
        if o is None:
            return {"error": "not_found", "id": oid}

        xyz = getattr(o, "xyz_world", None)
        cov = getattr(o, "cov_world", None)

        return {
            "id": oid,
            "position": {
                "xyz_world": xyz.tolist() if xyz is not None else None,
                "cov_world": cov.tolist() if cov is not None else None,
                "cov_diag_cm": [float(np.sqrt(c) * 100) for c in cov] if cov is not None else None,
            },
            "tracking": {
                "hits": int(getattr(o, "hits", 0)),
                "stability": float(getattr(o, "stability", 0.0)),
                "confirmed": bool(getattr(o, "confirmed", False)),
                "last_seen_px": list(getattr(o, "last_seen_px", [])) if getattr(o, "last_seen_px", None) else None,
            },
            "labels": {
                "primary": getattr(o, "label_primary", None),
                "scores": dict(getattr(o, "label_scores", {}) or {}),
            },
            "view_diversity": {
                "bins_filled": len(getattr(o, "view_bins", {}) or {}),
                "bin_ids": list((getattr(o, "view_bins", {}) or {}).keys()),
            },
            "gallery": {
                "image_crops_count": len(getattr(o, "image_crops", []) or []),
                "emb_gallery_shape": list(getattr(o, "emb_gallery", np.array([])).shape) if getattr(o, "emb_gallery", None) is not None else None,
            },
            "timestamps": {
                "created_wall_utc": float(getattr(o, "created_wall_utc", 0.0)),
                "last_seen_wall_utc": float(getattr(o, "last_seen_wall_utc", 0.0)),
                "age_s": time.time() - float(getattr(o, "created_wall_utc", time.time())),
            },
        }

    # ---- Reset endpoint ----
    @app.post("/reset")
    def reset() -> Dict[str, Any]:
        """
        Reset RTSM runtime state while keeping models loaded.

        Clears:
        - WorkingMemory (all objects, proto/confirmed)
        - ProximityIndex (spatial grid, via WM.clear())
        - SweepCache (sweep timestamps, camera snapshots)
        - FrameWindow (buffered RGB-D frames)
        - VisualizationServer registry (keyframes/point clouds)

        Does NOT clear:
        - FastSAM / CLIP models (expensive to reload)
        - FAISS LTM vectors (preserves long-term memory)
        - Configuration
        """
        result: Dict[str, Any] = {
            "status": "ok",
            "reset_time_utc": time.time(),
            "cleared": {},
        }

        # Clear WorkingMemory (also clears attached ProximityIndex)
        try:
            wm_result = working_memory.clear()
            result["cleared"]["working_memory"] = wm_result
        except Exception as e:
            result["cleared"]["working_memory"] = {"error": str(e)}

        # Clear SweepCache
        if reset_components and reset_components.sweep_cache:
            try:
                sc_result = reset_components.sweep_cache.clear()
                result["cleared"]["sweep_cache"] = sc_result
            except Exception as e:
                result["cleared"]["sweep_cache"] = {"error": str(e)}

        # Clear FrameWindow
        if reset_components and reset_components.frame_window:
            try:
                fw_result = reset_components.frame_window.clear()
                result["cleared"]["frame_window"] = fw_result
            except Exception as e:
                result["cleared"]["frame_window"] = {"error": str(e)}

        # Clear VisualizationServer registry (keyframes)
        if reset_components and reset_components.vis_server:
            try:
                vis = reset_components.vis_server
                if hasattr(vis, 'registry') and vis.registry:
                    kf_cleared = vis.registry.clear()
                    result["cleared"]["visualization"] = {"keyframes_cleared": kf_cleared}
            except Exception as e:
                result["cleared"]["visualization"] = {"error": str(e)}

        return result

    # ---- Detailed stats endpoint ----
    @app.get("/stats/detailed")
    def stats_detailed() -> Dict[str, Any]:
        """
        Get detailed stats from all RTSM components.
        """
        result: Dict[str, Any] = {}

        # WorkingMemory stats
        try:
            result["working_memory"] = dict(working_memory.stats())
        except Exception:
            result["working_memory"] = {}

        # SweepCache stats
        if reset_components and reset_components.sweep_cache:
            try:
                result["sweep_cache"] = reset_components.sweep_cache.stats()
            except Exception:
                result["sweep_cache"] = {}

        # FrameWindow stats
        if reset_components and reset_components.frame_window:
            try:
                result["frame_window"] = reset_components.frame_window.stats()
            except Exception:
                result["frame_window"] = {}

        # VisualizationServer stats
        if reset_components and reset_components.vis_server:
            try:
                vis = reset_components.vis_server
                if hasattr(vis, 'registry') and vis.registry:
                    result["visualization"] = vis.registry.stats()
            except Exception:
                result["visualization"] = {}

        # Extra stats provider
        if extra_stats_provider:
            try:
                result["extra"] = extra_stats_provider()
            except Exception:
                pass

        return result

    # ---- Semantic search endpoint ----
    @app.get("/search/semantic")
    def semantic_search(query: str, top_k: int = 10, threshold: float = 0.2) -> Dict[str, Any]:
        """
        Semantic search for objects using CLIP text encoding + FAISS KNN.

        Args:
            query: Natural language search query (e.g., "red cup", "chair")
            top_k: Maximum number of results to return
            threshold: Minimum cosine similarity threshold (0.0 to 1.0)

        Returns:
            List of matching objects with similarity scores
        """
        if not clip_adapter or not vectors:
            raise HTTPException(status_code=503, detail="Semantic search not available (CLIP or vectors not configured)")

        # 1. Encode query text with CLIP
        try:
            query_emb = clip_adapter.encode_text(query)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to encode query: {e}")

        # 2. KNN search via FAISS
        try:
            matches = vectors.search(query_emb, top_k=top_k)  # [(oid, score), ...]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vector search failed: {e}")

        # 3. Filter by threshold and enrich with WM metadata
        results = []
        for oid, score in matches:
            if score < threshold:
                continue
            obj = working_memory.get(oid)
            results.append({
                "id": oid,
                "score": round(float(score), 4),
                "label_hint": obj.label_primary if obj else None,
                "confirmed": obj.confirmed if obj else True,
                "xyz_world": obj.xyz_world.tolist() if obj and obj.xyz_world is not None else None,
            })

        return {"query": query, "results": results}

    return app


def start_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start a uvicorn server in a background daemon thread and return the thread."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    # Avoid uvicorn installing signal handlers in a child thread
    server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

    def _run() -> None:
        server.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


