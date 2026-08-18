from __future__ import annotations

import asyncio
import time
import threading
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass

import base64
import numpy as np
from fastapi import FastAPI, Response, HTTPException, WebSocket, WebSocketDisconnect
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
    seg_analytics: Optional[Any] = None,
    latency_analytics: Optional[Any] = None,
    mcp_enabled: bool = False,
    vis_server: Optional[Any] = None,
    vis_broadcaster: Optional[Any] = None,
    vis_registry: Optional[Any] = None,
    static_dir: Optional[str] = None,
    frame_flow_provider: Optional[Callable[[], Dict[str, Any]]] = None,
) -> FastAPI:
    """
    Build a FastAPI app exposing:
      - /healthz: liveness
      - /readyz: readiness (trivial true for now)
      - /stats: JSON snapshot (WorkingMemory.stats() + optional extra stats)
      - /metrics: Prometheus metrics (mounted ASGI app)
      - /ws: WebSocket for visualization (when vis_server is provided)
      - /: Static frontend (when static_dir is provided)

    When vis_server is provided, its periodic tasks (objects push, analytics)
    run on this server's event loop — no separate viz server port needed.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Start visualization periodic tasks on THIS event loop
        if vis_server is not None:
            await vis_server.start_tasks()
        yield
        if vis_server is not None:
            await vis_server.stop_tasks()

    app = FastAPI(
        title="RTSM API — Real-Time Spatio-Semantic Memory",
        version="1.0.0",
        lifespan=lifespan,
    )

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
    def healthz() -> Dict[str, Any]:
        # Additive shape: "status" stays "ok"/"degraded"; "frame_flow" and
        # "reasons" appear only when a watchdog is wired (see
        # rtsm/core/watchdog.py). Consumers reading only "status" are
        # unaffected.
        out: Dict[str, Any] = {"status": "ok"}
        if frame_flow_provider is not None:
            try:
                ff = frame_flow_provider() or {}
                out["frame_flow"] = ff
                if ff.get("degraded"):
                    out["status"] = "degraded"
                    out["reasons"] = list(ff.get("reasons", []))
            except Exception:
                # Health endpoint must never fail because the watchdog did.
                out["frame_flow"] = {"state": "unknown"}
        return out

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
    def list_objects(
        include_vectors: bool = False,
        include_snapshot: bool = False,
        confirmed_only: bool = False,
        offset: int = 0,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """List objects in working memory with pagination.

        Args:
            include_vectors: Include CLIP embedding vectors in response
            include_snapshot: Include latest observation crop (base64 JPEG)
                for multimodal agent verification
            confirmed_only: If true, only return confirmed objects
            offset: Skip first N objects (for pagination)
            limit: Maximum objects to return (default 100, max 500)
        """
        limit = min(max(1, limit), 500)
        offset = max(0, offset)

        try:
            objs: List[Any] = working_memory.iter_objects()
        except Exception:
            objs = []

        if confirmed_only:
            objs = [o for o in objs if getattr(o, 'confirmed', False)]

        total = len(objs)
        page = objs[offset : offset + limit]

        result_list = []
        for o in page:
            entry = _obj_detail(o, include_vectors=include_vectors) if include_vectors else _obj_summary(o)
            if include_snapshot:
                crops = getattr(o, 'image_crops', None) or []
                if crops:
                    entry["snapshot_b64"] = base64.b64encode(crops[-1]).decode('ascii')
                    entry["snapshot_count"] = len(crops)
            result_list.append(entry)
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "count": len(result_list),
            "objects": result_list,
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
        - Vector store (FAISS LTM index — stale ids otherwise surface as
          ghost objects in /search/semantic after reset)
        - SweepCache (sweep timestamps, camera snapshots)
        - FrameWindow (buffered RGB-D frames)
        - VisualizationServer registry (keyframes/point clouds)

        Does NOT clear:
        - FastSAM / CLIP models (expensive to reload)
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

        # Clear vector store — WM.clear() only clears the spatial index;
        # the FAISS index is a separate store and must be reset with it.
        if vectors is not None:
            try:
                if hasattr(vectors, "clear"):
                    result["cleared"]["vector_store"] = vectors.clear()
                else:
                    result["cleared"]["vector_store"] = {"error": "vector store has no clear()"}
            except Exception as e:
                result["cleared"]["vector_store"] = {"error": str(e)}

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

        # Clear VisualizationServer (registry + TSDF + broadcast clear to clients)
        if reset_components and reset_components.vis_server:
            try:
                vis = reset_components.vis_server
                vis_result = {}
                if hasattr(vis, 'registry') and vis.registry:
                    kf_cleared = vis.registry.clear()
                    vis_result["keyframes_cleared"] = kf_cleared
                if hasattr(vis, 'tsdf') and vis.tsdf is not None:
                    vis.tsdf.reset()
                    vis_result["tsdf_reset"] = True
                # Broadcast clear to all connected web clients
                if hasattr(vis, 'broadcaster') and vis._running:
                    vis.broadcaster.schedule(
                        vis.broadcaster._broadcast_json({"type": "clear"})
                    )
                    vis_result["clients_notified"] = True
                result["cleared"]["visualization"] = vis_result
            except Exception as e:
                result["cleared"]["visualization"] = {"error": str(e)}

        # Clear analytics buffers
        if seg_analytics:
            try:
                seg_analytics.clear()
                result["cleared"]["seg_analytics"] = True
            except Exception as e:
                result["cleared"]["seg_analytics"] = {"error": str(e)}
        if latency_analytics:
            try:
                latency_analytics.clear()
                result["cleared"]["latency_analytics"] = True
            except Exception as e:
                result["cleared"]["latency_analytics"] = {"error": str(e)}

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
    def semantic_search(
        query: str,
        top_k: int = 10,
        threshold: float = 0.0,
        include_snapshot: bool = False,
    ) -> Dict[str, Any]:
        """
        Semantic search for objects using CLIP text encoding + FAISS KNN.

        Cosine scores vary by model: CLIP ViT-B/32 clusters 0.25-0.35,
        SigLIP ViT-B-16 clusters 0.05-0.15 for indoor objects. The ranking
        is meaningful (top results are most relevant) even though absolute
        scores are low. Default threshold=0.0 returns all ranked results
        so agents can decide their own cutoff.

        For visual verification, set include_snapshot=true to get the most
        recent observation crop (base64 JPEG) for each result. This enables
        multimodal LLM planners to visually verify objects without relying
        on CLIP classification.

        Args:
            query: Natural language search query (e.g., "red cup", "chair")
            top_k: Maximum number of results to return
            threshold: Minimum cosine similarity threshold (default 0.0 = return all ranked)
            include_snapshot: If true, include base64 JPEG of most recent crop
        """
        if not clip_adapter or not vectors:
            raise HTTPException(status_code=503, detail="Semantic search not available (CLIP or vectors not configured)")

        # 1. Encode query text with CLIP
        # For OpenAI CLIP models, wrap short queries in caption format
        # ("a photo of a dog") since CLIP was trained on image-caption pairs.
        # SigLIP models work better with raw queries (trained differently).
        clip_query = query
        if hasattr(clip_adapter, '_prompt_wrap') and clip_adapter._prompt_wrap and len(query.split()) <= 3:
            clip_query = f"a photo of a {query}"
        try:
            query_emb = clip_adapter.encode_text(clip_query)
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
            entry: Dict[str, Any] = {
                "id": oid,
                "score": round(float(score), 4),
                # A vector-store id with no WM object is stale (e.g. survived
                # a partial clear) — never advertise it as confirmed.
                "confirmed": obj.confirmed if obj else False,
                "stability": round(float(obj.stability), 3) if obj else 0.0,
                "xyz_world": obj.xyz_world.tolist() if obj and obj.xyz_world is not None else None,
            }

            # Include most recent snapshot for multimodal agent verification
            if include_snapshot and obj:
                crops = getattr(obj, 'image_crops', None) or []
                if crops:
                    # Most recent crop is last in list
                    entry["snapshot_b64"] = base64.b64encode(crops[-1]).decode('ascii')
                    entry["snapshot_count"] = len(crops)

            results.append(entry)

        return {
            "query": query,
            "robot_pose": working_memory.get_robot_pose(),
            "results": results,
        }

    # ---- Spatial search endpoint ----
    @app.get("/search/spatial")
    def spatial_search(
        x: float, y: float, z: float,
        radius_m: float = 1.0,
        offset: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Spatial search for objects within a radius of a 3D point.

        Args:
            x, y, z: Center point in world coordinates (meters)
            radius_m: Search radius in meters (default 1.0)
            offset: Skip first N results (for pagination)
            limit: Maximum results to return (default 50, max 200)

        Returns:
            List of nearby objects sorted by distance, with pagination
        """
        limit = min(max(1, limit), 200)
        offset = max(0, offset)

        if working_memory.index is None:
            raise HTTPException(status_code=503, detail="Spatial search not available (no proximity index)")

        center = np.array([x, y, z], dtype=np.float32)
        grid = working_memory.index.grid
        rings = min(10, max(1, int(np.ceil(radius_m / grid.cell_m))))

        oids = working_memory.index.nearby_ids(center, rings=rings)

        all_results = []
        for oid in oids:
            obj = working_memory.get(oid)
            if obj is None:
                continue
            dist = float(np.linalg.norm(obj.xyz_world - center))
            if dist > radius_m:
                continue
            all_results.append({
                "id": oid,
                "distance_m": round(dist, 4),
                "xyz_world": obj.xyz_world.tolist(),
                "confirmed": bool(getattr(obj, "confirmed", False)),
                "stability": round(float(getattr(obj, "stability", 0.0)), 3),
            })

        all_results.sort(key=lambda r: r["distance_m"])
        total = len(all_results)
        page = all_results[offset : offset + limit]

        return {
            "center": [x, y, z],
            "radius_m": radius_m,
            "robot_pose": working_memory.get_robot_pose(),
            "total": total,
            "offset": offset,
            "limit": limit,
            "count": len(page),
            "results": page,
        }

    # ---- Analytics endpoint ----
    @app.get("/stats/analytics")
    def stats_analytics() -> Dict[str, Any]:
        """Get runtime analytics (segmentation breakdown + latency/throughput)."""
        if not seg_analytics and not latency_analytics:
            raise HTTPException(status_code=503, detail="Analytics not enabled")
        result: Dict[str, Any] = {}
        if latency_analytics:
            result["latency"] = {
                "aggregate": latency_analytics.aggregate(),
                "hourly": latency_analytics.hourly_history(),
            }
        if seg_analytics:
            result["segmentation"] = {
                "aggregate": seg_analytics.aggregate(),
                "hourly": seg_analytics.hourly_history(),
            }
        return result

    # ---- Embedded MCP server (optional) ----
    if mcp_enabled:
        try:
            from rtsm.io.mcp_embedded import create_mcp_app
            mcp_mount = create_mcp_app(
                working_memory=working_memory,
                clip_adapter=clip_adapter,
                vectors=vectors,
            )
            app.mount("/mcp", mcp_mount)
        except ImportError:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "MCP enabled in config but 'mcp' package not installed. "
                "Install with: pip install \"rtsm[mcp]\""
            )

    # ---- Visualization WebSocket (optional, for single-port demo) ----
    if vis_broadcaster is not None and vis_registry is not None:

        @app.websocket("/ws")
        async def viz_websocket(websocket: WebSocket):
            await websocket.accept()
            await vis_broadcaster.connect(websocket)
            synced = await vis_broadcaster.sync_new_client(websocket, vis_registry)
            # Sync latest TSDF mesh to new client
            if vis_server is not None and hasattr(vis_server, 'tsdf') and vis_server.tsdf is not None:
                latest = vis_server.tsdf.get_latest_mesh()
                if latest is not None:
                    import numpy as _np
                    positions, colors = latest
                    identity = _np.eye(4, dtype=_np.float32)
                    data = vis_broadcaster._pack_mesh_create(
                        "tsdf_fused", positions, colors, identity
                    )
                    await vis_broadcaster._try_send_bytes(websocket, data)
                    synced += 1
            import logging as _log
            _log.getLogger(__name__).info(f"[api/ws] Client connected, synced {synced} keyframes")
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle client commands (clear, stats)
                    try:
                        import json as _json
                        msg = _json.loads(data)
                        cmd = msg.get("cmd")
                        if cmd == "clear":
                            vis_registry.clear()
                            await vis_broadcaster._broadcast_json({"type": "clear"})
                    except Exception:
                        pass
            except WebSocketDisconnect:
                pass
            finally:
                await vis_broadcaster.disconnect(websocket)

    # ---- Static frontend (mount LAST so API routes take priority) ----
    if static_dir:
        import os
        if os.path.isdir(static_dir):
            from fastapi.staticfiles import StaticFiles
            app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


def start_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start a uvicorn server in a background daemon thread.

    Blocks until the server is listening and the lifespan startup has
    completed (so vis_server.start_tasks() has run before we return).
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    # Avoid uvicorn installing signal handlers in a child thread
    server.install_signal_handlers = lambda: None  # type: ignore[attr-defined]

    ready = threading.Event()
    _orig_startup = server.startup

    async def _startup_then_signal(*a, **kw):
        result = await _orig_startup(*a, **kw)
        ready.set()
        return result

    server.startup = _startup_then_signal  # type: ignore[attr-defined]

    def _run() -> None:
        server.run()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    ready.wait(timeout=30)  # block until lifespan completes
    return t


