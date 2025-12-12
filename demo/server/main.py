"""
RTSM Demo Server - FastAPI Entry Point

Standalone Python server that:
- Subscribes to ZMQ for RTABMap SLAM data
- Processes RGB-D into point clouds
- Broadcasts pre-computed meshes to WebSocket clients
- Serves as source of truth for mesh management

Run:
    python main.py
    # or: uvicorn main:app --host 0.0.0.0 --port 8081 --reload
"""

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from processor import PointCloudProcessor, ProcessorConfig
from registry import KeyframeRegistry
from broadcaster import WSBroadcaster
from zmq_listener import ZMQListener


# Configuration from environment
ZMQ_ENDPOINT = os.environ.get("ZMQ_ENDPOINT", "tcp://127.0.0.1:6000")
WS_PORT = int(os.environ.get("DEMO_WS_PORT", "8081"))

# Global instances
processor: Optional[PointCloudProcessor] = None
registry: Optional[KeyframeRegistry] = None
broadcaster: Optional[WSBroadcaster] = None
zmq_listener: Optional[ZMQListener] = None
zmq_task: Optional[asyncio.Task] = None


async def handle_kf_packet(
    map_id: str,
    kf_id: str,
    timestamp_ns: int,
    K: np.ndarray,
    jpeg_bytes: bytes,
    depth_png_bytes: bytes,
    depth_scale: float,
    width: int,
    height: int,
    pose: Optional[np.ndarray]
) -> None:
    """
    Handle incoming keyframe packet from ZMQ (rtabmap.kf_packet).

    1. Process RGB-D into point cloud
    2. Store in registry
    3. Broadcast mesh_create to all clients
    """
    global processor, registry, broadcaster

    if not processor or not registry or not broadcaster:
        return

    try:
        # Process RGB-D into point cloud
        positions, colors = processor.process(
            jpeg_bytes=jpeg_bytes,
            depth_bytes=depth_png_bytes,
            K=K,
            width=width,
            height=height,
            depth_scale=depth_scale,
            depth_encoding='png'
        )

        if positions.shape[0] == 0:
            print(f"[main] Keyframe {kf_id} produced 0 points, skipping")
            return

        # Use kf_id as mesh_id (backend assigns ID)
        mesh_id = kf_id

        # Check if we already have a pose for this keyframe (from earlier kf_pose message)
        existing = registry.get(mesh_id)
        if pose is None and existing is not None:
            pose = existing.pose

        # Register/update in registry
        registry.register(
            mesh_id=mesh_id,
            timestamp_ns=timestamp_ns,
            positions=positions,
            colors=colors,
            pose=pose,
            map_id=map_id
        )

        # Broadcast to all connected clients
        await broadcaster.send_mesh_create(mesh_id, positions, colors, pose)

        stats = registry.stats()
        print(f"[main] KF {mesh_id}: {positions.shape[0]:,} pts | Total: {stats['keyframes']} KFs, {stats['total_points']:,} pts")

    except Exception as e:
        print(f"[main] Error processing kf_packet {kf_id}: {e}")
        import traceback
        traceback.print_exc()


async def handle_kf_pose(
    kf_id: str,
    pose: np.ndarray
) -> None:
    """
    Handle initial pose for a keyframe (rtabmap.kf_pose).

    If keyframe exists, update pose and broadcast.
    Otherwise, store pose for when keyframe arrives.
    """
    global registry, broadcaster

    if not registry or not broadcaster:
        return

    mesh_id = kf_id

    if registry.exists(mesh_id):
        # Keyframe already exists, update pose
        registry.update_pose(mesh_id, pose)
        await broadcaster.send_mesh_update_pose(mesh_id, pose)
        print(f"[main] Pose set for existing KF {mesh_id}")
    else:
        # Keyframe not yet received - create placeholder
        # This handles the case where pose arrives before RGB-D data
        registry.register(
            mesh_id=mesh_id,
            timestamp_ns=0,
            positions=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.uint8),
            pose=pose,
            map_id="0"
        )
        print(f"[main] Pose cached for pending KF {mesh_id}")


async def handle_kf_pose_update(
    kf_id: str,
    pose: np.ndarray
) -> None:
    """
    Handle pose correction from bundle adjustment or loop closure (rtabmap.kf_pose_update).

    Only updates if keyframe exists.
    """
    global registry, broadcaster

    if not registry or not broadcaster:
        return

    mesh_id = kf_id

    if registry.update_pose(mesh_id, pose):
        await broadcaster.send_mesh_update_pose(mesh_id, pose)
        print(f"[main] Pose updated for KF {mesh_id} (BA/LC)")
    else:
        print(f"[main] Pose update for unknown KF {mesh_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    global processor, registry, broadcaster, zmq_listener, zmq_task

    # Initialize components
    processor = PointCloudProcessor(ProcessorConfig())
    registry = KeyframeRegistry()
    broadcaster = WSBroadcaster()

    # Initialize ZMQ listener with callbacks
    zmq_listener = ZMQListener(
        endpoint=ZMQ_ENDPOINT,
        on_kf_packet=handle_kf_packet,
        on_kf_pose=handle_kf_pose,
        on_kf_pose_update=handle_kf_pose_update,
    )

    # Start ZMQ listener
    await zmq_listener.start()
    zmq_task = asyncio.create_task(zmq_listener.run())

    print(f"[main] RTSM Demo Server started on port {WS_PORT}")
    print(f"[main] ZMQ endpoint: {ZMQ_ENDPOINT}")

    yield

    # Shutdown
    print("[main] Shutting down...")
    if zmq_listener:
        await zmq_listener.stop()
    if zmq_task:
        zmq_task.cancel()
        try:
            await zmq_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="RTSM Demo Server",
    description="Real-Time Spatio-Semantic Memory Demo - Point Cloud Streaming",
    lifespan=lifespan
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for mesh streaming.

    On connect:
    1. Accept connection
    2. Sync all existing keyframes
    3. Receive real-time updates

    Messages:
    - Binary: mesh_create (new point cloud)
    - JSON: mesh_update_pose, mesh_delete
    """
    global broadcaster, registry

    await websocket.accept()
    await broadcaster.connect(websocket)

    # Sync existing keyframes to new client
    if registry:
        synced = await broadcaster.sync_new_client(websocket, registry)
        print(f"[main] New client synced with {synced} keyframes")

    try:
        # Keep connection alive, handle any client messages
        while True:
            # Client can send commands (e.g., clear, export)
            try:
                data = await websocket.receive_json()
                await handle_client_command(websocket, data)
            except Exception:
                # Connection closed or invalid message
                break
    except WebSocketDisconnect:
        pass
    finally:
        await broadcaster.disconnect(websocket)
        print(f"[main] Client disconnected ({broadcaster.client_count} remaining)")


async def handle_client_command(websocket: WebSocket, data: dict) -> None:
    """Handle commands from WebSocket clients."""
    global registry, broadcaster

    cmd = data.get("cmd")

    if cmd == "clear":
        # Clear all keyframes
        if registry:
            count = registry.clear()
            # Broadcast clear to all clients
            await broadcaster._broadcast_json({"type": "clear"})
            print(f"[main] Cleared {count} keyframes")

    elif cmd == "stats":
        # Send stats to requesting client
        if registry:
            stats = registry.stats()
            stats["clients"] = broadcaster.client_count
            await websocket.send_json({"type": "stats", **stats})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    """Get current statistics."""
    global registry, broadcaster

    if not registry:
        return {"error": "not initialized"}

    return {
        **registry.stats(),
        "clients": broadcaster.client_count if broadcaster else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=WS_PORT)
