# RTSM Demo — Real-Time Spatio-Semantic Memory Visualization

Interactive 3D visualization for RTSM featuring:
- Real-time point cloud streaming from SLAM keyframes
- RTSM semantic objects overlay with labels
- Static PLY file loading
- Export accumulated point clouds

## Architecture

```
┌──────────────┐    ZMQ     ┌─────────────────────┐    WS     ┌─────────────┐
│  ORB-SLAM3   │ ────────▶  │   Python Demo       │ ───────▶  │   Browser   │
│  / Sensors   │            │   Server            │  binary   │   Client    │
└──────────────┘            │  - Point cloud gen  │           │  (render    │
                            │  - Pose management  │           │   only)     │
                            │  - Client sync      │           └─────────────┘
                            └─────────────────────┘                  │
                                                                     ▼
┌──────────────┐   HTTP    ┌─────────────────────────────────────────────────┐
│  RTSM API    │ ◀──────── │  Three.js Viewer + Objects Overlay             │
│  (FastAPI)   │           │  - Receives pre-computed meshes                 │
└──────────────┘           │  - Applies pose updates                         │
                           │  - Semantic labels                              │
                           │  - PLY export                                   │
                           └─────────────────────────────────────────────────┘
```

**Key Design**: Backend is source of truth for mesh management. Frontend is render-only.

- Backend receives RGB-D + intrinsics from ZMQ
- Backend computes point clouds (JPEG decode, depth filter, backprojection)
- Backend sends binary mesh data to clients (~50-75KB vs 600KB raw)
- Backend tracks all keyframes and syncs new clients
- Frontend only renders what backend tells it to

## Quick Start

### 1. Install Dependencies

```bash
# Client (Vite + Three.js)
cd demo
npm install

# Demo server (Python)
cd demo/server
pip install -r requirements.txt
```

### 2. Start Services

**Terminal 1 — RTSM Backend (optional, for object overlay)**
```bash
# From project root
python -m rtsm.run
# or: rtsm-run
# API available at http://localhost:8000
```

**Terminal 2 — Demo Server (for point cloud streaming)**
```bash
cd demo/server
python main.py
# WebSocket available at ws://localhost:8081/ws
```

**Terminal 3 — Web Client**
```bash
cd demo
npm run dev
# Open http://localhost:5173
```

### 3. Data Sources

**Option A: Live Camera + SLAM**
- Run ORB-SLAM3 publishing to ZMQ at `tcp://127.0.0.1:6001`
- Run D435i streaming: `python d435i/zmq_streaming.py`

**Option B: Dataset Replay**
```bash
python scripts/replay_dataset.py \
  --dataset test_dataset \
  --zmq \
  --zmq-endpoint tcp://127.0.0.1:6001
```

**Option C: Static PLY Only**
- Just run the web client
- Load a PLY file via the UI

## Features

### Real-Time Point Cloud Streaming
- Backend receives RGB-D keyframes via ZMQ
- Decodes JPEG + 16-bit depth on server
- Applies depth filtering (median + jump rejection)
- Back-projects to 3D with camera intrinsics
- Sends pre-computed binary meshes to clients
- Applies pose transforms from SLAM
- Updates poses on corrections (loop closure / BA)
- New clients receive all existing keyframes on connect

### RTSM Objects Overlay
- Polls `/api/objects` every 500ms
- Displays spheres at object positions
- Color-coded: green = confirmed, orange = proto
- Floating labels with semantic names

### Interactive Controls
| Control | Action |
|---------|--------|
| **Drag** | Orbit camera |
| **Scroll** | Zoom |
| **Double-click** | Set orbit focus point |
| **Z + Drag** | Pitch-only rotation |
| **X + Drag** | Yaw-only rotation |
| **C + Drag** | Roll rotation |

### UI Buttons
- **PLY**: Load static point cloud file
- **Stream**: Clear streamed data / Export to PLY
- **Objects**: Toggle semantic overlays
- **View**: Reset camera, flip axes
- **Rebuild**: Reconnect and re-sync all keyframes from server

## Configuration

### Demo Server Environment Variables
```bash
DEMO_WS_PORT=8081                     # WebSocket port (default: 8081)
ZMQ_ENDPOINT=tcp://127.0.0.1:6001     # ZMQ endpoint to subscribe
```

### Vite Proxy
The dev server proxies:
- `/api/*` → `http://localhost:8000` (RTSM backend)
- `/ws` → `ws://localhost:8081` (Demo server WebSocket)

## Binary Message Protocol

### mesh_create (binary)
```
[magic:4 'PCLD'][mesh_id_len:2][mesh_id:N][num_points:4]
[positions:N*12][colors:N*3][has_pose:1][pose:64?]
```
- positions: float32 LE, xyz interleaved
- colors: uint8 RGB
- pose: float32 LE, 4x4 row-major Twc

### mesh_update_pose (JSON)
```json
{"type": "mesh_update_pose", "mesh_id": "123", "pose": [16 floats]}
```

### mesh_delete (JSON)
```json
{"type": "mesh_delete", "mesh_id": "123"}
```

### clear (JSON)
```json
{"type": "clear"}
```

## Build for Production

```bash
cd demo
npm run build
# Output: demo/dist/
```

Serve with any static file server. The demo server is still required for real-time streaming.

## Keyboard Shortcuts

- `Z` — Hold for pitch-only rotation
- `X` — Hold for yaw-only rotation
- `C` — Hold for roll rotation
- `Escape` — Release rotation lock

## Troubleshooting

**No point cloud appearing?**
- Check demo server is running (`ws://localhost:8081/ws`)
- Check ZMQ publisher is sending data
- Open browser console for errors

**Objects not showing?**
- Ensure RTSM backend is running at `http://localhost:8000`
- Check `/api/objects` returns data
- Toggle objects with the UI button

**WebSocket disconnects?**
- Client auto-reconnects every 2s
- Check ZMQ endpoint configuration matches publisher
- Check demo server logs for errors

**New client not receiving existing keyframes?**
- Server syncs all keyframes on connect automatically
- Check server logs for sync count
