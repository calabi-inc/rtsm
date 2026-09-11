# WebSocket API

RTSM provides a WebSocket endpoint for real-time streaming of 3D meshes, object updates, and runtime analytics to the visualization frontend.

**Endpoint**: `ws://localhost:8083/ws`

---

## Connection

```javascript
const ws = new WebSocket('ws://localhost:8083/ws');

ws.onopen = () => {
  console.log('Connected to RTSM visualization');
};

ws.onmessage = (event) => {
  if (event.data instanceof ArrayBuffer) {
    handleBinaryMessage(event.data);
  } else {
    const data = JSON.parse(event.data);
    handleJsonMessage(data);
  }
};
```

!!! note "Binary + JSON protocol"
    The visualization WebSocket sends both **binary messages** (mesh geometry) and **JSON messages** (objects, analytics, commands). Clients must handle both.

---

## Message Types

### Mesh Create (Binary)

Sent when a new TSDF mesh extraction completes. Contains packed binary data:

| Field | Type | Description |
|-------|------|-------------|
| Mesh ID | string | Unique mesh identifier |
| Vertices | float32[] | XYZ positions (N×3) |
| Colors | uint8[] | RGB colors (N×3) |
| Transform | float32[16] | 4×4 transformation matrix |

Meshes replace naive per-frame point clouds via TSDF volumetric fusion. New mesh extractions are triggered every `extract_every_n` frames (default: 30) or `extract_interval_s` seconds (default: 2.0).

---

### Mesh Pose Update (Binary)

Sent when a SLAM loop closure corrects the pose of an existing mesh:

| Field | Type | Description |
|-------|------|-------------|
| Mesh ID | string | Mesh to update |
| Transform | float32[16] | Updated 4×4 pose matrix |

---

### Objects Update (JSON)

Pushed periodically (default: every 200ms) with the current working memory state:

```json
{
  "type": "objects",
  "objects": [
    {
      "id": "a3f2c1d8",
      "xyz_world": [1.2, 0.4, 2.1],
      "label_hint": "backpack",
      "label_scores": {"backpack": 0.87, "bag": 0.45},
      "confirmed": true,
      "stability": 0.82
    }
  ]
}
```

Includes proto (unconfirmed) objects when `visualization.objects.include_proto: true` (default).

Push interval is configurable via `visualization.objects.push_interval_ms`.

---

### Runtime Analytics (JSON)

Pushed every `visualization.analytics.push_interval_ms` (default 1000 ms) with pipeline performance data. The per-second buckets themselves are produced by the analytics ticker (`analytics.enable`; 1 Hz, with or without a client) — the push loop only forwards them, and the first push after a client attaches is always a full sync:

**Full sync** (every 30s by default):

```json
{
  "type": "runtime_analytics",
  "mode": "full",
  "config": { ... },
  "latency": {
    "aggregate": {
      "frame_count": 247,
      "processing_hz": 2.8,
      "t_total": {"mean": 0.362, "p50": 0.340, "p95": 0.612}
    },
    "hourly": [ ... ]
  },
  "segmentation": {
    "backend": "dual",
    "aggregate": { ... },
    "hourly": [ ... ]
  }
}
```

**Incremental append** — one message per new bucket; `bucket` is `null` when no bucket landed in this push interval, and several appends can follow one push when the push interval is slower than the 1 Hz rollup. Buckets carry `elapsed_s` and `stale_interval` (true when the interval exceeded 2 s, a late tick; counts stay exact):

```json
{
  "type": "runtime_analytics",
  "mode": "append",
  "latency": {
    "aggregate": { ... },
    "bucket": { ... }
  },
  "segmentation": {
    "backend": "dual",
    "aggregate": { ... },
    "bucket": { ... }
  }
}
```

---

### Clear (JSON)

Sent to all clients when a clear/reset is triggered:

```json
{
  "type": "clear"
}
```

---

## Client Commands

Send JSON text messages to request data:

### Request Stats

```json
{"cmd": "stats"}
```

Returns a stats snapshot with keyframe count, client count, etc.

### Clear Scene

```json
{"cmd": "clear"}
```

Clears all keyframes, meshes, resets TSDF state, and broadcasts `{"type": "clear"}` to all connected clients.

---

## Configuration

Visualization server settings in `config/rtsm.yaml`:

```yaml
visualization:
  enable: true
  host: 0.0.0.0
  port: 8083

  tsdf:
    enable: true
    voxel_size: 0.01
    extract_every_n: 30
    extract_interval_s: 2.0

  objects:
    push_interval_ms: 200
    include_proto: true

  analytics:
    push_interval_ms: 1000
    full_sync_interval_s: 30
```

---

## Next Steps

- [REST API](rest-api.md) — Query endpoints
- [Configuration](../getting-started/configuration.md) — All settings
