# REST API Reference

RTSM exposes a REST API for querying objects, system state, and runtime analytics.

**Base URL**: `http://localhost:8002`

All list endpoints support **offset-based pagination** via `offset` and `limit` parameters. Responses include `total`, `offset`, `limit`, and `count` fields for pagination metadata.

---

## Health

### Liveness

```http
GET /healthz
```

**Response** (every key besides `status` appears only when its source is wired):

```json
{
  "status": "ok",
  "frame_flow": {"state": "ok", "degraded": false, "reasons": [], "ingest_queue_depth": 0,
                 "backlog": {"lane_full": false, "age_dropped": 0, "depth": {"keyframe": 0, "latest": 0}}},
  "semantic_index": {"confirmed": 62, "indexed": 62, "lag": 0},
  "ingest": {"policy": "lossless", "maxsize": 32, "depth": {"fifo": 0}, "lane_full": false,
             "max_depth_seen": 2, "blocked_s": 0.0, "closed": false,
             "admitted_kf": 9, "admitted_nonkf": 77, "nonkf_superseded": 0, "kf_dropped": 0,
             "kf_lane_full": 0, "age_dropped": 0, "blocked_puts": 0, "closed_puts": 0}
}
```

| Key | Source | Folds into `status` |
|-----|--------|---------------------|
| `frame_flow` | the frame-flow watchdog (live receivers only; off under `--replay`): `state` is one of `waiting`, `ok`, `starved`, `hung`, `receiver_dead`, `backlogged`, `pose_degraded`, `no_ingestible_input`; `backlog` is the lane signal at its last poll | yes, when `degraded` |
| `semantic_index` | the vector store: confirmed objects vs indexed vectors | yes, on upsert / persistence failures or a large lag |
| `ingest` | the ingest lane object itself (`IngestLanes.stats()`, the same snapshot as `/stats.ingest_lanes`) — present live, under replay, in eval and in `rtsm demo`, no flag | **never**: under `lossless` (the replay/eval default) a full lane is a blocking FIFO's steady state, not a fault — read `blocked_s`; live runs keep the watchdog's sustained `backlogged` for that. A raising provider yields `{"error": "unavailable"}` |
| `reasons` | frame-flow reasons first, then semantic-retrieval reasons | present only when `status` is `degraded` |

Live, `lane_full` appears twice: `frame_flow.backlog` is the watchdog's last poll, `ingest` is read time.

### Readiness

```http
GET /readyz
```

**Response**:

```json
{"status": "ready"}
```

---

## Objects

### List Objects

```http
GET /objects
```

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_vectors` | bool | false | Include CLIP embedding vectors |
| `include_snapshot` | bool | false | Include latest observation crop (base64 JPEG) for multimodal agent verification |
| `confirmed_only` | bool | false | Only return confirmed objects |
| `offset` | int | 0 | Skip first N objects (pagination) |
| `limit` | int | 100 | Max objects to return (max 500) |

**Response**:

```json
{
  "total": 62,
  "offset": 0,
  "limit": 100,
  "count": 62,
  "objects": [
    {
      "id": "a3f2c1d8",
      "xyz_world": [1.2, 0.4, 2.1],
      "created_wall_utc": 1712345678.0,
      "stability": 0.82,
      "hits": 15,
      "confirmed": true,
      "view_bins": 3,
      "last_seen_mono": 1712345700.0
    }
  ]
}
```

**With `include_snapshot=true`** (for multimodal agent verification):

```json
{
  "total": 62,
  "offset": 0,
  "limit": 10,
  "count": 10,
  "objects": [
    {
      "id": "a3f2c1d8",
      "xyz_world": [1.2, 0.4, 2.1],
      "stability": 0.82,
      "confirmed": true,
      "snapshot_b64": "/9j/4AAQ...",
      "snapshot_count": 4
    }
  ]
}
```

**Pagination example**:

```bash
# First page
curl "http://localhost:8002/objects?limit=20"
# Next page
curl "http://localhost:8002/objects?limit=20&offset=20"
```

---

### Get Object by ID

```http
GET /objects/{oid}
```

**Query Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_vectors` | bool | false | Include embedding vectors |

---

### Object Debug Info

```http
GET /objects/{oid}/debug
```

Returns detailed diagnostic information: position with covariance, tracking state, view diversity, gallery size, and timestamps.

---

### Object Snapshots

```http
GET /objects/{oid}/snapshots
```

Returns base64-encoded JPEG image crops (most recent first). These are 224x224 RGB crops of the object from each observation — useful for multimodal LLM verification.

```http
GET /objects/{oid}/snapshots/{index}/image
```

Returns raw JPEG image for a specific snapshot index.

---

## Search

### Semantic Search

```http
GET /search/semantic?query={text}
```

Encodes the query text with CLIP and performs KNN search against all object embeddings.

> **Note on CLIP scores**: CLIP ViT-B/32 raw cosine similarities for indoor objects typically cluster in the 0.25-0.35 range. The **ranking is meaningful** (top results are most relevant) even though absolute scores appear low. For visual verification, use `include_snapshot=true` and pass crops to a multimodal LLM.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | Natural language query |
| `top_k` | int | 10 | Maximum results |
| `threshold` | float | 0.0 | Minimum cosine similarity (default: return all ranked) |
| `include_snapshot` | bool | false | Include latest crop as base64 JPEG |

**Example**:

```bash
# Basic search
curl "http://localhost:8002/search/semantic?query=coffee+mug&top_k=5"

# With snapshots for multimodal verification
curl "http://localhost:8002/search/semantic?query=coffee+mug&top_k=3&include_snapshot=true"
```

**Response**:

```json
{
  "query": "coffee mug",
  "robot_pose": {
    "xyz": [0.12, 0.05, 0.31],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    "timestamp": 1712345678.5,
    "frame_epoch": 0,
    "sensor_ts_ns": 683333172055083,
    "pose_clock": "sender",
    "age_s": 0.041,
    "stale": false,
    "stale_after_s": 0.5,
    "writes_accepted": 812,
    "sensor_ts_regressions": 0,
    "rejected_writes": 0,
    "rejected_by_reason": {"older_epoch": 0, "epochless_into_epoch": 0}
  },
  "results": [
    {
      "id": "b7d4e2f1",
      "score": 0.31,
      "confirmed": true,
      "stability": 0.82,
      "xyz_world": [0.8, 0.2, 1.5],
      "snapshot_b64": "/9j/4AAQ...",
      "snapshot_count": 3
    }
  ]
}
```

> **`robot_pose`**: All search responses include the robot's latest pose so agents can compute heading and distance to target objects in one atomic query. RTSM stores but does not compute pose — it's a passthrough from the sensor, written **at receive time by the receiver** (websocket, replay, ZeroMQ) for every tracking-normal frame at input rate; the pipeline does not write it. The store is a latest-value mailbox keyed on `(frame_epoch, sensor_ts_ns)`: `frame_epoch` bumps when the sender starts a new streaming session (websocket: a new `session_id`; ZeroMQ: the tracking stamp jumping back by more than 5 s), and poses across a bump do not share a world frame. `sensor_ts_ns` is the sender's monotonic sensor stamp; `timestamp` is a wall clock for display, the sender's (`pose_clock: "sender"`) or this server's (`"server"`; `null` before any tag is known), and is not guaranteed monotone. `age_s` is the time since the last accepted write on the server's clock and `stale` is `age_s > robot_pose.stale_after_s` — **diagnostic only**: an agent must measure freshness on its own clock (the RC-car agent uses `stale_abort_s`) and treat a non-advancing `timestamp` as a frozen feed. `sensor_ts_regressions` counts accepted writes whose stamp went backwards within one epoch (a looped replay, a bag loop, a stepped sender clock — the pose follows the sender); `rejected_writes` counts writes from a previous epoch or from an epoch-less writer, which only a second writer can produce.

**Agent workflow**: Query RTSM for candidates with snapshots, then pass crops to Gemini/GPT-4V for visual verification:

```python
import requests, base64

# 1. Get candidates with snapshots
resp = requests.get("http://localhost:8002/search/semantic",
    params={"query": "coffee mug", "top_k": 3, "include_snapshot": "true"})
candidates = resp.json()["results"]

# 2. Pass to multimodal LLM for verification
for c in candidates:
    image_data = base64.b64decode(c["snapshot_b64"])
    # Send to Gemini/GPT-4V: "Is this a coffee mug?"
    # If yes -> use c["xyz_world"] for navigation
```

### Spatial Search

```http
GET /search/spatial?x={x}&y={y}&z={z}&radius_m={r}
```

Finds objects within a radius of a 3D point in world coordinates.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | float | required | X coordinate (meters, world frame) |
| `y` | float | required | Y coordinate (meters, world frame) |
| `z` | float | required | Z coordinate (meters, world frame) |
| `radius_m` | float | 1.0 | Search radius in meters |
| `offset` | int | 0 | Skip first N results (pagination) |
| `limit` | int | 50 | Max results (max 200) |

**Example**:

```bash
curl "http://localhost:8002/search/spatial?x=1.0&y=0.5&z=2.0&radius_m=0.5"
```

**Response**:

```json
{
  "center": [1.0, 0.5, 2.0],
  "radius_m": 0.5,
  "total": 8,
  "offset": 0,
  "limit": 50,
  "count": 8,
  "results": [
    {
      "id": "b7d4e2f1",
      "distance_m": 0.23,
      "xyz_world": [1.1, 0.4, 2.1],
      "confirmed": true,
      "stability": 0.82
    }
  ]
}
```

Results are sorted by distance (nearest first). Returns `503` if the proximity index is not available.

---

## Statistics

### Basic Stats

```http
GET /stats
```

**Response**:

```json
{
  "objects": 111,
  "confirmed": 62,
  "avg_hits": 4.2,
  "upserts_total": 28,
  "ingest_q": 0,
  "ingest_lanes": {"policy": "latest", "maxsize": 4, "depth": {"keyframe": 0, "latest": 0}, "lane_full": false,
                   "nonkf_superseded": 0, "kf_dropped": 0, "age_dropped": 0, "blocked_puts": 0}
}
```

`ingest_q` is the number of frames waiting for the pipeline. Under
`ingest.policy: latest` (the live default) that is the keyframe lane plus the
one non-keyframe slot, so it never exceeds `keyframe_lane_depth + 1`; under
`lossless` / `legacy` it is the FIFO depth. `ingest_lanes` carries the policy,
per-lane depth and the lane counters (see the configuration guide).

### Detailed Stats

```http
GET /stats/detailed
```

Returns stats from all components: working memory, sweep cache, frame window, visualization registry.

---

### Runtime Analytics

```http
GET /stats/analytics
```

Returns real-time pipeline analytics with per-second time-series history (up to 1 hour):

```json
{
  "latency": {
    "aggregate": {"frame_count": 53, "input_hz": 5.4, "processing_hz": 1.16, "effective_ratio": 0.215,
                  "counters": {"received": 240, "processed": 53, "gate_rejections": 33, "throttle_skips": 154, "...": 0},
                  "...": {}},
    "hourly": [{"wall_ts": 1789120519.4, "input_hz": 5.0, "frames_received": 5, "processing_hz": 1.0, "frames_in_bucket": 1,
                "elapsed_s": 1.0, "stale_interval": false, "wm_total": 124, "wm_confirmed": 65, "...": 0}]
  },
  "segmentation": {"aggregate": {"...": 0}, "hourly": [{"...": 0}]},
  "rollup": {"interval_s": 1.0, "ticks": 84, "late_ticks": 0, "stale_rollups": 0, "ring_truncated": 0,
             "last_tick_age_s": 0.4, "stalled": false, "alive": true}
}
```

The per-second `hourly` buckets are produced by the **analytics ticker**, one daemon thread per process that runs whenever `analytics.enable` is true — with or without a visualization client — so headless runs (`--replay`, `rtsm demo --no-viz`, the eval harness) carry the same history and real `input_hz` / `effective_ratio`. Each bucket covers `elapsed_s` seconds (1.0 at the ticker's cadence); `stale_interval` is true when the interval exceeded 2 s (a late tick), in which case the bucket is wide, not empty — counts stay exact: every processed frame, received frame and drop is in exactly one bucket, so the per-bucket sums equal the lifetime `counters` (the first bucket after start also carries whatever the receiver recorded before the ticker started). `rollup` is the ticker's own health: `late_ticks` counts tick-to-tick gaps above 2 s (a 1 Hz thread that could not get scheduled), `stale_rollups` the buckets the buffers flagged, `ring_truncated` frames whose timings left the Tier-1 ring before a rollup, `stalled` the read-time view (no tick for more than 2 s — the counters only move when a tick eventually happens, so a ticker wedged behind a lock shows `stalled: true` with `late_ticks 0`); it is `null` when no ticker is wired, and none of it is reset by `POST /reset`.

Returns `503` if analytics is disabled (`analytics.enable: false` in config).

---

## System Control

### Reset

```http
POST /reset
```

Clears working memory, sweep cache, frame window, visualization registry, TSDF state, and analytics buffers. Models stay loaded.

---

## Prometheus Metrics

```http
GET /metrics
```

Exposes Prometheus-format metrics for external monitoring:

| Metric | Type | Description |
|--------|------|-------------|
| `rtsm_working_objects` | gauge | Total objects in working memory |
| `rtsm_confirmed_objects` | gauge | Confirmed objects |
| `rtsm_upserts_total` | gauge | Total LTM upserts emitted |

---

## MCP (Model Context Protocol)

When `mcp.enable: true` in config, RTSM mounts an MCP server on the API:

```
SSE endpoint: http://localhost:8002/mcp/sse
```

This exposes RTSM's spatial memory as tools for AI agents (Claude, GPT, local LLMs). See [MCP API](mcp.md) for tool details.

---

## Error Responses

```json
{
  "detail": "Object a3f2c1d8 not found"
}
```

| Status | Meaning |
|--------|---------|
| 404 | Object not found |
| 503 | Service unavailable (analytics disabled, search not configured) |
| 500 | Internal server error |

---

## Next Steps

- [WebSocket API](websocket.md) -- Real-time mesh, object, and analytics streaming
- [Quick Start](../getting-started/quick-start.md) -- Try these endpoints
