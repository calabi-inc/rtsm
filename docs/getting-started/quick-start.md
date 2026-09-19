# Quick Start

This guide walks you through running RTSM and making your first semantic query.

---

## 1. Start RTSM

Start the main service:

```bash
python -m rtsm          # headless: REST API (+ MCP)
python -m rtsm --viz    # plus the 3D dashboard (opens the browser)
```

This launches:

| Service | Address |
|---------|---------|
| REST API | `http://localhost:8002` |
| WebSocket (visualization) | `ws://localhost:8002/ws`, only with `--viz` (headless by default) |
| MCP (if enabled) | `http://localhost:8002/mcp/sse` |

RTSM listens for RGB-D frames via the configured receiver (WebSocket from Calabi Lens, or ZeroMQ from RealSense + RTABMap).

### Replay Mode

To replay a recorded session without a live camera:

```bash
python -m rtsm --replay recordings/session1
```

---

## 2. Verify It's Running

```bash
curl http://localhost:8002/healthz
```

```json
{"status": "ok"}
```

Check detailed stats:

```bash
curl http://localhost:8002/stats/detailed
```

---

## 3. List Detected Objects

Once frames are streaming, objects will appear in memory:

```bash
curl http://localhost:8002/objects
```

Response:

```json
{
  "count": 62,
  "objects": [
    {
      "id": "a3f2c1d8",
      "xyz_world": [1.2, 0.4, 2.1],
      "stability": 0.82,
      "hits": 15,
      "confirmed": true,
      "label_primary": "backpack",
      "view_bins": 3
    }
  ]
}
```

---

## 4. Semantic Search

Ask natural language queries:

```bash
curl "http://localhost:8002/search/semantic?query=red%20mug&top_k=5"
```

Response:

```json
{
  "query": "red mug",
  "results": [
    {
      "id": "b7d4e2f1",
      "score": 0.82,
      "label_hint": "mug",
      "confirmed": true,
      "xyz_world": [0.8, 0.2, 1.5]
    }
  ]
}
```

---

## 5. Spatial Search

Find objects near a 3D point:

```bash
curl "http://localhost:8002/search/spatial?x=1.0&y=0.5&z=2.0&radius_m=0.5"
```

---

## 6. View in 3D (Optional)

Open the visualization frontend in your browser. The 3D viewer connects to the WebSocket at `ws://localhost:8083/ws` and shows a live point cloud with detected objects overlaid.

---

## 7. Record and Replay Sessions

Record a live session for later replay and benchmarking:

```bash
# Record while running pipeline
python -m rtsm --record recordings/my_session

# Record without GPU (raw frame capture only)
python -m rtsm --record recordings/my_session --record-only
```

Replay a recorded session:

```bash
python -m rtsm --replay recordings/session1
```

This feeds the recorded frames through the full pipeline at the original recording rate — no camera hardware needed. See the [Record & Replay Guide](../guides/record-replay.md) for details.

---

## 8. Check Analytics (Optional)

While a session is running (live or replay), view runtime analytics:

```bash
# Per-stage latency breakdown
curl http://localhost:8002/stats/detailed

# Segmentation analytics (mask counts, confirmation rates)
curl http://localhost:8002/analytics/segmentation
```

The analytics dashboard is also available in the 3D visualization frontend (`--viz`, or `rtsm demo`) as a separate tab. See the [Analytics Dashboard Guide](../guides/analytics-dashboard.md) for details.

---

## Next Steps

- [Configuration](configuration.md) — Tune thresholds and endpoints
- [REST API Reference](../api/rest-api.md) — Full API documentation
- [Record & Replay](../guides/record-replay.md) — Capture and replay sessions
- [RTAB-Map Setup](../guides/rtabmap-setup.md) — Connect your SLAM system
