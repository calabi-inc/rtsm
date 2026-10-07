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
| WebSocket (visualization) | `ws://localhost:8083/ws`, only with `--viz` (headless by default) |
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

### Navigating the 3D view

The View row has a mouse-scheme switch, remembered by the browser:

| scheme | orbit | pan | zoom |
|---|---|---|---|
| three.js (default) | left drag | right drag, or Shift + left drag | wheel, middle drag |
| Blender | middle drag (Alt + left drag on a trackpad) | Shift + middle drag; Shift + wheel up/down, Ctrl + wheel left/right | wheel; Ctrl + middle drag |
| Maya | Alt + left drag | Alt + middle drag | wheel; Alt + right drag |

The orange anchor is the rotation centre. Drag its arrows to move it across the model (the view slides with it), double-click a point to put it there, and press **P** or the Pivot button to hide or show it. While the anchor is shown it stays where you put it; while it is hidden, an orbit pivots at the depth under the cursor without moving the view. In every scheme the wheel zooms towards the cursor. Keys: numpad 1 / 3 / 7 for front / right / top (Ctrl for the opposite side), 9 for the opposite side of the current view, 2 / 4 / 6 / 8 to orbit 15°, **F** to frame the selected object (everything when nothing is selected), **Home** to frame everything, arrows to pan, **H** for the key panel. The up axis stays locked (turntable); use Flip X / Y / Z when a recording's world is Z-up. There is no orthographic view.

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
