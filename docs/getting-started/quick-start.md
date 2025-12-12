# Quick Start

This guide walks you through running RTSM and making your first semantic query.

---

## 1. Start RTSM

RTSM expects an RGB-D stream with poses via ZeroMQ. Start the main service:

```bash
python -m rtsm.run
```

This launches:

| Service | Address |
|---------|---------|
| REST API | `http://localhost:8000` |
| WebSocket (visualization) | `ws://localhost:8081` |

---

## 2. Verify It's Running

```bash
curl http://localhost:8000/stats/detailed
```

You should see system stats including frame count, object count, and memory usage.

---

## 3. List Detected Objects

Once frames are streaming, objects will appear in memory:

```bash
curl http://localhost:8000/objects
```

Response:

```json
[
  {
    "id": "a3f2c1",
    "label": "backpack",
    "xyz": [1.2, 0.4, 2.1],
    "confidence": 0.87
  },
  ...
]
```

---

## 4. Semantic Search

Ask natural language queries:

```bash
curl "http://localhost:8000/search/semantic?query=red%20mug&top_k=5"
```

Response:

```json
{
  "query": "red mug",
  "results": [
    {
      "id": "b7d4e2",
      "label": "mug",
      "xyz": [0.8, 0.2, 1.5],
      "score": 0.82
    }
  ]
}
```

---

## 5. View in 3D (Optional)

Open the visualization demo in your browser:

```
http://localhost:8081
```

This shows a Three.js point cloud with detected objects overlaid.

---

## Next Steps

- [Configuration](configuration.md) — Tune thresholds and endpoints
- [REST API Reference](../api/rest-api.md) — Full API documentation
- [RTAB-Map Setup](../guides/rtabmap-setup.md) — Connect your SLAM system
