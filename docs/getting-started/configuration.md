# Configuration

RTSM is configured via `config/rtsm.yaml`. This page covers the main settings.

---

## Configuration File

```yaml
# config/rtsm.yaml

camera:
  width: 640
  height: 480
  fx: 615.0  # focal length x
  fy: 615.0  # focal length y
  cx: 320.0  # principal point x
  cy: 240.0  # principal point y

io:
  zmq_camera_addr: "tcp://localhost:5555"
  zmq_slam_addr: "tcp://localhost:5556"

pipeline:
  # Mask filtering
  min_mask_area: 0.001    # min area as fraction of image
  max_mask_area: 0.5      # max area as fraction of image
  top_k_masks: 20         # max masks per frame

  # Association thresholds
  proximity_radius: 0.3   # meters
  embedding_threshold: 0.7

memory:
  # Object promotion
  min_hits: 2
  min_stability: 0.5
  min_views: 2

  # Long-term memory
  ltm_upsert_interval: 5.0  # seconds
  vector_store: "faiss"     # or "milvus"

api:
  host: "0.0.0.0"
  port: 8000

visualization:
  ws_port: 8081
```

---

## Key Settings

### Camera Intrinsics

Match these to your RGB-D camera. For RealSense D435i:

```yaml
camera:
  width: 640
  height: 480
  fx: 615.0
  fy: 615.0
  cx: 320.0
  cy: 240.0
```

### I/O Endpoints

Configure ZeroMQ addresses for your data sources:

```yaml
io:
  zmq_camera_addr: "tcp://localhost:5555"  # RGB-D frames
  zmq_slam_addr: "tcp://localhost:5556"    # Pose estimates
```

### Pipeline Tuning

Adjust mask filtering for your environment:

```yaml
pipeline:
  min_mask_area: 0.001  # filter tiny noise
  max_mask_area: 0.5    # filter wall-sized masks
  top_k_masks: 20       # limit processing per frame
```

### Memory Settings

Control when proto-objects become confirmed:

```yaml
memory:
  min_hits: 2        # seen at least twice
  min_stability: 0.5 # embedding consistency
  min_views: 2       # from multiple viewpoints
```

---

## Environment Variables

Some settings can be overridden via environment:

```bash
export RTSM_API_PORT=9000
export RTSM_LOG_LEVEL=DEBUG
```

---

## Next Steps

- [Architecture](../concepts/architecture.md) — Understand the system design
- [REST API](../api/rest-api.md) — API reference
