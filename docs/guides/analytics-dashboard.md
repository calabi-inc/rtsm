# Analytics Dashboard

RTSM includes a runtime analytics system that tracks per-stage latency, segmentation output, and pipeline throughput. Analytics are available via the REST API and the built-in visualization dashboard.

---

## Enabling Analytics

Analytics are enabled by default. Configure in `config/rtsm.yaml`:

```yaml
analytics:
  enable: true
  retention_s: 3600     # Keep 1 hour of data
  buffer_frames: 300    # Ring buffer size (per-frame granularity)
```

---

## REST API Endpoints

### Pipeline latency

```bash
curl http://localhost:8002/stats/detailed
```

Returns per-stage timing breakdown:

| Stage | Description |
|-------|-------------|
| `segmentation` | Mask extraction (model inference) |
| `heuristics` | Mask filtering (area, depth, border, planarity) |
| `scoring` | Priority scoring and top-K selection |
| `clip` | CLIP visual encoding (224x224 crops) |
| `association` | Proximity + embedding matching to existing objects |

Each stage reports mean, P50, P95, and max latency.

### Segmentation analytics

```bash
curl http://localhost:8002/stats/analytics
```

The `segmentation` block returns segmentation-specific metrics (the `latency` block carries the throughput history, drop counters and the rollup health; see the [REST API](../api/rest-api.md)):

- Masks per frame (mean, min, max)
- Dual-confirmation rate (when using `dual` backend)
- Label distribution across confirmed objects
- Drop counters (frames rejected at each gate)

### Drop counters

The pipeline tracks where frames are dropped:

| Drop point | Description |
|------------|-------------|
| `tracking_state` | Camera tracking quality below threshold |
| `throttle` | Non-keyframe interval throttling |
| `queue_full` | Ingest queue overflow |
| `gate_rejection` | Sweep-based novelty gating |

---

## Visualization Dashboard

The analytics dashboard is built into the 3D visualization frontend. When the visualization server is running (`ws://localhost:8083/ws`), the frontend includes:

- **KPI Scorecards** — Input Hz, Processing Hz, Latency, Queue depth, Object count, Association rate
- **Time-series charts** — Throughput, per-stage latency breakdown, segmentation rates (powered by uPlot)
- **Config display** — Current backend, filtering thresholds, and scoring parameters

The dashboard updates in real-time via WebSocket push from the visualization server. The per-second history behind the charts is produced by the analytics ticker (`analytics.enable`), which runs whether or not the dashboard is open: a browser attaching mid-run receives the whole history since the process started, and headless runs expose the same data on `GET /stats/analytics`.

---

## Prometheus Metrics

RTSM exports Prometheus-compatible metrics at `http://localhost:8002/metrics`:

```bash
curl http://localhost:8002/metrics
```

Key gauges:

| Metric | Description |
|--------|-------------|
| `rtsm_live_objects` | Current number of live objects |
| `rtsm_confirmed_objects` | Number of confirmed objects |
| `rtsm_upserts_total` | Total vector DB upserts |

These can be scraped by Prometheus and visualized in Grafana for long-term monitoring.

---

## Next Steps

- [Benchmarks](../benchmarks.md) — Backend comparison with analytics data
- [REST API Reference](../api/rest-api.md) — Full API documentation
- [Record & Replay](record-replay.md) — Capture sessions for offline analysis
