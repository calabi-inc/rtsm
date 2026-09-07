# Configuration

RTSM ships its defaults in `rtsm/cfg/rtsm.yaml` and uses
`rtsm/cfg/demo_config.yaml` for `rtsm demo`. A source checkout may also have a
`config/` symlink. You can tune a small override file without editing the package.

## Start with the symptom

These commands work with core dependencies, without loading models or a GPU:

From a source checkout, `python -m rtsm config ...` uses the checkout directly
without requiring an updated console-script installation.

```bash
rtsm config explain --symptom pollution
rtsm config explain --symptom duplicates
rtsm config explain --demo --symptom missing
```

The guide shows the active backend's controls and their tradeoffs, and flags
settings that currently have no effect. Choose from `missing`, `duplicates`,
`pollution`, `latency`, and `search`, or omit `--symptom` for the full guide.

Older configuration files may still carry keys the pipeline never read:
`masks`, `staging.min_area_px`, `filters.depth.valid_min_pct`,
`filters.aspect_ratio`, `filters.solidity_min`, `filters.border_touch_max_pct`,
and the `filters.border` subsection. They are reported as advisories. The live
mask-area cutoff is `filters.min_area_px` and the live depth rejection
fraction is `staging.depth_valid_min`. Coverage and border contact are scored
softly through the `staging.w_*` weights rather than rejected outright.

The shipped main configuration includes the RC-car experiment's five-object
vocabulary. Inspect it before evaluating on a different scene. The demo has a
different vocabulary and confirmation policy. Neither is a universal reliability
preset.

## Repeatable tuning

Save just the values you want to investigate in a file such as `room.yaml`:

```yaml
# An experiment, not a calibrated recommendation.
staging:
  depth_valid_min: 0.10
```

Then inspect, validate and replay the same profile:

```bash
rtsm config explain --profile room.yaml --symptom pollution
rtsm config validate --profile room.yaml
rtsm --replay recordings/my-room --profile room.yaml
rtsm demo --profile room.yaml
```

Precedence is **base configuration → profiles in order → `--set` values in
order**. Nested mappings merge; lists replace. Omitted settings retain their
base values. Use `--profile` for a small patch; `--config` selects a complete
base file. Both runners support these flags. The demo's `--port` and `--no-viz`
flags take precedence over configuration values.

```bash
rtsm config show --profile room.yaml --set object.promote_hits=3 > trial.yaml
rtsm --replay recordings/my-room --config trial.yaml
```

`show` writes valid YAML to stdout and advisories to stderr. Each resolved
configuration has a SHA-256 fingerprint, also printed at runner startup.
Keep the snapshot with the recording and evaluation results. The fingerprint
identifies settings, not model weights, input data or code version.

Profiles and `--set` reject unknown paths to catch typos. They accept settings
from the shipped configurations, documented tuning controls, and any additional
expert settings already declared in your complete `--config` file. Validation
covers the documented tuning controls; it is not a complete schema or a check
of hardware/model compatibility.

Restart to apply a profile. There is no live-update API yet: component
constructors cache some values. Change one suspected cause, compare against the
same replay, and inspect wrong identities, misses, position error and stage
latency. More confirmed objects alone does not establish better quality.

---

## Minimal setup profile

For example, use the following as a `--profile` layered over the packaged
defaults. Incoming per-frame intrinsics take precedence in the pipeline:

```yaml
camera:
  width: 640
  height: 480
  fx: 604.6
  fy: 604.9
  cx: 318.8
  cy: 259.2

segmentation:
  backend: grounded_sam2   # Apache-2.0 default

io:
  receiver: websocket      # or zeromq for RealSense + RTABMap
```

---

## Camera Intrinsics

Match these to your RGB-D camera. Values below are for RealSense D435i:

```yaml
camera:
  model: D435i
  width: 640
  height: 480
  aligned_to: color
  fx: 604.634705
  fy: 604.906738
  cx: 318.806030
  cy: 259.239777
```

---

## Segmentation Backend

The segmentation backend determines which models extract object masks from each frame. This is the most impactful setting for both performance and licensing.

```yaml
segmentation:
  backend: grounded_sam2    # default (Apache-2.0)
  retina_masks: false       # false = 640x640 (fast), true = original resolution
```

| Backend | Description | License | Mean Latency |
|---------|-------------|---------|-------------|
| `grounded_sam2` | Grounding DINO + SAM2 (text-prompted) | Apache-2.0 | ~510 ms |
| `sam2` | SAM2 auto-mask (class-agnostic, no labels) | Apache-2.0 | ~860 ms |
| `dual` | FastSAM + YOLOE IoU merge (prompt-free, 1200+ categories) | AGPL-3.0 | ~210 ms |
| `fastsam` | FastSAM only (class-agnostic) | AGPL-3.0 | ~50 ms |
| `yoloe` | YOLOE only (open-vocab, prompt-free) | AGPL-3.0 | ~60 ms |

!!! note "AGPL backends require `ultralytics`"
    Backends using FastSAM or YOLOE require the `ultralytics` package (AGPL-3.0). Install with: `pip install "rtsm[gpu-ultralytics]"`

### Backend-Specific Settings

Each backend has its own configuration block:

```yaml
segmentation:
  # Grounding DINO + SAM2 (default)
  grounded_sam2:
    gdino_model_id: IDEA-Research/grounding-dino-tiny
    sam2_model_id: null          # null = use sam2.model_id
    device: cuda
    box_threshold: 0.25          # detection confidence
    text_threshold: 0.2          # text matching threshold
    vocab: null                  # null = default 30-class indoor vocab

  # SAM2 auto-mask
  sam2:
    model_id: facebook/sam2.1-hiera-small
    device: cuda
    points_per_side: 32
    pred_iou_thresh: 0.7
    stability_score_thresh: 0.92

  # FastSAM (AGPL)
  fastsam:
    model_path: model_store/fastsam/FastSAM-x.pt
    device: cuda
    imgsz: 640
    conf: 0.6
    iou: 0.7

  # YOLOE (AGPL)
  yoloe:
    model_path: model_store/yolo/yoloe-26s-seg-pf.pt
    device: cuda
    imgsz: 640
    conf: 0.25
    iou: 0.5

  # Dual-confirmation settings (backend: dual)
  dual:
    iou_confirm_threshold: 0.40     # IoU for cross-validation
    priority_boost_dual: 0.3        # priority boost for dual-confirmed masks
    prefer_mask: yoloe26            # which mask to keep for dual-confirmed
```

---

## I/O & Receiver

RTSM supports two input receiver backends:

```yaml
io:
  receiver: websocket               # websocket | zeromq
```

### WebSocket Receiver (Calabi Lens / ARKit)

```yaml
io:
  receiver: websocket
  websocket:
    host: "0.0.0.0"
    port: 8765
    require_tracking_normal: true    # drop frames with bad tracking
    keyframe_every_n: 30             # mark every Nth frame as keyframe
    nonkf_min_interval_s: 0.5        # throttle non-keyframes (~2/s)
    confidence_threshold: 2          # 0=all, 1=medium+high, 2=high only
```

### ZeroMQ Receiver (RealSense + RTABMap)

```yaml
io:
  receiver: zeromq
  camera_endpoint: tcp://172.27.240.1:5555   # D435i RGB-D frames
  rtabmap_endpoint: tcp://127.0.0.1:6000     # RTABMap pose topics
```

### Unit Conversion

If your depth source uses millimeters (e.g., RealSense D435i):

```yaml
units:
  depth_m_per_unit: 0.001     # mm → meters
  pose_m_per_unit: 1.0        # RTABMap poses are already in meters
```

---

## Frame-Quality Gate

Runs before segmentation on a strided subsample of each frame, so it costs well
under a millisecond and saves a full segmentation pass on unusable frames:

```yaml
gates:
  enable: true
  min_brightness: 5.0            # mean grey level (0-255); below = dark or covered lens
  min_std: 5.0                   # grey standard deviation; below = blank, uniform frame
  min_depth_valid: 0.02          # fraction of finite, positive depth pixels; below = depth failure
  sample_stride: 4               # pixel subsampling for the statistics
```

The defaults are deliberately conservative: they catch black, blank, and
depth-less frames only. Skipped frames are counted as `frame_rejections` in the
latency analytics and summarised in the log at most every ten seconds. Raise the
thresholds only with replay evidence, since a frame-level gate that is too
strict silently starves the map.

---

## Mask Filtering & Heuristics

Hard rejects applied to every mask before scoring:

```yaml
filters:
  min_area_px: 500               # hard reject: minimum mask area in pixels
  depth:
    z_min_m: 0.2                 # depth outside this range counts as invalid
    z_max_m: 8.0
    sigma_max_m: 0.50            # hard reject: max depth spread (metres)

staging:
  depth_erode_px: 1              # erode mask edges before depth statistics
  depth_valid_min: 0.02          # hard reject: min valid-depth fraction after erosion
  centroid_min_valid: 0.05       # min valid-depth fraction before a 3D centroid is computed
```

Coverage, border contact, and bounding-box size are not hard gates. They enter
the priority score through the `staging.w_*` weights in the next section, so a
wall-sized or edge-touching mask is ranked down rather than dropped.

---

## Staging & Top-K

Controls how masks are prioritized before CLIP encoding:

```yaml
staging:
  topk_preclip: 15               # max masks sent to CLIP per frame
  crop_pad_px: 6                 # padding around mask crops
  clip_input: 224                # CLIP input resolution

  # Priority weights
  w_coverage: 1.0
  w_border_fraction: -1.0
  w_depth_valid: 1.0
  w_depth_spread: -0.5
```

---

## Association

Controls how new observations are matched to existing objects:

```yaml
assoc:
  cos_min: 0.90                  # minimum cosine similarity for match
  gate_dist_base_m: 0.50         # 3D distance gate (meters)
  gate_reproj_px: 60             # reprojection distance gate (pixels)
  nearest_m_for_cos: 8           # compare cosine against K nearest by distance
  use_embeddings: true           # use CLIP embeddings for matching
  fallback_all_when_empty: true  # fallback to all WM objects if no nearby ones
```

---

## Object Promotion

Controls when proto-objects become confirmed:

```yaml
object:
  proto_ttl_s: 10.0              # seconds before unconfirmed proto expires
  promote_hits: 2                # observations needed to promote
  stability_promote: 0.55        # minimum stability score
  require_view_bins: 1           # minimum viewpoint directions
  stab_k: 0.45                   # stability update factor
  miss_decay: 0.5                # stability decay on missed frames
```

| Criterion | Config Key | Default |
|-----------|-----------|---------|
| Observation count | `promote_hits` | ≥ 2 |
| Stability score | `stability_promote` | ≥ 0.55 |
| View diversity | `require_view_bins` | ≥ 1 viewpoint |

---

## Sweep Cache & Spatial Grid

Controls the spatial indexing used for efficient neighbor lookups:

```yaml
sweep_cache:
  grid_size_m: 0.25              # cell size (meters)
  per_cell_cap: 64               # max object IDs per cell
  two_d: true                    # drop Z axis for indoor scenes
  yaw_bins: 12                   # 360° / 12 = 30° per bin
  pitch_bins: 5
```

---

## Vector Storage

Configure the embedding store for semantic search:

```yaml
vectors:
  enable: true
  backend: faiss                 # faiss | milvus
  dim: 512                       # CLIP ViT-B/32 embedding size
  faiss:
    index_path: model_store/faiss/index.flatip
  flush_period_s: 3.0
```

---

## API Server

```yaml
api:
  host: "0.0.0.0"
  port: 8002
```

The REST API is available at `http://localhost:8002`. See [REST API Reference](../api/rest-api.md).

---

## MCP (Model Context Protocol)

Enable the embedded MCP server to expose spatial memory to AI agents:

```yaml
mcp:
  enable: true                   # mounts at /mcp/ on the API server
```

When enabled, the MCP SSE endpoint is available at `http://localhost:8002/mcp/sse`. See [REST API — MCP](../api/rest-api.md#mcp-model-context-protocol) for available tools.

---

## Visualization

```yaml
visualization:
  enable: true
  host: 0.0.0.0
  port: 8083                     # WebSocket port for 3D frontend

  depth:
    min_m: 0.1
    max_m: 3.5                   # max depth for point cloud

  tsdf:
    enable: true
    voxel_size: 0.01             # 1cm voxels
    sdf_trunc: 0.04              # truncation distance
    max_depth_m: 3.5
    extract_every_n: 30          # extract mesh every N frames

  objects:
    push_interval_ms: 200        # push WM objects to clients
    include_proto: true          # include unconfirmed objects

  analytics:
    push_interval_ms: 1000       # analytics push cadence
    full_sync_interval_s: 30     # full history re-send interval
```

The 3D visualization frontend connects via WebSocket at `ws://localhost:8083/ws`. See [WebSocket API](../api/websocket.md).

---

## Analytics

```yaml
analytics:
  enable: true                   # enable runtime analytics buffers
  retention_s: 3600              # Tier 2 history retention (1 hour)
  buffer_frames: 300             # Tier 1 per-frame ring buffer size
```

When enabled, analytics are available via `GET /stats/analytics` and pushed to visualization clients.

---

## SLAM (On-Device)

Settings for handling SLAM-corrected poses from Calabi Lens (on-device RTABMap):

```yaml
slam:
  log_pose_source: true
  accept_pose_corrections: true     # accept loop closure corrections
  correct_working_memory: true      # update WM positions on correction
```

---

## Logging

```yaml
logging:
  periodic_summary: true
  summary_interval_s: 10.0
```

---

## Next Steps

- [Architecture](../concepts/architecture.md) — Understand the system design
- [REST API](../api/rest-api.md) — API reference
- [Benchmarks](../benchmarks.md) — Performance data per backend
