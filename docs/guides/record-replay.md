# Record & Replay

RTSM can record live WebSocket sessions to disk and replay them through the full pipeline at the original rate (or faster/slower with `--replay-speed`). This enables reproducible benchmarking and offline testing without camera hardware. Under replay the ingest clock defaults to the frames' own timestamps (`ingest.clock: auto` → `sensor`), so the admitted frames and the resulting memory do not depend on replay pacing; pass `--set ingest.clock=wall` to time the replay by process time instead. Under replay `/stats.robot_pose` is written at receive time for every tracking-normal frame, exactly as live (the replayer feeds the same pose sink), so replay-based pose-freshness checks are meaningful; its `frame_epoch` is the recording's epoch (`0` for a single-session recording). Replay also selects `ingest.policy: lossless` by default (`auto`): the replayer waits when the ingest FIFO is full instead of dropping, so every recorded frame that passes the throttle reaches the pipeline regardless of `--replay-speed`; `--set ingest.policy=latest` replays through the live lanes instead (bounded, newest-first, not reproducible run to run). Do not combine `latest` with `--replay-speed` above 1 under the sensor clock: the keyframe-first order can hand the clock a slot frame several seconds of sensor time behind the keyframe just processed, and past 5 s the clock re-bases instead of clamping.

---

## Recording a Session

### Record while running the pipeline

Stream frames from a camera while running the full perception pipeline:

```bash
python -m rtsm --record recordings/my_session
```

This writes raw binary frames and metadata to the specified directory while processing them normally.

### Record without GPU (raw capture)

Capture raw frames without running segmentation or CLIP — useful for quick capture on a laptop without a GPU:

```bash
python -m rtsm --record recordings/my_session --record-only
```

!!! tip
    `--record-only` mode requires no GPU dependencies. Install with just `pip install rtsm` (core only).

### What gets recorded

Each recording session creates a directory with:

| File | Contents |
|------|----------|
| `messages.bin` | Raw binary WebSocket frames (RGB + depth + pose) |
| `index.jsonl` | Per-frame metadata (timestamps, offsets, sizes) |
| `meta.json` | Session metadata (config snapshot, device info) |

---

## Replaying a Session

Feed a recorded session through the full pipeline:

```bash
python -m rtsm --replay recordings/session1
```

This launches the complete RTSM stack (segmentation, CLIP, association, memory, API, visualization) and replays frames at the original recording rate.

### Included test dataset

RTSM ships with a test recording for quick verification:

```bash
# 162-frame bedroom scan (~458 seconds, iPhone ARKit via Calabi Lens)
python -m rtsm --replay recordings/session1
```

The recording is stored with git-lfs. After cloning, run `git lfs pull` if the binary files aren't downloaded.

---

## Use Cases

- **Benchmarking** — Compare segmentation backends on the same input frames
- **Debugging** — Reproduce pipeline issues without needing the camera
- **CI/CD** — Run automated pipeline tests with deterministic input
- **A/B testing** — Use `scripts/debug_segmentation.py` to generate side-by-side comparisons

### A/B segmentation comparison

```bash
python scripts/debug_segmentation.py --recording recordings/session1
```

This generates an HTML viewer in `debug/session1/compare.html` showing FastSAM vs YOLOE overlays per frame.

---

## Next Steps

- [Quick Start](../getting-started/quick-start.md) — Run your first session
- [Benchmarks](../benchmarks.md) — Full performance comparison
- [Configuration](../getting-started/configuration.md) — Tune backends and thresholds
