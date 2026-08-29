# Fix: IngestQueue should drop OLDEST, not newest (keep-latest for freshness)

**Status:** TODO (parked) · **Type:** robustness/correctness · **Scope:** small (~ingest path only)

## TL;DR
RTSM's frame ingest queue currently drops the **newest** incoming frame when full and
processes oldest-first (FIFO). Under sustained overload the pipeline ends up building
spatial memory from **stale** frames while discarding live ones — backwards for a
freshness-first spatial memory. Change it to **drop-oldest / keep-latest**, and shrink
the oversized buffer.

---

## Context (for an agent with no prior session)
RTSM ingests RGB-D + pose frames from a sensor (iPhone ARKit / RealSense), runs each
through a perception pipeline (segmentation → heuristics → CLIP/SigLIP embed →
association), and maintains a queryable spatial memory of 3D objects.

Frame flow: **WebSocket receiver (producer thread) → `IngestQueue` → pipeline worker
(consumer thread).** The queue decouples network I/O from the GPU pipeline. Processing
is **serial and latency-bounded** (one frame at a time; FastSAM ≈160 ms on RTX 5090,
≈716 ms on Jetson Orin). When frames arrive faster than the pipeline drains them, the
queue fills and the overload policy kicks in.

This is intentionally **not** a worker-pool design: the association step is a stateful,
order-dependent update of the single working memory, so processing must be sequential,
and on a single-GPU edge device extra workers would only contend for the GPU. So this
fix is about the **overload shedding policy**, not parallelism. Do **not** introduce
worker nodes here.

---

## The bug (with citations)
- `rtsm/io/ingest_queue.py` — `IngestQueue` wraps `queue.Queue(maxsize=256)`. `put(block=False)`
  → on `queue.Full` returns `False` (frame **not** enqueued).
- `rtsm/io/websocket.py` (~line 379) — on `put()==False`, the receiver drops the
  **incoming/newest** frame (`record_queue_drop()`, logs `"ingest queue full; dropping frame"`).
- `rtsm/core/pipeline.py:151` (`_get_snapshot_via_queue`) — consumer pulls from the front →
  **oldest-first**.

**Net under sustained overload:** the queue stays packed with the oldest 256 frames, the
pipeline grinds oldest→newest, and every fresh frame is discarded → memory reflects
seconds-old viewpoints.

**Latency caveat:** this is currently *latent*. With the upstream input throttle
(`websocket.py` non-KF min-interval) + the novelty/sweep ingest gate
(`rtsm/stores/sweep_policy.py` via `pipeline.py:156`) thinning the stream, and low frame
rates, the 256-deep buffer rarely fills in short replay tests — so it's invisible today.
It surfaces under sustained real streaming, especially with heavy backends (grounded_sam2).

---

## The fix
**1. Drop-oldest / keep-latest.** When the queue is full, evict the oldest frame and
enqueue the new one, so the freshest frames always win.

**2. Shrink the buffer.** `maxsize=256` is far too large for a freshness-first memory
(256 frames of backlog = many seconds of latency before shedding even begins). The
novelty/sweep gate already drops redundant frames upstream, so a small buffer (≈2–8) is
enough to smooth jitter without hoarding stale frames. Make it a config knob in
`rtsm/cfg/rtsm.yaml` (e.g. `ingest.queue_maxsize`) rather than hardcoded; default small.

### Implementation sketch — `rtsm/io/ingest_queue.py`
```python
import queue, threading

class IngestQueue:
    def __init__(self, maxsize: int = 8) -> None:
        self._q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()  # guards the evict+put compound op

    def put(self, pkt, block: bool = False, timeout=None) -> bool:
        """Enqueue the newest frame, keeping latest.
        Returns True if a STALE (oldest) frame was evicted to make room
        (i.e. a drop happened), False otherwise. The newest frame is ALWAYS accepted."""
        with self._lock:
            try:
                self._q.put_nowait(pkt)
                return False           # accepted, no drop
            except queue.Full:
                try:
                    self._q.get_nowait()   # evict oldest
                except queue.Empty:
                    pass
                self._q.put_nowait(pkt)    # room now; keep latest
                return True            # accepted newest, dropped a stale frame
```

### Corresponding caller update — `rtsm/io/websocket.py` (~line 379)
```python
# BEFORE: put() False meant the incoming frame was dropped
if self.ingest_q.put(pkt):
    frames_enqueued += 1
else:
    self._latency_analytics.record_queue_drop()
    logger.info("[websocket] ingest queue full; dropping frame")

# AFTER: newest is always accepted; True means a stale frame was shed
dropped_old = self.ingest_q.put(pkt)
frames_enqueued += 1
if dropped_old:
    self._latency_analytics.record_queue_drop()       # counts a STALE frame dropped
    logger.info("[websocket] ingest queue full; dropped oldest to keep latest")
```

### Gotchas — must handle
- **Thread-safety:** the evict-then-put is a compound op; `queue.Queue`'s internal lock
  does NOT cover it. Use the external lock shown so a concurrent producer can't slip in
  between the `get_nowait` and `put_nowait`. (Likely single-producer today, but be safe.)
- **Analytics semantics flip:** the current code counts a drop when the *incoming* frame
  is rejected. After this change the newest is always accepted and an *old* frame is
  dropped instead — update the counter's meaning and the log string (don't leave a
  message implying the live frame was lost). `record_queue_drop()` still fires, but now
  represents "stale frame shed."
- **Buffer size is a tradeoff:** too small → drops on transient jitter; too big →
  staleness. Default small, make it configurable, document the tradeoff.

---

## Scope / non-goals
- **In scope:** drop-oldest policy + smaller (configurable) buffer + caller/analytics update.
- **Out of scope:** worker pools / distributed queue (work is sequential + single-GPU —
  not needed); a worker-thread watchdog (separate reliability item, track elsewhere); any
  backend/TensorRT perf optimization.

---

## Verification
- **Unit test:** fill the queue to `maxsize`, `put()` N more, assert the queue now holds the
  **newest** items (oldest evicted) and the drop count == N.
- **Analytics:** confirm `record_queue_drop()` fires on eviction and the log no longer
  claims the incoming frame was lost.
- **Regression:** `python scripts/benchmark_datasheet.py fastsam` (replay) — object
  discovery should be unchanged (short sessions don't hit the limit, so behavior matches).
- **Optional stress test:** feed frames faster than the pipeline drains and assert memory
  reflects the most-recent viewpoints (e.g. a moving-camera replay where the latest pose
  should dominate).

## Files
| File | Change |
|------|--------|
| `rtsm/io/ingest_queue.py` | drop-oldest put + (optional) configurable maxsize — **primary** |
| `rtsm/io/websocket.py` (~379) | update drop accounting + log message |
| `rtsm/cfg/rtsm.yaml` | add `ingest.queue_maxsize` knob (if making configurable) |
| `rtsm/core/pipeline.py:151` | no change — FIFO `get` stays correct once we keep-latest |
