"""
Diagnostic event log: append-only JSONL writer (frame-flow trace).

Off by default; opt in via cfg.diagnostics.enabled. Each pipeline run gets a
fresh, auto-timestamped file (no appending across runs). Every line is a JSON
object with a ``kind`` field; the first line is ``kind: "meta"`` and carries
``schema_version``.

Line kinds (schema_version 2, 2026-09-09 — P1 task 0 of the Gate 4.5 plan):

  meta      once per file: schema_version, wall time, pid, and what the runner
            adds (ingest_clock: wall | sensor).
  receiver  one per RECEIVER DECISION (websocket / replay / zeromq thread):
            enqueued, or dropped with the reason (malformed, parse_error,
            tracking_state, throttle, duplicate_ts, no_camera_frame,
            queue_full). Carries the source's seq / t_sensor_ns / is_keyframe
            when the header parsed. ZeroMQ has no source seq: its join key is
            (t_sensor_ns, is_keyframe).
            depth_valid_frac (P1 task 2): the finite fraction of the decoded
            depth BEFORE the confidence filter -- one statistic on every
            websocket / replay line written after the depth decode (throttled
            and queue_full frames included; malformed / tracking lines carry
            None). ZeroMQ lines never carry it (depth stays encoded for the
            frames it refuses). It is a per-frame statistic, not part of the
            comparator tuple below.
  dequeue   one per DEQUEUED frame (pipeline thread), including the frames the
            ingest gate or the frame-quality gate rejects and the frames whose
            present pose fails conversion: outcome (processed | gate_rejected
            | frame_rejected | dropped), the reason (IngestDecision.reason,
            FrameGateDecision.reason, "keyframe", "no_pose", "gate_error",
            "pose_conversion_failed"), queue_wait_s = dequeue time minus
            TimeBundle.t_mono_s (both wall; that stamp is set when the
            FramePacket is built, after decode, so this is the ingest-queue
            wait — deliberately kept on wall so it stays a latency measure),
            and clock_s = the ingest clock (rtsm/core/clock.py) after
            advancing to this frame. Under ingest.clock=sensor clock_s is
            anchored to the first frame's wall time, so compare it as
            differences, never as absolute values.
            outcome=processed means ADMITTED to processing: the line is written
            before segmentation so a crash mid-frame still leaves it; join with
            the frame line (same t_sensor_ns) for completion.
  frame     one per PROCESSED frame: masks, filter, scoring, association,
            timings (the schema_version-1 line, plus kind and t_sensor_ns;
            timestamp is now time.monotonic() like the other kinds).

A/A comparator contract: compare the receiver and dequeue streams PER KIND as
ordered sequences of (frame_seq, t_sensor_ns, decision/outcome, reason); never
compare file order across kinds (an enqueued line is written after put(), so
the pipeline's dequeue line can precede it), and ignore timestamp,
queue_wait_s, queue_depth and the meta line, which are run-specific; under
ingest.clock=sensor the (frame_seq, t_sensor_ns, outcome, reason) sequences
are identical across runs and replay speeds (P1 gate G1-B), and clock_s
differences match too. Writes are serialised with a lock because the receiver
thread and the pipeline thread both write.

Path resolution rules:
  - None / unset  -> eval_output/<YYYYMMDD_HHMMSS>/events.jsonl  (per-run, auto)
  - "foo/bar/"    -> foo/bar/<YYYYMMDD_HHMMSS>/events.jsonl     (per-run inside dir)
  - "foo/bar.jsonl" -> exact path (overwrites with warning if exists)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

# Receiver decisions
RX_ENQUEUED = "enqueued"
RX_DROPPED = "dropped"
# Receiver drop reasons
RX_MALFORMED = "malformed"            # truncated / unparseable framing (before or after the header)
RX_PARSE_ERROR = "parse_error"        # header parsed, but pose / decode / field parsing raised
RX_TRACKING = "tracking_state"
RX_THROTTLE = "throttle"
RX_DUPLICATE_TS = "duplicate_ts"
RX_NO_CAMERA_FRAME = "no_camera_frame"
RX_QUEUE_FULL = "queue_full"
# Dequeue outcomes
DQ_PROCESSED = "processed"            # admitted to processing (written before segmentation)
DQ_GATE_REJECTED = "gate_rejected"
DQ_FRAME_REJECTED = "frame_rejected"
DQ_DROPPED = "dropped"                # dequeued and discarded before the gates (pose conversion failed)
# Dequeue reasons that are not an IngestDecision / FrameGateDecision reason
DQ_REASON_KEYFRAME = "keyframe"
DQ_REASON_NO_POSE = "no_pose"
DQ_REASON_GATE_ERROR = "gate_error"
DQ_REASON_POSE_CONVERSION = "pose_conversion_failed"


@dataclass
class FrameEvent:
    """One line per PROCESSED frame. Numbers in milliseconds where named *_ms."""
    timestamp: float
    frame_seq: int
    is_keyframe: bool
    n_masks_raw: int
    filter: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    n_matched: int = 0
    n_created: int = 0
    n_objects_confirmed: int = 0
    timing_ms: Dict[str, float] = field(default_factory=dict)
    t_sensor_ns: Optional[int] = None
    kind: str = "frame"


@dataclass
class DequeueEvent:
    """One line per DEQUEUED frame (pipeline thread), rejected or not."""
    timestamp: float                 # time.monotonic() at dequeue
    frame_seq: Optional[int]
    t_sensor_ns: Optional[int]
    is_keyframe: bool
    queue_wait_s: float              # dequeue mono - FramePacket.time.t_mono_s (stamped at packet build, ~enqueue)
    queue_depth: int                 # ingest queue depth right after this dequeue
    outcome: str                     # processed | gate_rejected | frame_rejected | dropped
    reason: str                      # gate reason, or one of the DQ_REASON_* labels
    clock_s: Optional[float] = None  # ingest clock (rtsm/core/clock.py) after advancing to this frame
    kind: str = "dequeue"


@dataclass
class ReceiverEvent:
    """One line per RECEIVER DECISION (websocket / replay / zeromq thread)."""
    timestamp: float                 # time.monotonic() at the decision
    source: str                      # websocket | replay | zeromq
    decision: str                    # enqueued | dropped
    reason: str                      # "" when enqueued, else the drop reason
    frame_seq: Optional[int] = None  # source seq (header frame_id) when parsed
    t_sensor_ns: Optional[int] = None
    is_keyframe: Optional[bool] = None
    frame_count: Optional[int] = None  # receiver's running count after the tracking filter
    queue_depth: Optional[int] = None  # ingest queue depth after the decision
    depth_valid_frac: Optional[float] = None  # finite fraction of the decoded depth BEFORE the confidence filter (websocket/replay; dropped frames included)
    kind: str = "receiver"


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy scalars / arrays."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _resolve_path(configured: Optional[str], repo_root: Path) -> Path:
    """See module docstring for rules."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if configured is None or str(configured).strip() == "":
        return repo_root / "eval_output" / ts / "events.jsonl"

    p = Path(configured)
    if not p.is_absolute():
        p = repo_root / p

    if str(configured).endswith("/") or str(configured).endswith("\\") or p.is_dir():
        return p / ts / "events.jsonl"

    return p  # explicit file path


class EventLogWriter:
    """Append-only JSONL writer for diagnostic events.

    When `enabled=False`, every call is a no-op and `sink()` returns None so
    producers can skip building events entirely.
    """

    def __init__(self, enabled: bool, configured_path: Optional[str], repo_root: Optional[Path] = None,
                 extra_meta: Optional[Dict[str, Any]] = None):
        self._enabled = bool(enabled)
        self._fh = None
        self._path: Optional[Path] = None
        self._lock = threading.Lock()
        if not self._enabled:
            return

        root = repo_root if repo_root is not None else Path.cwd()
        self._path = _resolve_path(configured_path, root)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            logger.warning(f"event_log: overwriting existing file {self._path}")

        # WRITE mode (truncate) — each run starts fresh. Line-buffered so a
        # crash mid-run still preserves prior frames.
        self._fh = self._path.open("w", encoding="utf-8", buffering=1)
        meta: Dict[str, Any] = {
            "kind": "meta",
            "schema_version": SCHEMA_VERSION,
            "created_wall_utc_s": time.time(),
            "created_mono_s": time.monotonic(),
            "pid": os.getpid(),
        }
        if extra_meta:
            meta.update({k: v for k, v in extra_meta.items() if k != "kind"})
        self.write(meta)
        logger.info(f"event_log: writing diagnostics to {self._path}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def write(self, event: Any) -> None:
        """Write one event (a dataclass or a plain dict). No-op when disabled."""
        if not self._enabled or self._fh is None:
            return
        payload = asdict(event) if is_dataclass(event) else dict(event)
        line = json.dumps(payload, default=_json_default) + "\n"
        with self._lock:
            fh = self._fh
            if fh is not None:
                fh.write(line)

    def sink(self) -> Optional[Callable[[Any], None]]:
        """The callable receivers get as `event_sink`: None when disabled."""
        return self.write if self._enabled else None

    def close(self) -> None:
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001 — closing a log must never raise
                logger.debug("event_log: close failed", exc_info=True)

    def __enter__(self) -> "EventLogWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def summarize_sources(entries) -> Dict[str, int]:
    """Helper: count entries by confirmation_source."""
    counts: Dict[str, int] = {"dual": 0, "fastsam_only": 0, "yoloe_only": 0, "none": 0}
    for e in entries:
        src = e.confirmation_source or "none"
        counts[src] = counts.get(src, 0) + 1
    return counts
