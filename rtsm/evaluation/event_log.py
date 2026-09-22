"""
Diagnostic event log: append-only JSONL writer (frame-flow trace).

Off by default; opt in via cfg.diagnostics.enabled. Each pipeline run gets a
fresh, auto-timestamped file (no appending across runs). Every line is a JSON
object with a ``kind`` field; the first line is ``kind: "meta"`` and carries
``schema_version``.

Line kinds (schema_version 3: 2 = P1 task 0 2026-09-09, 3 = the P2 ledgers 2026-09-21,
additive):

  meta      once per file: schema_version, wall time, pid, and what the runner
            adds (ingest_clock: wall | sensor; ingest_policy: latest | lossless
            | legacy).
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
            lane / rx_seq (P1 task 3): the ingest lane the frame was admitted
            to and the receiver-local frame count. Under ingest.policy=latest
            a frame can lose a SECOND receiver line after its enqueued one:
            source="lanes", decision=dropped, reason superseded | kf_dropped
            | age, written by the runner's lane drop handler from whichever
            thread discarded it (put side or dequeue side). Once the pipeline
            has drained, every enqueued frame is either dequeued or has a
            lanes-source dropped line; a frame still waiting at shutdown is
            neither (close() keeps draining, it does not report). Under
            lossless / legacy the lanes write nothing.
            rx_seq: websocket / replay = 1-based count of binary messages
            received by this receiver in this process (malformed and
            tracking-dropped included, never reset per session); zeromq =
            count of enqueue attempts (None on its malformed-pose lines).
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
  pose      LEDGER (P2 stage A, ledger schema 1; written only when
            diagnostics.ledgers is on): one per SENSOR FRAME the receiver saw,
            at input rate. Websocket / replay write it at two points of the
            parser: in the tracking-state filter, BEFORE the frame is dropped
            (tracking-limited frames are in the ledger with their state and,
            when it parses, their pose; pose_error otherwise), and for frames
            that pass the filter right after the depth decode -- before the
            keyframe rule, the throttle and the queue admission, so throttled
            and refused frames are in it too. The line carries what the
            receiver knows there: the post-flip pose (the same the mailbox and
            the FramePacket get), the header wall stamp + pose_clock, the
            epoch, depth_valid_frac (pre-confidence-filter, the SAME value as
            the receiver line) and conf_hist (counts of confidence 0/1/2 over
            the RAW map, before it is resized to the depth). mailbox_write
            says whether the receiver called its pose sink for the frame. It
            does NOT repeat the admission outcome: join it to the receiver
            line on (source, rx_seq) (websocket / replay) or on
            (source, t_sensor_ns) (zeromq: one line per rtabmap.tracking_pose,
            tracking_state "not_available", never for kf_pose). Ledger kinds
            are excluded from the A/A comparator below by kind.
  obs       LEDGER (P2 stage B): one per CANDIDATE the associator looked at on
            a processed frame (pipeline thread, written right after
            association), so per frame #obs == the frame line's
            scoring.n_selected. outcome = matched | created | spawn_capped |
            no_p_cam | no_embedding | create_failed. Carries the RAW
            measurement before any memory smoothing: p_world = T_wc @ p_cam as
            the associator computed it, p_cam / range_m, the WM's own view_bin
            for that direction, the winning match's residuals (cos_sim, dist_m,
            px_err) plus n_nearby / n_gate_survivors / max_cos so a spawn can
            be audited against the gates (matched_without_scoring marks a
            match the associator's fallback path made without gating this
            candidate: no residuals), label_topk, priority, the MaskStats
            numbers, and the frame context (ids, epoch, lane, keyframe origin,
            camera pose). Join to `frame` / `dequeue` on t_sensor_ns.

A/A comparator contract: compare the receiver and dequeue streams PER (KIND,
SOURCE) as ordered sequences of (frame_seq, t_sensor_ns, decision/outcome,
reason) -- lanes-source lines are their own sequence and are nondeterministic
in file order under policy=latest (two writer threads); never
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
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3

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
# Ingest-lane reasons (rtsm/io/ingest_lanes.py). kf_lane_full / closed are
# refusals traced by the receiver; the other three happen INSIDE the lanes
# after the receiver traced the frame as enqueued and are written with
# source = SOURCE_LANES by the runner's drop handler.
RX_KF_LANE_FULL = "kf_lane_full"      # source keyframe refused before decode (overflow=reject)
RX_SUPERSEDED = "superseded"          # waiting non-keyframe replaced by a newer one (latest slot)
RX_KF_DROPPED = "kf_dropped"          # oldest waiting keyframe discarded (overflow=drop_oldest)
RX_AGE = "age"                        # non-keyframe older than max_frame_age_s at dequeue
RX_CLOSED = "closed"                  # put() after the queue was closed (shutdown)
SOURCE_LANES = "lanes"
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

# ---- Ledgers (P2). Additive kinds with their own schema number in the meta
# line's `ledgers` block (frozen at G2-C). Stage A = pose; obs / view follow.
LEDGER_SCHEMA = 1
LEDGER_FORMATS = ("jsonl", "parquet")
KIND_POSE = "pose"
KIND_OBS = "obs"
LEDGER_KINDS = (KIND_POSE, KIND_OBS)
# Observation outcomes (the associator's six exits per candidate)
OBS_MATCHED = "matched"
OBS_CREATED = "created"
OBS_SPAWN_CAPPED = "spawn_capped"       # per-cell spawn cap hit (only when a caller passes the counter)
OBS_NO_P_CAM = "no_p_cam"               # no camera-frame centroid (depth missing under the mask)
OBS_NO_EMBEDDING = "no_embedding"       # embeddings on, candidate has none (crop / encode failed)
OBS_CREATE_FAILED = "create_failed"     # WorkingMemory.create_object returned None
OBS_OUTCOMES = (OBS_MATCHED, OBS_CREATED, OBS_SPAWN_CAPPED, OBS_NO_P_CAM, OBS_NO_EMBEDDING, OBS_CREATE_FAILED)
# tracking_state values as the receivers see them (ARKit header strings;
# ZeroMQ has none and writes TS_NOT_AVAILABLE on every line).
TS_NORMAL = "normal"
TS_NOT_AVAILABLE = "not_available"


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
    source: str                      # websocket | replay | zeromq | lanes (the runner's lane drop handler)
    decision: str                    # enqueued | dropped
    reason: str                      # "" when enqueued, else the drop reason
    frame_seq: Optional[int] = None  # source seq (header frame_id) when parsed
    t_sensor_ns: Optional[int] = None
    is_keyframe: Optional[bool] = None
    frame_count: Optional[int] = None  # receiver's running count after the tracking filter
    queue_depth: Optional[int] = None  # ingest queue depth after the decision
    depth_valid_frac: Optional[float] = None  # finite fraction of the decoded depth BEFORE the confidence filter (websocket/replay; dropped frames included)
    lane: Optional[str] = None       # IngestMeta.lane once admitted (keyframe | latest | fifo; None = legacy queue / not admitted)
    rx_seq: Optional[int] = None     # receiver-local count: binary messages received (websocket/replay) | enqueue attempts (zeromq)
    kind: str = "receiver"


@dataclass
class PoseEvent:
    """One line per sensor frame at the receiver (P2 pose ledger, schema 1)."""
    timestamp: float                          # time.monotonic() at the write
    source: str                               # websocket | replay | zeromq
    rx_seq: Optional[int]                     # websocket/replay: join key to the receiver line; zeromq: None
    frame_seq: Optional[int]                  # header frame_id (websocket/replay); None on zeromq
    t_sensor_ns: Optional[int]                # header timestamp_ns (0 / missing -> None); zeromq: the pose stamp (join key)
    t_wall_utc_s: float                       # header unix_timestamp, or this process's time.time()
    pose_clock: str                           # sender | server: where t_wall_utc_s came from
    epoch: int                                # frame_epoch at the write
    tracking_state: str                       # header string verbatim; zeromq: not_available
    mailbox_write: bool                       # the receiver called its pose sink for this frame
    t_wc: Optional[List[float]] = None        # post-flip translation (m); None when the pose failed to parse
    q_wc_xyzw: Optional[List[float]] = None   # post-flip unit quaternion; None with t_wc
    pose_error: Optional[str] = None          # parse failure text (only on frames the tracking filter drops)
    depth_valid_frac: Optional[float] = None  # pre-confidence-filter finite fraction; None when depth was not decoded
    conf_hist: Optional[List[int]] = None     # counts of confidence 0 / 1 / 2 over the raw map; None without a map
    kind: str = "pose"


@dataclass
class ObservationEvent:
    """One line per candidate the associator looked at (P2 observation ledger, schema 1)."""
    timestamp: float                          # time.monotonic() at the write (after association)
    frame_seq: Optional[int]
    t_sensor_ns: Optional[int]                # join key to the frame / dequeue lines
    epoch: Optional[int]
    is_keyframe: bool
    lane: Optional[str]                       # IngestMeta.lane (None under the legacy queue)
    keyframe_origin: Optional[str]            # minted | source | None
    rx_seq: Optional[int]
    cam_t_wc: Optional[List[float]]           # the packet's post-flip camera pose
    cam_q_wc_xyzw: Optional[List[float]]
    cand_idx: int                             # mask index in the segmentation output (joins ScoringTrace.mask_idx)
    outcome: str                              # one of OBS_OUTCOMES
    object_id: Optional[str] = None           # matched or created id
    p_world: Optional[List[float]] = None     # RAW world point (T_wc @ p_cam), before any EMA
    p_cam: Optional[List[float]] = None       # camera-frame centroid of the mask
    range_m: Optional[float] = None           # |p_cam|
    view_bin: Optional[int] = None            # the WM's bin for p_cam's direction
    cos_sim: Optional[float] = None           # winning match's cosine (matched only)
    dist_m: Optional[float] = None            # winning match's 3-D distance (matched only)
    px_err: Optional[float] = None            # winning match's reprojection error (matched only; 0 without intrinsics)
    n_nearby: int = 0                         # objects the index returned around p_world
    n_gate_survivors: int = 0                 # of those, how many passed the distance / z / reprojection gates
    max_cos: Optional[float] = None           # best cosine seen among scored survivors, passed or not
    matched_without_scoring: bool = False     # matched via a stale best_id (associator fallback path; residuals absent)
    label_topk: Optional[List[List[Any]]] = None   # [[label, score], ...] (detection label first)
    priority: float = 0.0
    mask: Optional[Dict[str, Any]] = None     # MaskStats numbers (area_px, bbox, coverage, ..., centroid_px)
    kind: str = "obs"


def _pyarrow_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("pyarrow") is not None


@dataclass(frozen=True)
class LedgerConfig:
    """The validated ``diagnostics:`` block as the runners read it (P2)."""
    enabled: bool
    ledgers: bool
    ledger_format: str
    event_log_path: Optional[str]


def resolve_ledger_config(cfg: Any) -> LedgerConfig:
    """Validate ``diagnostics.enabled`` / ``ledgers`` / ``ledger_format`` /
    ``event_log_path``. Raises ValueError; the runners route it through
    ``parser.error`` before any model loads. ``ledgers: true`` with
    ``enabled: false`` is accepted (the writer logs that they are ignored);
    ``parquet`` without pyarrow is refused only when it would take effect."""
    diag = (cfg.get("diagnostics") or {}) if isinstance(cfg, dict) else {}
    if not isinstance(diag, dict):
        raise ValueError(f"diagnostics: must be a mapping of settings; got {diag!r}")
    enabled = bool(diag.get("enabled", False))
    ledgers = diag.get("ledgers", False)
    if not isinstance(ledgers, bool):
        raise ValueError(f"diagnostics.ledgers must be true or false; got {ledgers!r}")
    fmt = diag.get("ledger_format", "jsonl")
    if not isinstance(fmt, str) or fmt.strip().lower() not in LEDGER_FORMATS:
        raise ValueError(f"diagnostics.ledger_format must be one of {', '.join(LEDGER_FORMATS)}; got {fmt!r}")
    fmt = fmt.strip().lower()
    if fmt == "parquet" and enabled and ledgers and not _pyarrow_available():
        raise ValueError("diagnostics.ledger_format=parquet needs pyarrow: pip install \"rtsm[eval]\"")
    path = diag.get("event_log_path")
    return LedgerConfig(enabled=enabled, ledgers=ledgers, ledger_format=fmt,
                        event_log_path=(str(path) if path not in (None, "") else None))


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
                 extra_meta: Optional[Dict[str, Any]] = None, *, ledgers: bool = False,
                 ledger_format: str = "jsonl"):
        self._enabled = bool(enabled)
        # P2 ledgers ride in the same file behind a second switch: off => the
        # ledger sink is None and no producer builds a ledger line.
        self._ledgers = bool(self._enabled and ledgers)
        self._ledger_format = str(ledger_format or "jsonl").strip().lower()
        self._fh = None
        self._path: Optional[Path] = None
        self._lock = threading.Lock()
        if not self._enabled:
            if ledgers:
                logger.info("event_log: diagnostics.ledgers ignored (diagnostics.enabled is false)")
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
            "ledgers": ({"enabled": True, "schema": LEDGER_SCHEMA, "format": self._ledger_format}
                        if self._ledgers else {"enabled": False}),
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

    @property
    def ledgers_enabled(self) -> bool:
        return self._ledgers

    @property
    def ledger_format(self) -> str:
        return self._ledger_format

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

    def ledger_sink(self) -> Optional[Callable[[Any], None]]:
        """The callable producers get as `ledger_sink` (P2): None unless
        diagnostics.enabled AND diagnostics.ledgers are both true."""
        return self.write if self._ledgers else None

    def close(self) -> None:
        with self._lock:
            fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except Exception:  # noqa: BLE001 — closing a log must never raise
                logger.debug("event_log: close failed", exc_info=True)
            # Parquet is an offline conversion of the finished JSONL (which
            # stays the source of truth); a failure here is logged, never raised.
            if self._ledgers and self._ledger_format == "parquet" and self._path is not None:
                try:
                    from rtsm.evaluation.ledger import to_parquet
                    written = to_parquet(self._path)
                    logger.info("event_log: ledgers converted to parquet: %s",
                                ", ".join(str(v) for v in written.values()) or "(nothing)")
                except Exception:  # noqa: BLE001
                    logger.warning("event_log: parquet conversion failed (the JSONL is intact)", exc_info=True)

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
