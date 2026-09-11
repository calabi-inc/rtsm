"""
Ingest lanes: per-lane admission between the receivers and the pipeline
(Gate 4.5 plan, P1 task 3; design memo ingest-drop-policy-2026-09; design
block + review outcome in execution-plan-gate45-2026-09.md).

Why: the legacy ingest queue is a 512-deep tail-drop FIFO. Under congestion
(a ~1 Hz segmenter against a ~5 Hz phone) it holds minutes-old frames, ~4 GB
of decoded pixels, and drops the NEWEST frame -- the E1 wedge. Freshness has
a different value per frame kind, so admission is per lane:

  keyframe lane   small FIFO (``keyframe_lane_depth``), drained first. On
                  overflow ``drop_oldest`` discards the OLDEST waiting
                  keyframe (counted ``kf_dropped``) so the freshest keyframe
                  always wins; ``reject`` refuses a SOURCE keyframe before it
                  is decoded (``kf_lane_full``) and still drop-oldest for
                  receiver-minted ones. A keyframe is never demoted and never
                  age-dropped: its value is its pose + view, not its age (the
                  v2 plan's promotion-stop was removed by the design review --
                  with equal defaults it demoted the freshest keyframe, kept
                  the three stalest, and could demote the session seed).
  latest slot     one non-keyframe waiting; a newer one supersedes it
                  (counted ``nonkf_superseded``; MediaPipe FlowLimiter shape).
                  Supersession is the designed steady state under congestion,
                  not a fault.
  age check       at dequeue, slot only: a non-keyframe older than
                  ``max_frame_age_s`` (wall-monotonic since admission) is
                  dropped (``age_dropped``) instead of processed. A stall
                  valve; it never fires in steady state.

Policies (``ingest.policy``):
  latest    the lanes above -- the live default.
  lossless  one FIFO of ``lossless_depth``; put() BLOCKS the producer instead
            of dropping, until a slot frees or close() is called. REPLAY /
            EVAL ONLY: a blocking put inside a live receive loop would stall
            the websocket event loop or the ZeroMQ subscriber, so
            resolve_policy() refuses it for live receivers. No age check.
            Blocking is determinism-neutral under the sensor clock (the
            throttle stamps at the admit decision, before put; the pipeline
            gates on SensorClock), so the depth only bounds memory.
  legacy    today's queue.Queue(512) tail-drop (rtsm/io/ingest_queue.py,
            untouched), kept one release as the config-only rollback.
  auto      lossless under --replay / ``rtsm demo``, latest live -- the same
            shape as ``ingest.clock: auto``.

Surface shared with the legacy IngestQueue (duck-typed; every consumer uses
only this): put(pkt, block=False) -> bool, get(timeout) -> pkt | None,
qsize(), full(), maxsize, policy, refusal(is_keyframe, keyframe_origin) ->
reason | None (the admit-BEFORE-decode check), depth() -> dict, stats() ->
dict, backlog_signal() -> dict, set_on_drop(cb), close(). ``block`` /
``timeout`` on put() are accepted for compatibility and IGNORED: the policy
decides whether a put blocks (lossless) or never does (latest, legacy).

Drops that happen INSIDE the lanes (superseded, kf_dropped, age) happen after
the receiver already traced the frame as enqueued; ``on_drop(pkt, reason)``
fires for them, outside the lock, so the runner can write a receiver-kind
trace line with ``source: "lanes"`` and feed the analytics counters. A
refused put() (returns False) is NOT reported through on_drop: the caller
traces it, reading the reason from ``pkt.ingest.drop_reason``.

Analytics mapping (lane_drop_handler): receiver refusals (queue_full,
kf_lane_full) and kf_dropped -> ``queue_drops`` (frames lost to ingest
capacity, comparable across policies); superseded -> ``superseded`` (its own
counter: a replaced non-keyframe is the designed steady state under
congestion, not a loss the dashboard should paint red); age -> ``age_drops``.

Thread model: receivers put() from their thread, the pipeline get()s from its
thread, the watchdog / API read depth() / stats() from others. One Condition
guards every field; callbacks never run under it; the lossless wait is
Condition.wait() in 100 ms slices and never sleeps holding the lock.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from rtsm.core.datamodel import FramePacket, IngestMeta
from rtsm.io.ingest_queue import IngestQueue

logger = logging.getLogger(__name__)

POLICY_AUTO = "auto"
POLICY_LATEST = "latest"
POLICY_LOSSLESS = "lossless"
POLICY_LEGACY = "legacy"
POLICIES = (POLICY_AUTO, POLICY_LATEST, POLICY_LOSSLESS, POLICY_LEGACY)

OVERFLOW_DROP_OLDEST = "drop_oldest"
OVERFLOW_REJECT = "reject"
OVERFLOWS = (OVERFLOW_DROP_OLDEST, OVERFLOW_REJECT)

# IngestMeta.lane values
LANE_KEYFRAME = "keyframe"
LANE_LATEST = "latest"
LANE_FIFO = "fifo"

# IngestMeta.keyframe_origin values
KF_MINTED = "minted"     # receiver-minted (websocket / replay: frame_count == 1 or every Nth)
KF_SOURCE = "source"     # source-flagged (ZeroMQ kf_pose, i.e. a SLAM keyframe)

# Drop reasons (= frame-flow trace reasons; see rtsm/evaluation/event_log.py)
DROP_QUEUE_FULL = "queue_full"        # legacy tail-drop refusal
DROP_KF_LANE_FULL = "kf_lane_full"    # source keyframe refused, overflow=reject
DROP_SUPERSEDED = "superseded"        # waiting non-keyframe replaced by a newer one
DROP_KF_DROPPED = "kf_dropped"        # oldest waiting keyframe discarded, overflow=drop_oldest
DROP_AGE = "age"                      # non-keyframe older than max_frame_age_s at dequeue
DROP_CLOSED = "closed"                # put() after close() (shutdown)

LEGACY_DEPTH = 512

DropCallback = Callable[[FramePacket, str], None]


def resolve_policy(configured: Any, *, replay: bool) -> str:
    """``ingest.policy`` -> concrete policy. ``auto`` = lossless under replay,
    latest live. ``lossless`` is refused for live receivers (its blocking put
    would stall the receive loop). Raises ValueError; the runners turn it into
    a config error at startup, like ``ingest.clock``."""
    mode = str(configured if configured is not None else POLICY_AUTO).strip().lower()
    if mode == POLICY_AUTO:
        return POLICY_LOSSLESS if replay else POLICY_LATEST
    if mode == POLICY_LOSSLESS and not replay:
        raise ValueError(
            "ingest.policy=lossless is replay/eval-only: its producer-paced put would block the live "
            "receive loop. Use latest (bounded lanes) or legacy (the old 512-deep queue) live."
        )
    if mode in (POLICY_LATEST, POLICY_LOSSLESS, POLICY_LEGACY):
        return mode
    raise ValueError(f"ingest.policy must be one of {', '.join(POLICIES)}; got {configured!r}")


@dataclass(frozen=True)
class LaneConfig:
    """The validated ``ingest:`` block: lane policy + depths, the receiver
    timing that moved here from ``io.websocket.*`` in P1 task 6
    (``keyframe_every_n`` -- websocket / replay only, ZeroMQ keyframes are the
    SLAM node's; ``nonkf_min_interval_s`` -- every receiver), and the ZeroMQ
    pairing window (``pair_window_s`` / ``pair_window_fps`` -> the derived
    frame cap; nothing else reads the rate). Build with :meth:`from_cfg` at
    startup so a typo fails through the config-error path before any model
    loads; the runners read these values from here, never from ``io.websocket``."""
    policy: str
    keyframe_lane_depth: int = 3
    keyframe_lane_overflow: str = OVERFLOW_DROP_OLDEST
    max_frame_age_s: Optional[float] = 2.0
    lossless_depth: int = 32
    configured_policy: str = POLICY_AUTO
    keyframe_every_n: int = 30
    nonkf_min_interval_s: float = 0.5
    pair_window_s: float = 2.0
    pair_window_fps: float = 30.0
    # Ingest-gate timing (read by IngestGate from the cfg; validated HERE so a
    # bad value exits before any model loads): 0 = off for both.
    non_kf_grace_s: float = 0.0
    dup_window_ns: int = 200_000_000

    @property
    def pair_window_frames(self) -> int:
        """ZeroMQ FrameWindow count cap: ceil(pair_window_s x pair_window_fps x 1.5)
        (90 at the defaults; the 1.5 is the same margin the kwarg carried).
        Rounded to 1e-6 first so a float product a hair above an integer does
        not ceil one frame too high (4.48 x 25 x 1.5 = 168.00000000000003)."""
        v = round(float(self.pair_window_s) * float(self.pair_window_fps) * 1.5, 6)
        return max(1, int(-(-v // 1)))

    @classmethod
    def from_cfg(cls, cfg: Optional[Dict[str, Any]], *, replay: bool) -> "LaneConfig":
        ing = (cfg or {}).get("ingest") or {}
        if not isinstance(ing, dict):
            raise ValueError(f"ingest: must be a mapping; got {ing!r}")
        configured = ing.get("policy", POLICY_AUTO)
        policy = resolve_policy(configured, replay=replay)
        kf_depth = _positive_int(ing.get("keyframe_lane_depth", 3), "ingest.keyframe_lane_depth")
        overflow = str(ing.get("keyframe_lane_overflow", OVERFLOW_DROP_OLDEST) or OVERFLOW_DROP_OLDEST).strip().lower()
        if overflow not in OVERFLOWS:
            raise ValueError(f"ingest.keyframe_lane_overflow must be drop_oldest | reject; got {overflow!r}")
        age = ing.get("max_frame_age_s", 2.0)
        if age is not None:
            try:
                age = float(age)
            except (TypeError, ValueError):
                raise ValueError(f"ingest.max_frame_age_s must be a number of seconds or null; got {age!r}")
            if age <= 0:
                raise ValueError(f"ingest.max_frame_age_s must be > 0 or null; got {age!r}")
        depth = _positive_int(ing.get("lossless_depth", 32), "ingest.lossless_depth")
        kf_every = _positive_int(ing.get("keyframe_every_n", 30), "ingest.keyframe_every_n")
        interval = _finite_float(ing.get("nonkf_min_interval_s", 0.5), "ingest.nonkf_min_interval_s", minimum=0.0)
        window_s = _finite_float(ing.get("pair_window_s", 2.0), "ingest.pair_window_s", minimum=0.0, strict=True)
        window_fps = _finite_float(ing.get("pair_window_fps", 30.0), "ingest.pair_window_fps", minimum=0.0, strict=True)
        grace = _finite_float(ing.get("non_kf_grace_s", 0.0), "ingest.non_kf_grace_s", minimum=0.0)
        dup_ns = _nonneg_int(ing.get("dup_window_ns", 200_000_000), "ingest.dup_window_ns")
        return cls(policy=policy, keyframe_lane_depth=kf_depth, keyframe_lane_overflow=overflow,
                   max_frame_age_s=age, lossless_depth=depth, configured_policy=str(configured),
                   keyframe_every_n=kf_every, nonkf_min_interval_s=interval,
                   pair_window_s=window_s, pair_window_fps=window_fps,
                   non_kf_grace_s=grace, dup_window_ns=dup_ns)


def _finite_float(v: Any, key: str, *, minimum: float, strict: bool = False) -> float:
    """A finite number >= minimum (> minimum when strict). Booleans and strings
    are not numbers (the same rule the tuning controls apply: YAML 1.1 reads
    `1e3` as a string; write `1000.0`)."""
    if isinstance(v, (bool, str)):
        raise ValueError(f"{key} must be a number; got {v!r}")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number; got {v!r}")
    if f != f or f in (float("inf"), float("-inf")):
        raise ValueError(f"{key} must be finite; got {v!r}")
    if (f <= minimum) if strict else (f < minimum):
        raise ValueError(f"{key} must be {'>' if strict else '>='} {minimum:g}; got {v!r}")
    return f


def _nonneg_int(v: Any, key: str) -> int:
    """A whole number >= 0 (0 = off for a window); booleans are not numbers."""
    if isinstance(v, (bool, str)):
        raise ValueError(f"{key} must be a whole number >= 0; got {v!r}")
    try:
        f = float(v)
        i = int(f)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a whole number >= 0; got {v!r}")
    if f != i or i < 0:
        raise ValueError(f"{key} must be a whole number >= 0; got {v!r}")
    return i


def _positive_int(v: Any, key: str) -> int:
    if isinstance(v, bool):
        raise ValueError(f"{key} must be a positive integer; got {v!r}")
    try:
        f = float(v)
        i = int(f)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a positive integer; got {v!r}")
    if f != i:
        raise ValueError(f"{key} must be a whole number; got {v!r}")
    if i < 1:
        raise ValueError(f"{key} must be >= 1; got {v!r}")
    return i


def ensure_meta(pkt: Any) -> Optional[IngestMeta]:
    """Return pkt.ingest, creating it for a FramePacket that arrived without
    one. Non-FramePacket objects (test sentinels) get None."""
    meta = getattr(pkt, "ingest", None)
    if meta is None and isinstance(pkt, FramePacket):
        meta = IngestMeta()
        pkt.ingest = meta
    return meta


class IngestLanes:
    """Per-lane admission (policy ``latest``) or a producer-paced FIFO
    (policy ``lossless``). See the module docstring."""

    def __init__(
        self,
        policy: str = POLICY_LATEST,
        *,
        keyframe_lane_depth: int = 3,
        keyframe_lane_overflow: str = OVERFLOW_DROP_OLDEST,
        max_frame_age_s: Optional[float] = 2.0,
        lossless_depth: int = 32,
        on_drop: Optional[DropCallback] = None,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if policy not in (POLICY_LATEST, POLICY_LOSSLESS):
            raise ValueError(f"IngestLanes takes policy latest | lossless (legacy is IngestQueue); got {policy!r}")
        if keyframe_lane_overflow not in OVERFLOWS:
            raise ValueError(f"keyframe_lane_overflow must be drop_oldest | reject; got {keyframe_lane_overflow!r}")
        self._policy = policy
        self._kf_depth = max(1, int(keyframe_lane_depth))
        self._kf_overflow = keyframe_lane_overflow
        self._max_age_s = None if max_frame_age_s is None else float(max_frame_age_s)
        if max_frame_age_s is not None and float(max_frame_age_s) <= 0:
            raise ValueError("max_frame_age_s must be > 0 seconds, or None to disable the age check")
        self._fifo_depth = max(1, int(lossless_depth))
        self._on_drop = on_drop
        self._now = now_fn

        self._cv = threading.Condition(threading.Lock())
        self._closed = False
        # latest
        self._kf: Deque[FramePacket] = collections.deque()
        self._slot: Optional[FramePacket] = None
        # lossless
        self._fifo: Deque[FramePacket] = collections.deque()
        # counters (monotonically increasing; read under the lock)
        self._c: Dict[str, int] = {
            "admitted_kf": 0, "admitted_nonkf": 0,
            "nonkf_superseded": 0, "kf_dropped": 0, "kf_lane_full": 0, "age_dropped": 0,
            "blocked_puts": 0, "closed_puts": 0,
        }
        self._blocked_s: float = 0.0
        self._max_depth_seen: int = 0

    # ── identity ────────────────────────────────────────────────────────

    @property
    def policy(self) -> str:
        return self._policy

    @property
    def maxsize(self) -> int:
        """Capacity in frames as the legacy consumers understand it: the FIFO
        depth under lossless; keyframe lane + the one slot under latest."""
        return self._fifo_depth if self._policy == POLICY_LOSSLESS else self._kf_depth + 1

    def set_on_drop(self, cb: Optional[DropCallback]) -> None:
        """Install the lane-side drop callback (the runners build the queue
        before the event log exists)."""
        self._on_drop = cb

    # ── producer side ───────────────────────────────────────────────────

    def refusal(self, is_keyframe: bool, keyframe_origin: Optional[str] = None) -> Optional[str]:
        """Would put() refuse this frame right now? Returns the drop reason or
        None. Receivers call this BEFORE decoding pixels (admit-before-decode).
        latest: only a SOURCE keyframe meeting a full keyframe lane under
        overflow ``reject`` is refused (non-keyframes supersede the slot,
        minted keyframes drop the oldest). lossless: never (it blocks).
        After close(): always ``closed``."""
        with self._cv:
            if self._closed:
                return DROP_CLOSED
            if self._policy == POLICY_LOSSLESS or not is_keyframe:
                return None
            if (keyframe_origin == KF_SOURCE and self._kf_overflow == OVERFLOW_REJECT
                    and len(self._kf) >= self._kf_depth):
                return DROP_KF_LANE_FULL
            return None

    def full(self) -> bool:
        """Compat: True when a non-keyframe put would not be admitted right
        now. Never under ``latest`` (the slot supersedes); under ``lossless``
        the FIFO is at capacity (a put would block, not fail)."""
        with self._cv:
            if self._policy == POLICY_LATEST:
                return False
            return len(self._fifo) >= self._fifo_depth

    def put(self, pkt: FramePacket, block: bool = False, timeout: Optional[float] = None) -> bool:
        """Admit a frame. Returns False only when the frame was refused (reason
        in ``pkt.ingest.drop_reason``: kf_lane_full | closed); the caller
        traces refusals. ``block`` / ``timeout`` are ignored: lossless blocks
        until space or close(), latest never blocks."""
        if self._policy == POLICY_LOSSLESS:
            return self._put_lossless(pkt)
        return self._put_latest(pkt)

    def _put_latest(self, pkt: FramePacket) -> bool:
        meta = ensure_meta(pkt)
        now = self._now()
        drops: List[Tuple[Any, str]] = []
        with self._cv:
            if self._closed:
                self._c["closed_puts"] += 1
                _mark(meta, DROP_CLOSED)
                return False
            if bool(getattr(pkt, "is_keyframe", False)):
                if len(self._kf) >= self._kf_depth:
                    origin = getattr(meta, "keyframe_origin", None)
                    if origin == KF_SOURCE and self._kf_overflow == OVERFLOW_REJECT:
                        self._c["kf_lane_full"] += 1
                        _mark(meta, DROP_KF_LANE_FULL)
                        return False
                    old = self._kf.popleft()
                    self._c["kf_dropped"] += 1
                    _mark(getattr(old, "ingest", None), DROP_KF_DROPPED)
                    drops.append((old, DROP_KF_DROPPED))
                self._admit_locked(pkt, meta, now, LANE_KEYFRAME, len(self._kf))
                self._kf.append(pkt)
                self._c["admitted_kf"] += 1
            else:
                depth_before = 0
                if self._slot is not None:
                    old = self._slot
                    self._c["nonkf_superseded"] += 1
                    _mark(getattr(old, "ingest", None), DROP_SUPERSEDED)
                    drops.append((old, DROP_SUPERSEDED))
                    depth_before = 1
                self._admit_locked(pkt, meta, now, LANE_LATEST, depth_before)
                self._slot = pkt
                self._c["admitted_nonkf"] += 1
            self._max_depth_seen = max(self._max_depth_seen, self._qsize_locked())
            self._cv.notify()
        self._fire(drops)
        return True

    def _put_lossless(self, pkt: FramePacket) -> bool:
        meta = ensure_meta(pkt)
        t0 = self._now()
        waited = False
        with self._cv:
            while not self._closed and len(self._fifo) >= self._fifo_depth:
                waited = True
                self._cv.wait(0.1)
            if waited:                      # accounted whether admitted or aborted by close()
                self._c["blocked_puts"] += 1
                self._blocked_s += self._now() - t0
            if self._closed:
                self._c["closed_puts"] += 1
                _mark(meta, DROP_CLOSED)
                return False
            self._admit_locked(pkt, meta, self._now(), LANE_FIFO, len(self._fifo))
            self._fifo.append(pkt)
            if bool(getattr(pkt, "is_keyframe", False)):
                self._c["admitted_kf"] += 1
            else:
                self._c["admitted_nonkf"] += 1
            self._max_depth_seen = max(self._max_depth_seen, len(self._fifo))
            self._cv.notify()
        return True

    @staticmethod
    def _admit_locked(pkt: Any, meta: Optional[IngestMeta], now: float, lane: str, depth_before: int) -> None:
        if meta is None:
            return
        meta.lane = lane
        meta.lane_depth_at_admit = depth_before
        meta.admitted_mono = now
        meta.admitted_sensor_ns = _sensor_ns(pkt)
        meta.drop_reason = None

    # ── consumer side ───────────────────────────────────────────────────

    def get(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        """Next frame: keyframe lane first, then the slot (``latest``); FIFO
        order (``lossless``). ``timeout`` None / 0 = non-blocking, like the
        legacy queue. Aged-out slot frames are dropped here, never returned.
        get() keeps draining after close()."""
        wait_s = 0.0 if timeout is None else max(0.0, float(timeout))
        # The wait deadline is REAL time (Condition.wait sleeps in real time);
        # now_fn is only for admitted_mono / the age check, so an injected or
        # frozen clock cannot make a timed get() spin forever.
        deadline = time.monotonic() + wait_s
        drops: List[Tuple[Any, str]] = []
        try:
            with self._cv:
                while True:
                    pkt = self._pop_locked(drops)
                    if pkt is not None:
                        return pkt
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cv.wait(remaining)
        finally:
            self._fire(drops)

    def _pop_locked(self, drops: List[Tuple[Any, str]]) -> Optional[FramePacket]:
        if self._policy == POLICY_LOSSLESS:
            if self._fifo:
                pkt = self._fifo.popleft()
                self._cv.notify_all()      # wake a blocked producer
                return pkt
            return None
        if self._kf:
            return self._kf.popleft()      # keyframes are never age-dropped
        if self._slot is not None:
            pkt, self._slot = self._slot, None
            if self._max_age_s is not None:
                adm = getattr(getattr(pkt, "ingest", None), "admitted_mono", None)
                if adm is not None and (self._now() - float(adm)) > self._max_age_s:
                    self._c["age_dropped"] += 1
                    _mark(getattr(pkt, "ingest", None), DROP_AGE)
                    drops.append((pkt, DROP_AGE))
                    return None
            return pkt
        return None

    def close(self) -> None:
        """Shutdown hook: wake producers blocked in a lossless put() and make
        every later put() return False (``closed``). get() still drains."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    # ── observation ─────────────────────────────────────────────────────

    def qsize(self) -> int:
        with self._cv:
            return self._qsize_locked()

    def _qsize_locked(self) -> int:
        if self._policy == POLICY_LOSSLESS:
            return len(self._fifo)
        return len(self._kf) + (1 if self._slot is not None else 0)

    def _depth_locked(self) -> Dict[str, int]:
        if self._policy == POLICY_LOSSLESS:
            return {LANE_FIFO: len(self._fifo)}
        return {LANE_KEYFRAME: len(self._kf), LANE_LATEST: 1 if self._slot is not None else 0}

    def depth(self) -> Dict[str, int]:
        """Per-lane occupancy."""
        with self._cv:
            return self._depth_locked()

    def backlog_signal(self) -> Dict[str, Any]:
        """Inputs for the watchdog's sustained ``backlogged`` rule (the rule
        itself lives in rtsm/core/watchdog.py): ``lane_full`` = the keyframe
        lane (latest) / the FIFO (lossless) is at capacity right now;
        ``age_dropped`` = cumulative non-keyframe age drops. Supersession is
        deliberately NOT a signal: it is the designed steady state."""
        with self._cv:
            return {"lane_full": self._lane_full_locked(), "age_dropped": int(self._c["age_dropped"]),
                    "depth": self._depth_locked()}

    def _lane_full_locked(self) -> bool:
        """The bounded lane (keyframe lane under ``latest``, the FIFO under
        ``lossless``) is at capacity right now. Must hold the lock."""
        if self._policy == POLICY_LOSSLESS:
            return len(self._fifo) >= self._fifo_depth
        return len(self._kf) >= self._kf_depth

    def stats(self) -> Dict[str, Any]:
        """Snapshot served as ``/stats.ingest_lanes`` and ``/healthz.ingest``
        (P1 task 5): policy, capacity, per-lane depth, ``lane_full`` (the
        bounded lane is at capacity at read time), ``max_depth_seen``,
        ``blocked_s``, ``closed`` and the eight cumulative counters. Under
        ``lossless`` a full lane is a blocking FIFO's steady state, not a
        fault — read ``blocked_s``; the sustained ``backlogged`` verdict stays
        with the watchdog (live only)."""
        with self._cv:
            out: Dict[str, Any] = {
                "policy": self._policy,
                "maxsize": self.maxsize,
                "depth": self._depth_locked(),
                "lane_full": self._lane_full_locked(),
                "max_depth_seen": self._max_depth_seen,
                "blocked_s": round(self._blocked_s, 3),
                "closed": self._closed,
            }
            out.update(self._c)
            if self._policy == POLICY_LATEST:
                out["keyframe_lane_depth"] = self._kf_depth
                out["keyframe_lane_overflow"] = self._kf_overflow
                out["max_frame_age_s"] = self._max_age_s
            else:
                out["lossless_depth"] = self._fifo_depth
            return out

    # ── helpers ─────────────────────────────────────────────────────────

    def _fire(self, drops: List[Tuple[Any, str]]) -> None:
        cb = self._on_drop
        if cb is None or not drops:
            return
        for pkt, reason in drops:
            try:
                cb(pkt, reason)
            except Exception:
                logger.debug("[ingest] on_drop callback failed", exc_info=True)


def _mark(meta: Optional[IngestMeta], reason: str) -> None:
    if meta is not None:
        meta.drop_reason = reason


def _sensor_ns(pkt: Any) -> Optional[int]:
    try:
        v = pkt.time.t_sensor_ns
        return None if v is None else int(v)
    except Exception:
        return None


def lane_drop_handler(event_sink: Optional[Callable[[Any], None]], latency_analytics: Any = None,
                      queue: Any = None) -> DropCallback:
    """The runner's ``on_drop`` callback: feeds the analytics counters
    (kf_dropped -> queue drops; superseded -> superseded; age -> age drops)
    and, when a frame-flow trace sink is set, writes one receiver-kind line
    with ``source: "lanes"`` carrying the packet's ids (frame_seq,
    t_sensor_ns, is_keyframe, rx_seq, lane, depth_valid_frac). Never raises."""
    from rtsm.evaluation.event_log import RX_DROPPED, SOURCE_LANES, ReceiverEvent

    def on_drop(pkt: Any, reason: str) -> None:
        if latency_analytics is not None:
            try:
                if reason == DROP_AGE:
                    latency_analytics.record_age_drop()
                elif reason == DROP_SUPERSEDED:
                    rec = getattr(latency_analytics, "record_superseded", None)
                    if rec is not None:
                        rec()
                else:
                    latency_analytics.record_queue_drop()
            except Exception:
                logger.debug("[ingest] analytics counter failed", exc_info=True)
        if event_sink is None:
            return
        meta = getattr(pkt, "ingest", None)
        tb = getattr(pkt, "time", None)
        try:
            seq = getattr(tb, "seq", None)
            ts = getattr(tb, "t_sensor_ns", None)
            event_sink(ReceiverEvent(
                timestamp=time.monotonic(),
                source=SOURCE_LANES,
                decision=RX_DROPPED,
                reason=reason,
                frame_seq=(int(seq) if seq is not None else None),
                t_sensor_ns=(int(ts) if ts is not None else None),
                is_keyframe=bool(getattr(pkt, "is_keyframe", False)),
                frame_count=None,
                queue_depth=(int(queue.qsize()) if queue is not None else None),
                depth_valid_frac=getattr(meta, "depth_valid_frac", None),
                lane=getattr(meta, "lane", None),
                rx_seq=getattr(meta, "rx_seq", None),
            ))
        except Exception:
            logger.debug("[ingest] lane drop trace failed", exc_info=True)

    return on_drop


def make_ingest_queue(
    lanes: LaneConfig,
    *,
    on_drop: Optional[DropCallback] = None,
    now_fn: Callable[[], float] = time.monotonic,
):
    """Build the runner's ingest queue from a validated LaneConfig: the legacy
    IngestQueue (policy legacy, untouched) or IngestLanes (latest / lossless).
    The resolved policy is on ``.policy`` in both cases."""
    if lanes.policy == POLICY_LEGACY:
        return IngestQueue(maxsize=LEGACY_DEPTH)
    return IngestLanes(
        lanes.policy,
        keyframe_lane_depth=lanes.keyframe_lane_depth,
        keyframe_lane_overflow=lanes.keyframe_lane_overflow,
        max_frame_age_s=lanes.max_frame_age_s,
        lossless_depth=lanes.lossless_depth,
        on_drop=on_drop,
        now_fn=now_fn,
    )
