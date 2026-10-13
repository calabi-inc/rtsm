"""Session registry: a SLAM boundary freezes the live session instead of
discarding it or piling the next map into it.

What a *session* is here: one map frame of the pose source. The ingest
front-end mints a ``frame_epoch`` whenever the source's session id changes
(a Lens hello, a bag, a ROS 2 run, ``rtabmap-map-<id>`` on ZeroMQ) or when a
receiver infers a restart (a large backwards stamp jump). The registry turns
that into a *boundary*: the live session is snapshotted (objects, keyframe
clouds, pose history, vectors, crops, timeline), the live stores are cleared
the way ``POST /reset`` clears them (minus the ingest clock and the pairing
window, which already belong to the next session), and a new live session
opens. A pose-correction batch (any lane: the Lens ``pose_corrections`` text
message, the ZeroMQ adapter's coalesced ``kf_pose_update`` burst, a plug-in) is
*refinement*, never a boundary: it is routed by keyframe id to the session
that owns the keyframes, live or frozen.

One lock. ``mutate`` is held for the whole of a freeze and for the whole of a
correction batch, so a batch lands entirely before or entirely after a freeze.
Store locks (``WorkingMemory._lock``, ``KeyframeRegistry._lock``, the vector
store's) are only ever taken inside it: one lock order, no deadlock.
Persistence runs after the lock is released (a writer thread, or inline when
``persist_async`` is False, which the tests use).

Keys a correction may use, all resolved here: the pipeline's ``ws_<seq>``; a
source keyframe id through the session's alias table (learnt from
``FramePacket.kf_id``); ``ts:<sensor ns>`` matched to a keyframe stamp within
``sessions.alias_stamp_tol_ms``. An empty batch is *announced-only* (a SLAM
that reports a map change without delivering poses): a timeline event,
nothing moves.

What keeps the list finite: a session that ends with fewer than
``min_keyframes`` keyframes and no objects is discarded, not frozen; a frozen
session that a correction ties to the live map is marked merged (display and
registry only; its objects stay its own); retention removes the oldest
sessions past a count and an age, never the live chain, a pinned session, or
one served recently.

Nothing in this module knows which SLAM or transport produced a frame.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = 1
KINDS = ("loop_closure", "relocalized_in_prior_map", "reset", "unknown")
END_REASONS = ("session_id", "restart_inferred", "map_reset", "manual", "process_exit")
TIMELINE_KINDS = ("start", "reconnect", "tracking_gap", "refine", "relocalized", "merge", "unmerge", "freeze")
COUNTER_KEYS = ("frames", "keyframes", "objects", "confirmed", "refinements", "corrections_applied",
                "objects_demoted_by_corrections", "tracking_gaps", "announced_only")
STAMP_KEY_PREFIX = "ts:"
TMP_SUFFIX = ".tmp"
GC_SUFFIX = ".gc"
LIVE_HISTORY_MAX = 200_000          # frames with a pose kept for ownership resolution in the live session


# ───────────────────────────── records ─────────────────────────────

@dataclass
class TimelineEvent:
    t_wall_utc_s: float
    t_sensor_ns: Optional[int]
    kind: str                                   # one of TIMELINE_KINDS
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyframeSnapshot:
    mesh_id: str
    timestamp_ns: int
    pose: np.ndarray                            # (4, 4) float32 T_wc
    n_points: int
    cloud_ref: str


@dataclass
class SessionRecord:
    """Live and frozen sessions share this shape."""
    key: str
    session_id: Optional[str]
    epoch: int
    source: str
    started_wall_utc: float
    started_sensor_ns: Optional[int] = None
    ended_wall_utc: Optional[float] = None
    end_reason: Optional[str] = None            # one of END_REASONS once frozen
    aligned_to: Optional[Dict[str, Any]] = None  # {session_key, kind, t_wall_utc_s}
    merged_into: Optional[str] = None
    merged_from: List[str] = field(default_factory=list)
    pinned: bool = False
    last_served_wall_utc: Optional[float] = None
    timeline: List[TimelineEvent] = field(default_factory=list)
    counters: Dict[str, int] = field(default_factory=lambda: {k: 0 for k in COUNTER_KEYS})
    persisted: bool = False
    persist_error: Optional[str] = None
    kf_alias: Dict[str, str] = field(default_factory=dict)        # source keyframe id -> ws_<seq>
    discarded_predecessor: Optional[Dict[str, Any]] = None
    # Ownership bookkeeping (complete for the session; the viz server's own
    # history is capped and serves its delta math only).
    pose_history: Dict[str, np.ndarray] = field(default_factory=dict)   # ws_<seq> -> (4, 4) T_wc
    alias_stamp: Dict[str, int] = field(default_factory=dict)           # ws_<seq> -> t_sensor_ns (keyframes)


@dataclass
class FrozenSession(SessionRecord):
    objects: List[Dict[str, Any]] = field(default_factory=list)
    frame_to_objects: Dict[str, List[str]] = field(default_factory=dict)
    keyframes: Dict[str, KeyframeSnapshot] = field(default_factory=dict)
    clouds: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None   # mesh_id -> (positions f32 Nx3, colors u8 Nx3)
    vectors: Optional[Tuple[List[str], np.ndarray]] = None
    crops: Optional[Dict[str, List[bytes]]] = None                       # oid -> JPEGs (until persisted)
    crops_dir: Optional[Path] = None
    dirty: bool = False
    on_disk: bool = False
    payload_loaded: bool = True                 # objects / keyframes / clouds / vectors in memory


@dataclass
class Boundary:
    reason: str
    frozen_key: Optional[str]
    live_key: str
    discarded: bool = False


@dataclass
class LiveComponents:
    """The live stores a freeze snapshots and clears (any may be None; tests pass fakes)."""
    wm: Any = None                              # snapshot_for_session(), clear(), apply_pose_corrections(), pose_demote_thresh_m
    vectors: Any = None                         # export(), clear()
    sweep_cache: Any = None                     # clear()
    vis_server: Any = None                      # freeze_live(next_key) -> LiveVizState


@dataclass
class LiveVizState:
    """What ``VisualizationServer.freeze_live`` hands back."""
    records: List[Any] = field(default_factory=list)       # KeyframeRecord: mesh_id, timestamp_ns, positions, colors, pose
    pose_history: Dict[str, np.ndarray] = field(default_factory=dict)
    kf_alias: Dict[str, str] = field(default_factory=dict)
    alias_stamp: Dict[str, int] = field(default_factory=dict)


@dataclass
class SessionSnapshot:
    """What ``WorkingMemory.snapshot_for_session`` hands back."""
    objects: List[Dict[str, Any]] = field(default_factory=list)
    frame_to_objects: Dict[str, List[str]] = field(default_factory=dict)
    crops: Dict[str, List[bytes]] = field(default_factory=dict)


@dataclass
class LiveApplyResult:
    """What the live-apply callable returns: the per-frame deltas it computed."""
    frame_deltas: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)  # id -> (old_cam_pos, dR, dt)
    max_shift_m: float = 0.0
    objects_moved: int = 0
    objects_demoted: int = 0


@dataclass(frozen=True)
class SessionsConfig:
    enable: bool = True
    dir: str = "sessions"
    keep_in_memory: int = 3
    cloud_max_points: int = 150_000
    persist_crops: bool = True
    persist_on_freeze: bool = True
    boundary_on_clock_rebase: bool = True
    tracking_gap_s: float = 2.0
    min_keyframes: int = 1
    merge_on_link: bool = True
    alias_stamp_tol_ms: float = 30.0
    keep_on_disk: int = 20
    max_age_days: float = 30.0
    grace_served_min: float = 10.0

    @classmethod
    def from_cfg(cls, cfg: Any) -> "SessionsConfig":
        sc = (cfg or {}).get("sessions") or {}
        if not isinstance(sc, dict):
            raise ValueError(f"sessions: must be a mapping of settings; got {sc!r}")
        rt = sc.get("retention") or {}
        return cls(
            enable=bool(sc.get("enable", True)),
            dir=str(sc.get("dir", "sessions")),
            keep_in_memory=max(0, int(sc.get("keep_in_memory", 3))),
            cloud_max_points=max(0, int(sc.get("cloud_max_points", 150_000))),
            persist_crops=bool(sc.get("persist_crops", True)),
            persist_on_freeze=bool(sc.get("persist_on_freeze", True)),
            boundary_on_clock_rebase=bool(sc.get("boundary_on_clock_rebase", True)),
            tracking_gap_s=float(sc.get("tracking_gap_s", 2.0)),
            min_keyframes=max(0, int(sc.get("min_keyframes", 1))),
            merge_on_link=bool(sc.get("merge_on_link", True)),
            alias_stamp_tol_ms=float(sc.get("alias_stamp_tol_ms", 30.0)),
            keep_on_disk=max(0, int(rt.get("keep_on_disk", 20))),
            max_age_days=max(0.0, float(rt.get("max_age_days", 30.0))),
            grace_served_min=max(0.0, float(rt.get("grace_served_min", 10.0))),
        )


# ───────────────────────────── helpers ─────────────────────────────

def _json_default(o: Any) -> Any:
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bytes):
        return None                              # crops never go through JSON
    if isinstance(o, (set, frozenset, tuple)):
        return sorted(o) if isinstance(o, (set, frozenset)) else list(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def packet_frame_id(pkt: Any) -> Optional[str]:
    """The frame id the working memory, the viz registry and the session registry
    share: ``ws_<seq>`` from the source's seq, else ``rx_<rx_seq>`` from the
    receiver's attempt count (ZeroMQ), else None."""
    tm = getattr(pkt, "time", None)
    seq = getattr(tm, "seq", None) if tm is not None else None
    if seq is not None:
        return f"ws_{seq}"
    ing = getattr(pkt, "ingest", None)
    rx = getattr(ing, "rx_seq", None) if ing is not None else None
    if rx is not None:
        return f"rx_{rx}"
    return None


def _pose4(p: Any) -> Optional[np.ndarray]:
    if p is None:
        return None
    a = np.asarray(p, dtype=np.float32)
    if a.shape == (16,):
        a = a.reshape(4, 4)
    if a.shape != (4, 4):
        return None
    return a.copy()


def _copy_object(o: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in o.items():
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
        elif isinstance(v, (list, dict)):
            out[k] = json.loads(json.dumps(v, default=_json_default)) if k != "image_crops" else list(v)
        else:
            out[k] = v
    return out


def subsample_cloud(mesh_id: str, positions: np.ndarray, colors: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform subsample seeded by the mesh id: the same cloud selects the same points every time."""
    n = int(positions.shape[0])
    if max_points <= 0 or n <= max_points:
        return positions.copy(), colors.copy()
    rng = np.random.default_rng(zlib.crc32(mesh_id.encode("utf-8")))
    idx = np.sort(rng.choice(n, size=max_points, replace=False))
    return positions[idx].copy(), colors[idx].copy()


def frame_deltas_from_history(history: Dict[str, np.ndarray], corrections: Dict[str, np.ndarray]
                              ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-frame rigid deltas, exactly as the viz server computes them:
    delta_T = T_new @ inv(T_old); p_new = dR @ p_old + dt."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for fid, new_pose in corrections.items():
        old = history.get(fid)
        new = _pose4(new_pose)
        if old is None or new is None:
            continue
        try:
            delta = new.astype(np.float64) @ np.linalg.inv(old.astype(np.float64))
        except np.linalg.LinAlgError:
            continue
        out[fid] = (old[:3, 3].astype(np.float32).copy(), delta[:3, :3].astype(np.float32), delta[:3, 3].astype(np.float32))
    return out


# ───────────────────────────── the registry ─────────────────────────────

class SessionRegistry:
    def __init__(self, cfg: Any, *, live: Optional[LiveComponents] = None,
                 event_sink: Optional[Callable[[Any], None]] = None, source: str = "unknown",
                 base_dir: Optional[str] = None, now: Optional[Callable[[], float]] = None,
                 persist_async: bool = True) -> None:
        self.cfg = SessionsConfig.from_cfg(cfg)
        self.enabled = self.cfg.enable
        self.live_components = live or LiveComponents()
        self.event_sink = event_sink
        self.source = str(source)
        self._now = now or time.time
        d = Path(self.cfg.dir)
        self.dir: Path = d if d.is_absolute() else Path(base_dir or os.getcwd()) / d
        self.mutate = threading.RLock()
        self.live: Optional[SessionRecord] = None
        self.frozen: List[FrozenSession] = []           # most recent first
        self.counters: Dict[str, int] = {
            "discarded": 0, "gc_removed": 0, "corrections_unrouted": 0, "corrections_unaliased": 0,
            "merge_ambiguous": 0, "announced_only": 0, "boundaries": 0,
        }
        self.last_boundary: Optional[Dict[str, Any]] = None
        self._last_rebases = 0
        self._pending_map_reset = False
        self._last_admitted_ns: Optional[int] = None
        self.persist_async = bool(persist_async)
        # Viewer hooks (the visualization server sets them): on_change after any
        # change of the session list or a record's summary; on_refined after a
        # frozen session's keyframes / objects moved (key, poses, moved objects).
        self.on_change: Optional[Callable[[], None]] = None
        self.on_refined: Optional[Callable[[str, Dict[str, List[float]], List[Dict[str, Any]]], None]] = None
        self._q: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._writer: Optional[threading.Thread] = None
        self._busy = False

    # ── pipeline thread ──

    def observe_frame(self, pkt: Any, *, clock_rebases: int = 0) -> Optional[Boundary]:
        """Once per dequeued packet: a boundary before this packet, or a tracking gap."""
        if not self.enabled or pkt is None:
            return None
        epoch_raw = getattr(pkt, "frame_epoch", None)
        sid = getattr(pkt, "session_id", None)
        tm = getattr(pkt, "time", None)
        ts = getattr(tm, "t_sensor_ns", None) if tm is not None else None
        ts = int(ts) if ts is not None else None
        with self.mutate:
            if self.live is None:
                epoch = int(epoch_raw) if epoch_raw is not None else 0
                self._open_live(epoch, sid, ts)
                self._last_rebases = int(clock_rebases)
                self._track(pkt, ts)
                return None
            epoch = int(epoch_raw) if epoch_raw is not None else self.live.epoch
            reason: Optional[str] = None
            if epoch != self.live.epoch:
                reason = "session_id" if sid != self.live.session_id else "restart_inferred"
            elif self.cfg.boundary_on_clock_rebase and int(clock_rebases) > self._last_rebases:
                reason = "restart_inferred"
            elif self._pending_map_reset:
                reason = "map_reset"
            self._last_rebases = int(clock_rebases)
            if reason is None:
                self._tracking_gap(ts)
                self._track(pkt, ts)
                return None
            self._pending_map_reset = False
            frozen = self._freeze_locked(reason, next_epoch=epoch, next_session_id=sid, t_sensor_ns=ts)
            self._track(pkt, ts)
            assert self.live is not None
            return Boundary(reason, frozen.key if frozen is not None else None, self.live.key, discarded=frozen is None)

    def on_session_change(self, session_id: Optional[str], epoch: int) -> None:
        """Receiver thread: a same-id, same-epoch call is a reconnect. Boundaries are
        decided on the pipeline thread from the packets, never here."""
        if not self.enabled:
            return
        with self.mutate:
            live = self.live
            if live is not None and session_id == live.session_id and int(epoch) == live.epoch:
                self._event(live, "reconnect", None, {"session_id": session_id})

    def on_process_exit(self) -> None:
        """Freeze the live session with reason process_exit, persist synchronously, open nothing."""
        if not self.enabled:
            return
        with self.mutate:
            if self.live is None:
                return
            prev_async = self.persist_async
            self.persist_async = False
            try:
                self._freeze_locked("process_exit", next_epoch=self.live.epoch, next_session_id=self.live.session_id,
                                    t_sensor_ns=self._last_admitted_ns, open_next=False)
            finally:
                self.persist_async = prev_async

    # ── API thread ──

    def end_live(self, reason: str = "manual") -> Optional[Boundary]:
        """``POST /sessions/end`` and the default ``POST /reset``: freeze, reopen with the same identity."""
        if not self.enabled:
            return None
        with self.mutate:
            if self.live is None:
                return None
            epoch, sid = self.live.epoch, self.live.session_id
            frozen = self._freeze_locked(reason, next_epoch=epoch, next_session_id=sid, t_sensor_ns=self._last_admitted_ns)
            assert self.live is not None
            return Boundary(reason, frozen.key if frozen is not None else None, self.live.key, discarded=frozen is None)

    def discard_live(self) -> Optional[str]:
        """``POST /reset`` with discard: reopen the live record (same identity), drop its links."""
        if not self.enabled:
            return None
        with self.mutate:
            if self.live is None:
                return None
            old = self.live
            for k in old.merged_from:
                rec = self._find(k)
                if rec is not None:
                    rec.merged_into = None
            self._open_live(old.epoch, old.session_id, self._last_admitted_ns)
            return self.live.key

    def freeze(self, reason: str = "manual") -> Optional[FrozenSession]:
        if not self.enabled:
            return None
        with self.mutate:
            if self.live is None:
                return None
            return self._freeze_locked(reason, next_epoch=self.live.epoch, next_session_id=self.live.session_id,
                                       t_sensor_ns=self._last_admitted_ns)

    # ── corrections (receiver thread, through the viz server) ──

    def apply_corrections(self, corrections: Dict[str, Any], *, kind: str = "loop_closure",
                          session_id: Optional[str] = None, map_ref: Optional[str] = None,
                          live_apply: Optional[Callable[[Dict[str, np.ndarray]], LiveApplyResult]] = None,
                          ) -> Dict[str, Any]:
        """Route a batch by keyframe id and apply it. ``live_apply`` is the viz server's
        live path (WM, TSDF, registry, broadcast); the registry does the frozen part."""
        corrections = dict(corrections or {})
        if not self.enabled:
            res = live_apply(corrections) if (live_apply is not None and corrections) else None
            return {"enabled": False, "live": len(corrections), "frozen": {}, "unrouted": [], "unaliased": [],
                    "merged": None, "result": res}
        kind = kind if kind in KINDS else "unknown"
        with self.mutate:
            live = self.live
            summary: Dict[str, Any] = {"enabled": True, "kind": kind, "live": 0, "frozen": {}, "unrouted": [],
                                       "unaliased": [], "merged": None, "announced_only": False}
            resolved, unaliased, unrouted = self._resolve_keys(corrections)
            summary["unaliased"] = unaliased
            summary["unrouted"] = unrouted
            self.counters["corrections_unaliased"] += len(unaliased)
            self.counters["corrections_unrouted"] += len(unrouted)
            if kind == "reset":
                self._pending_map_reset = True
                if live is not None:
                    self._event(live, "refine", None, {"kind": kind, "n_keyframes": 0, "pending_boundary": True, "map_ref": map_ref})
                return summary
            if not resolved:
                summary["announced_only"] = True
                self.counters["announced_only"] += 1
                if live is not None:
                    live.counters["announced_only"] += 1
                    self._event(live, "refine", None, {"kind": kind, "n_keyframes": 0, "announced_only": True, "map_ref": map_ref})
                    self._ledger("refine", live, {"announced_only": True, "kind": kind})
            # ownership
            live_part: Dict[str, np.ndarray] = {}
            frozen_parts: Dict[str, Dict[str, np.ndarray]] = {}
            hinted = self._find_by_session_id(session_id) if session_id else None
            for fid, pose in resolved.items():
                if live is not None and fid in live.pose_history:
                    live_part[fid] = pose
                    continue
                owner = None
                if hinted is not None and fid in hinted.pose_history:
                    owner = hinted
                else:
                    for rec in self.frozen:
                        if fid in rec.pose_history:
                            owner = rec
                            break
                if owner is None:
                    summary["unrouted"].append(fid)
                    self.counters["corrections_unrouted"] += 1
                    continue
                frozen_parts.setdefault(owner.key, {})[fid] = pose
            # live part
            if live_part and live is not None:
                if live_apply is not None:
                    res = live_apply(live_part) or LiveApplyResult()
                else:
                    res = LiveApplyResult(frame_deltas=frame_deltas_from_history(live.pose_history, live_part))
                max_shift = float(res.max_shift_m)
                if not max_shift:
                    for fid, (old_pos, _dR, _dt) in res.frame_deltas.items():
                        new = _pose4(live_part[fid])
                        if new is not None:
                            max_shift = max(max_shift, float(np.linalg.norm(new[:3, 3] - old_pos)))
                for fid, pose in live_part.items():
                    p = _pose4(pose)
                    if p is not None:
                        live.pose_history[fid] = p
                live.counters["refinements"] += 1
                live.counters["corrections_applied"] += len(live_part)
                live.counters["objects_demoted_by_corrections"] += int(res.objects_demoted)
                self._event(live, "refine", None, {"kind": kind, "n_keyframes": len(live_part), "max_shift_m": round(max_shift, 4),
                                                   "objects_moved": int(res.objects_moved), "objects_demoted": int(res.objects_demoted),
                                                   "map_ref": map_ref})
                self._ledger("refine", live, {"kind": kind, "n_keyframes": len(live_part), "max_shift_m": round(max_shift, 4)})
                summary["live"] = len(live_part)
            # frozen parts
            for key, part in frozen_parts.items():
                rec = self._find(key)
                if rec is None:
                    continue
                n = self._refine_frozen(rec, part, kind, map_ref)
                summary["frozen"][key] = n
            # relocalisation link
            if kind == "relocalized_in_prior_map" and live is not None:
                target = self._find_by_session_id(map_ref) if map_ref else None
                live.aligned_to = {"session_key": (target.key if target is not None else map_ref), "kind": kind,
                                   "t_wall_utc_s": self._now()}
                self._event(live, "relocalized", None, {"map_ref": map_ref, "session_key": live.aligned_to["session_key"]})
            # merge trigger
            if self.cfg.merge_on_link and live is not None:
                target: Optional[FrozenSession] = None
                if kind == "relocalized_in_prior_map":
                    target = self._find_by_session_id(map_ref) if map_ref else None
                    if target is None and len(frozen_parts) == 1:
                        target = self._find(next(iter(frozen_parts)))
                elif live_part and len(frozen_parts) == 1:
                    target = self._find(next(iter(frozen_parts)))
                elif len(frozen_parts) > 1:
                    self.counters["merge_ambiguous"] += 1
                if target is not None and target.merged_into is None and target.key != live.key:
                    self._merge_locked(target, live)
                    summary["merged"] = target.key
            return summary

    # ── merge / pin / delete / serve ──

    def merge(self, key: str, into: str = "live") -> Optional[str]:
        """Returns None on success, else the refusal reason."""
        with self.mutate:
            live = self.live
            if live is None or (into not in ("live", live.key)):
                return "into_must_be_live"
            rec = self._find(key)
            if rec is None:
                return "unknown_session"
            if rec.merged_into is not None:
                return "already_merged"
            self._merge_locked(rec, live)
            return None

    def unmerge(self, key: str) -> Optional[str]:
        with self.mutate:
            rec = self._find(key)
            if rec is None:
                return "unknown_session"
            if rec.merged_into is None:
                return "not_merged"
            succ = self._find(rec.merged_into) or (self.live if self.live and self.live.key == rec.merged_into else None)
            rec.merged_into = None
            if succ is not None and key in succ.merged_from:
                succ.merged_from.remove(key)
                self._event(succ, "unmerge", None, {"session_key": key})
            self._event(rec, "unmerge", None, {"session_key": succ.key if succ else None})
            self._ledger("unmerge", rec, {})
            self._persist_meta(rec)
            self._notify_change()
            return None

    def pin(self, key: str, flag: bool = True) -> Optional[str]:
        with self.mutate:
            rec = self._find(key)
            if rec is None:
                return "unknown_session"
            rec.pinned = bool(flag)
            self._persist_meta(rec)
            self._notify_change()
            return None

    def delete(self, key: str, *, force: bool = False) -> Optional[str]:
        """Apply the retention rules by hand. Returns None on success, else the reason."""
        with self.mutate:
            if self.live is not None and key == self.live.key:
                return "live_session"
            rec = self._find(key)
            if rec is None:
                return "unknown_session"
            if rec.pinned and not force:
                return "pinned"
            if key in self._live_chain():
                return "in_live_chain"
            self._remove_locked(rec, reason="delete")
            return None

    def mark_served(self, key: str) -> None:
        with self.mutate:
            rec = self._find(key)
            if rec is not None:
                rec.last_served_wall_utc = self._now()

    # ── queries ──

    def get(self, key: str) -> Optional[SessionRecord]:
        with self.mutate:
            if self.live is not None and key == self.live.key:
                return self.live
            return self._find(key)

    def list_summaries(self) -> Dict[str, Any]:
        with self.mutate:
            out = []
            if self.live is not None:
                out.append(self.summary(self.live))
            out.extend(self.summary(r) for r in self.frozen)
            return {"live": self.live.key if self.live else None, "sessions": out}

    def summary(self, rec: SessionRecord) -> Dict[str, Any]:
        d = {
            "key": rec.key, "session_id": rec.session_id, "epoch": rec.epoch, "source": rec.source,
            "started_wall_utc": rec.started_wall_utc, "started_sensor_ns": rec.started_sensor_ns,
            "ended_wall_utc": rec.ended_wall_utc, "end_reason": rec.end_reason, "aligned_to": rec.aligned_to,
            "merged_into": rec.merged_into, "merged_from": list(rec.merged_from), "pinned": rec.pinned,
            "last_served_wall_utc": rec.last_served_wall_utc, "counters": dict(rec.counters),
            "persisted": rec.persisted, "persist_error": rec.persist_error, "n_timeline": len(rec.timeline),
            "n_alias": len(rec.kf_alias), "discarded_predecessor": rec.discarded_predecessor,
            "live": self.live is not None and rec.key == self.live.key,
        }
        if isinstance(rec, FrozenSession):
            d["on_disk"] = rec.on_disk
            d["in_memory"] = rec.clouds is not None
            d["n_keyframes"] = len(rec.keyframes) if rec.payload_loaded else rec.counters.get("keyframes", 0)
            d["n_objects"] = len(rec.objects) if rec.payload_loaded else rec.counters.get("objects", 0)
        return d

    def detail(self, key: str) -> Optional[Dict[str, Any]]:
        with self.mutate:
            rec = self.get(key)
            if rec is None:
                return None
            d = self.summary(rec)
            d["timeline"] = [{"t_wall_utc_s": e.t_wall_utc_s, "t_sensor_ns": e.t_sensor_ns, "kind": e.kind, "detail": e.detail}
                             for e in rec.timeline]
            d["retention"] = self._retention_info(rec)
            if isinstance(rec, FrozenSession):
                self._ensure_loaded(rec)
                d["objects"] = [self._object_summary(o) for o in rec.objects]
                d["keyframes"] = [{"mesh_id": k.mesh_id, "timestamp_ns": k.timestamp_ns, "n_points": k.n_points}
                                  for k in rec.keyframes.values()]
            rec.last_served_wall_utc = self._now()
            return d

    def health(self) -> Dict[str, Any]:
        with self.mutate:
            return {
                "enabled": self.enabled,
                "live": self.live.key if self.live else None,
                "n_frozen": len(self.frozen),
                "n_on_disk": sum(1 for r in self.frozen if r.on_disk),
                "n_merged": sum(1 for r in self.frozen if r.merged_into),
                **{k: int(v) for k, v in self.counters.items()},
                "last_boundary": self.last_boundary,
            }

    # ── persistence and retention ──

    def load_from_disk(self) -> int:
        """List the session folders (metadata + timeline only); stray tmp / gc folders are removed."""
        if not self.enabled or not self.dir.exists():
            return 0
        n = 0
        with self.mutate:
            known = {r.key for r in self.frozen}
            for child in sorted(self.dir.iterdir()):
                if not child.is_dir():
                    continue
                if child.name.endswith(TMP_SUFFIX) or child.name.endswith(GC_SUFFIX):
                    shutil.rmtree(child, ignore_errors=True)
                    continue
                meta = child / "meta.json"
                if not meta.is_file() or child.name in known:
                    continue
                try:
                    rec = self._load_meta(child)
                except Exception as e:  # noqa: BLE001
                    logger.warning("[sessions] unreadable session folder %s: %s", child, e)
                    continue
                self.frozen.append(rec)
                n += 1
            self.frozen.sort(key=lambda r: r.started_wall_utc, reverse=True)
        self.gc()
        if n:
            self._notify_change()
        return n

    def gc(self) -> int:
        """Retention: remove the oldest frozen sessions on disk past the count / age caps,
        never the live chain, a pinned one, or one served within the grace window."""
        if not self.enabled:
            return 0
        removed = 0
        with self.mutate:
            cfg = self.cfg
            if cfg.keep_on_disk <= 0 and cfg.max_age_days <= 0:
                return 0
            now = self._now()
            protected = self._live_chain()
            candidates = [r for r in self.frozen if r.on_disk and not r.pinned and r.key not in protected
                          and not (r.last_served_wall_utc is not None and now - r.last_served_wall_utc < cfg.grace_served_min * 60.0)]
            candidates.sort(key=lambda r: r.started_wall_utc)
            n_on_disk = sum(1 for r in self.frozen if r.on_disk)
            for rec in candidates:
                age_days = (now - (rec.ended_wall_utc or rec.started_wall_utc)) / 86400.0
                over_count = cfg.keep_on_disk > 0 and n_on_disk > cfg.keep_on_disk
                over_age = cfg.max_age_days > 0 and age_days > cfg.max_age_days
                if not (over_count or over_age):
                    continue
                self._remove_locked(rec, reason="gc")
                n_on_disk -= 1
                removed += 1
        return removed

    def wait_idle(self, timeout: float = 10.0) -> bool:
        """Tests and process exit: wait for queued persistence."""
        if self._writer is None:
            return True
        deadline = time.monotonic() + timeout
        while not self._q.empty() or getattr(self, "_busy", False):
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    # ───────────────────────── internals (hold mutate) ─────────────────────────

    def _open_live(self, epoch: int, session_id: Optional[str], t_sensor_ns: Optional[int],
                   discarded: Optional[Dict[str, Any]] = None) -> SessionRecord:
        now = self._now()
        key = self._make_key(now, epoch, session_id)
        rec = SessionRecord(key=key, session_id=session_id, epoch=int(epoch), source=self.source,
                            started_wall_utc=now, started_sensor_ns=t_sensor_ns, discarded_predecessor=discarded)
        self.live = rec
        self._last_admitted_ns = None
        vs = self.live_components.vis_server
        if vs is not None and hasattr(vs, "set_live_session"):
            try:
                vs.set_live_session(rec.key, rec.session_id)
            except Exception as e:  # noqa: BLE001
                logger.error("[sessions] viz server refused the live key: %s", e)
        self._event(rec, "start", t_sensor_ns, {"source": self.source, "discarded_predecessor": discarded})
        self._ledger("start", rec, {})
        self._notify_change()
        return rec

    def _make_key(self, now: float, epoch: int, session_id: Optional[str]) -> str:
        short = (session_id or "noid")[:8]
        base = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime(now))}_e{int(epoch)}_{short}"
        key, n = base, 1
        existing = {r.key for r in self.frozen} | ({self.live.key} if self.live else set())
        while key in existing or (self.dir / key).exists():
            n += 1
            key = f"{base}_{n}"
        return key

    def _track(self, pkt: Any, ts: Optional[int]) -> None:
        live = self.live
        if live is None:
            return
        live.counters["frames"] += 1
        is_kf = bool(getattr(pkt, "is_keyframe", False))
        if is_kf:
            live.counters["keyframes"] += 1
        fid = packet_frame_id(pkt)
        pose = getattr(pkt, "pose", None)
        if fid is not None and pose is not None and hasattr(pose, "T_wc"):
            if len(live.pose_history) < LIVE_HISTORY_MAX:
                live.pose_history[fid] = np.asarray(pose.T_wc(), dtype=np.float32).copy()
            if is_kf and ts is not None:
                live.alias_stamp[fid] = int(ts)
            kf_id = getattr(pkt, "kf_id", None)
            if kf_id is not None:
                live.kf_alias[str(kf_id)] = fid
        if ts is not None:
            self._last_admitted_ns = ts

    def _tracking_gap(self, ts: Optional[int]) -> None:
        live = self.live
        if live is None or ts is None or self._last_admitted_ns is None:
            return
        gap_s = (int(ts) - int(self._last_admitted_ns)) / 1e9
        if gap_s > self.cfg.tracking_gap_s:
            live.counters["tracking_gaps"] += 1
            self._event(live, "tracking_gap", ts, {"gap_s": round(gap_s, 3), "prev_t_sensor_ns": int(self._last_admitted_ns)})

    def _freeze_locked(self, reason: str, *, next_epoch: int, next_session_id: Optional[str],
                       t_sensor_ns: Optional[int], open_next: bool = True) -> Optional[FrozenSession]:
        live = self.live
        assert live is not None
        lc = self.live_components
        now = self._now()
        # 1. snapshots (store locks nest inside mutate)
        snap = lc.wm.snapshot_for_session() if (lc.wm is not None and hasattr(lc.wm, "snapshot_for_session")) else SessionSnapshot()
        next_key = self._make_key(now, next_epoch, next_session_id) if open_next else None
        viz = lc.vis_server.freeze_live(next_key) if (lc.vis_server is not None and hasattr(lc.vis_server, "freeze_live")) else LiveVizState()
        vec = lc.vectors.export() if (lc.vectors is not None and hasattr(lc.vectors, "export")) else None
        self.counters["boundaries"] += 1
        self.last_boundary = {"reason": reason, "t_wall_utc_s": now, "from": live.key}
        n_keyframes = max(live.counters.get("keyframes", 0), len(viz.records))
        trivial = n_keyframes < self.cfg.min_keyframes and not snap.objects
        frozen: Optional[FrozenSession] = None
        if trivial:
            discarded = {"session_id": live.session_id, "epoch": live.epoch, "frames": live.counters.get("frames", 0),
                         "end_reason": reason, "key": live.key}
            self.counters["discarded"] += 1
            self._ledger("discard", live, {"reason": reason})
            logger.info("[sessions] session %s discarded at boundary (%s): no keyframes, no objects", live.key, reason)
        else:
            frozen = self._build_frozen(live, snap, viz, vec, reason, now, t_sensor_ns)
            self.frozen.insert(0, frozen)
            self._enforce_keep_cap()
            self._ledger("freeze", frozen, {"reason": reason})
            logger.info("[sessions] session %s frozen (%s): %d objects, %d keyframes", frozen.key, reason,
                        len(frozen.objects), len(frozen.keyframes))
            discarded = None
        # 2. clear the live stores (/reset order minus the clock and the pairing window)
        self._clear_live_stores()
        # 3. next live session
        self.live = None
        if open_next:
            self._open_live(next_epoch, next_session_id, t_sensor_ns, discarded=discarded)
        else:
            self._notify_change()
        # 4. persistence after the lock (the caller still holds mutate; the writer thread does not need it)
        if frozen is not None and self.cfg.persist_on_freeze:
            self._schedule(lambda rec=frozen: self._persist_full(rec))
            self._schedule(self.gc)
        return frozen

    def _build_frozen(self, live: SessionRecord, snap: SessionSnapshot, viz: LiveVizState,
                      vec: Optional[Tuple[List[str], np.ndarray]], reason: str, now: float,
                      t_sensor_ns: Optional[int]) -> FrozenSession:
        objects = [_copy_object(o) for o in snap.objects]
        for o in objects:
            o.setdefault("demoted_by_correction", False)
        keyframes: Dict[str, KeyframeSnapshot] = {}
        clouds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        pose_history: Dict[str, np.ndarray] = {k: v.copy() for k, v in live.pose_history.items()}
        for r in viz.records:
            mid = str(getattr(r, "mesh_id"))
            pos = np.asarray(getattr(r, "positions"), dtype=np.float32)
            col = np.asarray(getattr(r, "colors"), dtype=np.uint8)
            pose = _pose4(getattr(r, "pose", None))
            if pose is None:
                pose = pose_history.get(mid, np.eye(4, dtype=np.float32))
            p, c = subsample_cloud(mid, pos, col, self.cfg.cloud_max_points)
            clouds[mid] = (p, c)
            keyframes[mid] = KeyframeSnapshot(mesh_id=mid, timestamp_ns=int(getattr(r, "timestamp_ns", 0) or 0),
                                              pose=pose.copy(), n_points=int(p.shape[0]), cloud_ref=mid)
            pose_history[mid] = pose.copy()
        for k, v in viz.pose_history.items():
            p = _pose4(v)
            if p is not None and k not in pose_history:
                pose_history[k] = p
        alias = dict(live.kf_alias)
        alias.update({str(k): str(v) for k, v in viz.kf_alias.items()})
        stamps = dict(live.alias_stamp)
        stamps.update({str(k): int(v) for k, v in viz.alias_stamp.items()})
        counters = dict(live.counters)
        counters["objects"] = len(objects)
        counters["confirmed"] = sum(1 for o in objects if o.get("confirmed"))
        counters["keyframes"] = max(counters.get("keyframes", 0), len(keyframes))
        frozen = FrozenSession(
            key=live.key, session_id=live.session_id, epoch=live.epoch, source=live.source,
            started_wall_utc=live.started_wall_utc, started_sensor_ns=live.started_sensor_ns,
            ended_wall_utc=now, end_reason=reason, aligned_to=live.aligned_to, merged_into=None,
            merged_from=list(live.merged_from), pinned=live.pinned, last_served_wall_utc=live.last_served_wall_utc,
            timeline=list(live.timeline), counters=counters, kf_alias=alias,
            discarded_predecessor=live.discarded_predecessor, pose_history=pose_history, alias_stamp=stamps,
            objects=objects, frame_to_objects={k: list(v) for k, v in snap.frame_to_objects.items()},
            keyframes=keyframes, clouds=clouds,
            vectors=((list(vec[0]), np.asarray(vec[1], dtype=np.float32).copy()) if vec is not None else None),
            crops={k: list(v) for k, v in (snap.crops or {}).items()} if self.cfg.persist_crops else None,
        )
        for k in frozen.merged_from:
            rec = self._find(k)
            if rec is not None:
                rec.merged_into = frozen.key
        self._event(frozen, "freeze", t_sensor_ns, {"reason": reason, "counters": dict(counters)})
        return frozen

    def _clear_live_stores(self) -> None:
        lc = self.live_components
        for name, obj in (("sweep_cache", lc.sweep_cache), ("wm", lc.wm), ("vectors", lc.vectors)):
            if obj is None or not hasattr(obj, "clear"):
                continue
            try:
                obj.clear()
            except Exception as e:  # noqa: BLE001
                logger.error("[sessions] clearing %s at the boundary failed: %s", name, e)

    def _enforce_keep_cap(self) -> None:
        for rec in self.frozen[self.cfg.keep_in_memory:]:
            if rec.clouds is None and rec.vectors is None:
                continue
            if not rec.on_disk and not self.cfg.persist_on_freeze:
                logger.warning("[sessions] %s evicted from RAM without a copy on disk (persist_on_freeze is off)", rec.key)
                rec.timeline.append(TimelineEvent(self._now(), None, "freeze", {"evicted_unpersisted": True}))
            self._evict_payload(rec)

    @staticmethod
    def _evict_payload(rec: FrozenSession) -> None:
        rec.clouds = None
        rec.vectors = None
        rec.crops = None
        for o in rec.objects:
            o["emb_mean"] = None
        rec.payload_loaded = False if rec.on_disk else rec.payload_loaded

    def _refine_frozen(self, rec: FrozenSession, part: Dict[str, Any], kind: str, map_ref: Optional[str]) -> int:
        self._ensure_loaded(rec)
        deltas = frame_deltas_from_history(rec.pose_history, part)
        if not deltas:
            return 0
        thresh = float(getattr(self.live_components.wm, "pose_demote_thresh_m", 0.30) or 0.30)
        moved_oids: set = set()
        demoted = 0
        max_shift = 0.0
        by_id = {o["id"]: o for o in rec.objects if "id" in o}
        for fid, (old_pos, dR, dt) in deltas.items():
            new = _pose4(part[fid])
            if new is not None:
                max_shift = max(max_shift, float(np.linalg.norm(new[:3, 3] - old_pos)))
            for oid in rec.frame_to_objects.get(fid, []):
                o = by_id.get(oid)
                if o is None or o.get("xyz_world") is None:
                    continue
                old = np.asarray(o["xyz_world"], dtype=np.float32)
                new_xyz = (dR @ old + dt).astype(np.float32)
                if float(np.linalg.norm(new_xyz - old)) >= thresh:
                    if not o.get("demoted_by_correction"):
                        demoted += 1
                    o["demoted_by_correction"] = True
                o["xyz_world"] = new_xyz
                moved_oids.add(oid)
        if deltas:
            cams = np.array([v[0] for v in deltas.values()], dtype=np.float32)
            dl = list(deltas.values())
            for o in rec.objects:
                if o.get("id") in moved_oids or o.get("xyz_world") is None:
                    continue
                old = np.asarray(o["xyz_world"], dtype=np.float32)
                nearest = int(np.argmin(np.linalg.norm(cams - old[None, :], axis=1)))
                _p, dR, dt = dl[nearest]
                new_xyz = (dR @ old + dt).astype(np.float32)
                if float(np.linalg.norm(new_xyz - old)) >= thresh:
                    if not o.get("demoted_by_correction"):
                        demoted += 1
                    o["demoted_by_correction"] = True
                o["xyz_world"] = new_xyz
                moved_oids.add(o.get("id"))
        for fid, pose in part.items():
            p = _pose4(pose)
            if p is None:
                continue
            rec.pose_history[fid] = p
            if fid in rec.keyframes:
                rec.keyframes[fid].pose = p.copy()
        rec.counters["refinements"] += 1
        rec.counters["corrections_applied"] += len(deltas)
        rec.counters["objects_demoted_by_corrections"] += demoted
        rec.dirty = True
        self._event(rec, "refine", None, {"kind": kind, "n_keyframes": len(deltas), "max_shift_m": round(max_shift, 4),
                                          "objects_moved": len(moved_oids), "objects_demoted": demoted, "map_ref": map_ref})
        self._ledger("refine", rec, {"kind": kind, "n_keyframes": len(deltas), "max_shift_m": round(max_shift, 4)})
        self._persist_meta(rec)
        poses_out = {fid: rec.pose_history[fid].reshape(-1).tolist() for fid in deltas if fid in rec.pose_history}
        moved_out = [{"id": oid, "xyz_world": np.asarray(by_id[oid]["xyz_world"], dtype=np.float32).tolist(),
                      "demoted_by_correction": bool(by_id[oid].get("demoted_by_correction"))}
                     for oid in moved_oids if oid in by_id and by_id[oid].get("xyz_world") is not None]
        self._notify_refined(rec.key, poses_out, moved_out)
        return len(deltas)

    def _resolve_keys(self, corrections: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
        resolved: Dict[str, np.ndarray] = {}
        unaliased: List[str] = []
        unrouted: List[str] = []
        tol_ns = int(self.cfg.alias_stamp_tol_ms * 1e6)
        live = self.live
        for raw, pose in corrections.items():
            key = str(raw)
            p = _pose4(pose)
            if p is None:
                unaliased.append(key)
                continue
            if key.startswith("ws_") or key.startswith("rx_"):
                if self._known_frame(key):
                    resolved[key] = p
                else:
                    unrouted.append(key)
                continue
            if key.startswith(STAMP_KEY_PREFIX):
                try:
                    stamp = int(key[len(STAMP_KEY_PREFIX):])
                except ValueError:
                    unaliased.append(key)
                    continue
                fid = self._nearest_by_stamp(stamp, tol_ns)
                if fid is None:
                    unaliased.append(key)
                else:
                    resolved[fid] = p
                continue
            fid = live.kf_alias.get(key) if live is not None else None
            if fid is None:
                for rec in self.frozen:
                    fid = rec.kf_alias.get(key)
                    if fid is not None:
                        break
            if fid is None:
                unaliased.append(key)
            else:
                resolved[fid] = p
        return resolved, unaliased, unrouted

    def _known_frame(self, fid: str) -> bool:
        if self.live is not None and fid in self.live.pose_history:
            return True
        return any(fid in r.pose_history for r in self.frozen)

    def _nearest_by_stamp(self, stamp: int, tol_ns: int) -> Optional[str]:
        best: Optional[Tuple[int, str]] = None
        sources: List[Dict[str, int]] = []
        if self.live is not None:
            sources.append(self.live.alias_stamp)
        for rec in self.frozen:
            sources.append({k: v.timestamp_ns for k, v in rec.keyframes.items()} if rec.keyframes else rec.alias_stamp)
        for table in sources:
            for fid, ts in table.items():
                d = abs(int(ts) - stamp)
                if d <= tol_ns and (best is None or d < best[0]):
                    best = (d, fid)
            if best is not None:
                return best[1]
        return None

    def _merge_locked(self, rec: FrozenSession, live: SessionRecord) -> None:
        rec.merged_into = live.key
        if rec.key not in live.merged_from:
            live.merged_from.append(rec.key)
        now = self._now()
        self._event(rec, "merge", None, {"into": live.key})
        self._event(live, "merge", None, {"from": rec.key})
        self._ledger("merge", rec, {"into": live.key})
        rec.last_served_wall_utc = now
        self._persist_meta(rec)
        self._notify_change()

    def _live_chain(self) -> set:
        out: set = set()
        live = self.live
        if live is None:
            return out
        out.add(live.key)
        stack = list(live.merged_from)
        if live.aligned_to and live.aligned_to.get("session_key"):
            stack.append(live.aligned_to["session_key"])
        while stack:
            k = stack.pop()
            if k in out:
                continue
            out.add(k)
            rec = self._find(k)
            if rec is not None:
                stack.extend(rec.merged_from)
                if rec.aligned_to and rec.aligned_to.get("session_key"):
                    stack.append(rec.aligned_to["session_key"])
        return out

    def _retention_info(self, rec: SessionRecord) -> Dict[str, Any]:
        protected: List[str] = []
        if self.live is not None and rec.key == self.live.key:
            protected.append("live")
        elif rec.key in self._live_chain():
            protected.append("live_chain")
        if rec.pinned:
            protected.append("pinned")
        now = self._now()
        if rec.last_served_wall_utc is not None and now - rec.last_served_wall_utc < self.cfg.grace_served_min * 60.0:
            protected.append("recently_served")
        removable_after = None
        if not protected and self.cfg.max_age_days > 0:
            removable_after = (rec.ended_wall_utc or rec.started_wall_utc) + self.cfg.max_age_days * 86400.0
        return {"removable_after": removable_after, "protected_by": protected}

    def _remove_locked(self, rec: FrozenSession, *, reason: str) -> None:
        folder = self.dir / rec.key
        if folder.exists():
            gc_folder = self.dir / (rec.key + GC_SUFFIX)
            try:
                if gc_folder.exists():
                    shutil.rmtree(gc_folder, ignore_errors=True)
                os.replace(folder, gc_folder)
                shutil.rmtree(gc_folder, ignore_errors=True)
            except OSError as e:
                logger.error("[sessions] removing %s failed: %s", folder, e)
                return
        self.frozen = [r for r in self.frozen if r.key != rec.key]
        if rec.merged_into:
            succ = self._find(rec.merged_into) or (self.live if self.live and self.live.key == rec.merged_into else None)
            if succ is not None and rec.key in succ.merged_from:
                succ.merged_from.remove(rec.key)
        if reason == "gc":
            self.counters["gc_removed"] += 1
        self._ledger(reason if reason == "gc" else "delete", rec, {})
        self._notify_change()

    def _find(self, key: str) -> Optional[FrozenSession]:
        for r in self.frozen:
            if r.key == key:
                return r
        return None

    def _find_by_session_id(self, session_id: Optional[str]) -> Optional[FrozenSession]:
        if not session_id:
            return None
        for r in self.frozen:
            if r.session_id == session_id or r.key == session_id:
                return r
        return None

    @staticmethod
    def _object_summary(o: Dict[str, Any]) -> Dict[str, Any]:
        d = {k: v for k, v in o.items() if k not in ("emb_mean", "image_crops")}
        d["n_crops"] = len(o.get("image_crops") or []) if o.get("image_crops") is not None else o.get("n_crops", 0)
        if isinstance(d.get("xyz_world"), np.ndarray):
            d["xyz_world"] = d["xyz_world"].tolist()
        return d

    def _event(self, rec: SessionRecord, kind: str, t_sensor_ns: Optional[int], detail: Dict[str, Any]) -> None:
        rec.timeline.append(TimelineEvent(self._now(), t_sensor_ns, kind, dict(detail)))

    def _ledger(self, action: str, rec: SessionRecord, extra: Dict[str, Any]) -> None:
        if self.event_sink is None:
            return
        try:
            from rtsm.evaluation.event_log import SessionEvent
            self.event_sink(SessionEvent(timestamp=time.monotonic(), action=action, session_key=rec.key,
                                         session_id=rec.session_id, epoch=int(rec.epoch), reason=rec.end_reason,
                                         counts={**dict(rec.counters), **extra}))
        except Exception as e:  # noqa: BLE001
            logger.error("[sessions] ledger line failed: %s", e)

    # ── viewer accessors (the REST / WebSocket surface) ──

    def keyframes(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """A frozen session's keyframes: mesh id, stamp, 16-float pose, point count. None for an unknown key."""
        with self.mutate:
            rec = self._find(key)
            if rec is None:
                return None
            self._ensure_loaded(rec)
            return [{"mesh_id": k.mesh_id, "timestamp_ns": k.timestamp_ns, "pose": k.pose.reshape(-1).tolist(),
                     "n_points": k.n_points} for k in rec.keyframes.values()]

    def clouds(self, key: str) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """A frozen session's clouds (loaded on demand, counted as served). None for an unknown key."""
        with self.mutate:
            rec = self._find(key)
            if rec is None:
                return None
            self._ensure_loaded(rec)
            rec.last_served_wall_utc = self._now()
            return dict(rec.clouds or {})

    def snapshots(self, key: str, oid: str) -> Optional[List[bytes]]:
        """An object's JPEG crops in a frozen session, most recent first (the live
        endpoint's order). None for an unknown session; [] for no crops."""
        with self.mutate:
            rec = self._find(key)
            if rec is None:
                return None
            if rec.crops is not None and oid in rec.crops:
                return list(reversed(rec.crops[oid]))
            folder = (rec.crops_dir or (self.dir / rec.key / "crops")) / str(oid)
            if not folder.is_dir():
                return []
            files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            return [f.read_bytes() for f in reversed(files)]

    def _notify_change(self) -> None:
        cb = self.on_change
        if cb is None:
            return
        try:
            cb()
        except Exception as e:  # noqa: BLE001
            logger.error("[sessions] on_change hook failed: %s", e)

    def _notify_refined(self, key: str, poses: Dict[str, List[float]], objects: List[Dict[str, Any]]) -> None:
        cb = self.on_refined
        if cb is None:
            return
        try:
            cb(key, poses, objects)
        except Exception as e:  # noqa: BLE001
            logger.error("[sessions] on_refined hook failed: %s", e)

    # ── persistence ──

    def _schedule(self, fn: Callable[[], None]) -> None:
        if not self.persist_async:
            fn()
            return
        if self._writer is None:
            self._writer = threading.Thread(target=self._writer_loop, name="sessions-writer", daemon=True)
            self._writer.start()
        self._q.put(fn)

    def _writer_loop(self) -> None:
        while True:
            fn = self._q.get()
            self._busy = True
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                logger.error("[sessions] writer: %s", e)
            finally:
                self._busy = False
                self._q.task_done()

    def _persist_meta(self, rec: FrozenSession) -> None:
        if not self.cfg.persist_on_freeze or not rec.on_disk:
            return
        self._schedule(lambda r=rec: self._write_meta_files(r))

    def _meta_dict(self, rec: FrozenSession) -> Dict[str, Any]:
        return {
            "schema": SCHEMA, "key": rec.key, "session_id": rec.session_id, "epoch": rec.epoch, "source": rec.source,
            "started_wall_utc": rec.started_wall_utc, "started_sensor_ns": rec.started_sensor_ns,
            "ended_wall_utc": rec.ended_wall_utc, "end_reason": rec.end_reason, "aligned_to": rec.aligned_to,
            "merged_into": rec.merged_into, "merged_from": list(rec.merged_from), "pinned": rec.pinned,
            "last_served_wall_utc": rec.last_served_wall_utc, "counters": dict(rec.counters), "kf_alias": dict(rec.kf_alias),
            "discarded_predecessor": rec.discarded_predecessor, "vectors_exported": rec.vectors is not None,
        }

    def _write_meta_files(self, rec: FrozenSession, folder: Optional[Path] = None) -> None:
        folder = folder or (self.dir / rec.key)
        folder.mkdir(parents=True, exist_ok=True)
        with self.mutate:
            meta = self._meta_dict(rec)
            objects = [self._object_summary(o) for o in rec.objects]
            kfs = {"keyframes": [{"mesh_id": k.mesh_id, "timestamp_ns": k.timestamp_ns, "pose": k.pose.reshape(-1).tolist(),
                                  "n_points": k.n_points, "cloud_ref": k.cloud_ref} for k in rec.keyframes.values()],
                   "pose_history": {fid: p.reshape(-1).tolist() for fid, p in rec.pose_history.items()
                                    if fid in rec.keyframes or fid in rec.frame_to_objects},
                   "alias_stamp": dict(rec.alias_stamp)}
            timeline = [{"t_wall_utc_s": e.t_wall_utc_s, "t_sensor_ns": e.t_sensor_ns, "kind": e.kind, "detail": e.detail}
                        for e in rec.timeline]
            rec.dirty = False
        for name, payload in (("meta.json", meta), ("objects.json", objects), ("keyframes.json", kfs), ("timeline.jsonl", timeline)):
            tmp = folder / (name + TMP_SUFFIX)
            with open(tmp, "w", encoding="utf-8") as fh:
                if name.endswith(".jsonl"):
                    for line in payload:
                        fh.write(json.dumps(line, default=_json_default) + "\n")
                else:
                    json.dump(payload, fh, default=_json_default)
            os.replace(tmp, folder / name)

    def _persist_full(self, rec: FrozenSession) -> None:
        final = self.dir / rec.key
        tmp = self.dir / (rec.key + TMP_SUFFIX)
        try:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=False)
            with self.mutate:
                clouds = dict(rec.clouds or {})
                frame_index = {k: list(v) for k, v in rec.frame_to_objects.items()}
                vec = rec.vectors
                crops = dict(rec.crops or {}) if self.cfg.persist_crops else {}
            self._write_meta_files(rec, folder=tmp)
            with open(tmp / "frame_index.json", "w", encoding="utf-8") as fh:
                json.dump(frame_index, fh)
            arrays: Dict[str, np.ndarray] = {}
            for mid, (p, c) in clouds.items():
                arrays[f"positions_{mid}"] = np.asarray(p, dtype=np.float32)
                arrays[f"colors_{mid}"] = np.asarray(c, dtype=np.uint8)
            np.savez(tmp / "clouds.npz", **arrays)
            if vec is not None:
                np.savez(tmp / "vectors.npz", ids=np.array(list(vec[0]), dtype="U"), embs=np.asarray(vec[1], dtype=np.float32))
            for oid, jpegs in crops.items():
                d = tmp / "crops" / str(oid)
                d.mkdir(parents=True, exist_ok=True)
                for i, b in enumerate(jpegs):
                    (d / f"{i}.jpg").write_bytes(b)
            if final.exists():
                shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
            with self.mutate:
                rec.on_disk = True
                rec.persisted = True
                rec.persist_error = None
                rec.crops_dir = final / "crops"
                rec.crops = None
                if rec.dirty:
                    self._write_meta_files(rec)
            self._notify_change()                      # on_disk / persisted changed
        except Exception as e:  # noqa: BLE001
            with self.mutate:
                rec.persisted = False
                rec.persist_error = str(e)
            logger.error("[sessions] persisting %s failed: %s", rec.key, e)
            shutil.rmtree(tmp, ignore_errors=True)

    def _load_meta(self, folder: Path) -> FrozenSession:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        rec = FrozenSession(
            key=str(meta.get("key") or folder.name), session_id=meta.get("session_id"), epoch=int(meta.get("epoch", 0)),
            source=str(meta.get("source", "unknown")), started_wall_utc=float(meta.get("started_wall_utc", 0.0)),
            started_sensor_ns=meta.get("started_sensor_ns"), ended_wall_utc=meta.get("ended_wall_utc"),
            end_reason=meta.get("end_reason"), aligned_to=meta.get("aligned_to"), merged_into=meta.get("merged_into"),
            merged_from=list(meta.get("merged_from") or []), pinned=bool(meta.get("pinned", False)),
            last_served_wall_utc=meta.get("last_served_wall_utc"), counters={**{k: 0 for k in COUNTER_KEYS}, **(meta.get("counters") or {})},
            kf_alias={str(k): str(v) for k, v in (meta.get("kf_alias") or {}).items()},
            discarded_predecessor=meta.get("discarded_predecessor"), on_disk=True, persisted=True, payload_loaded=False,
            clouds=None, vectors=None, crops_dir=(folder / "crops") if (folder / "crops").is_dir() else None,
        )
        tl = folder / "timeline.jsonl"
        if tl.is_file():
            for line in tl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    rec.timeline.append(TimelineEvent(float(d.get("t_wall_utc_s", 0.0)), d.get("t_sensor_ns"),
                                                      str(d.get("kind", "start")), dict(d.get("detail") or {})))
        kf = folder / "keyframes.json"
        if kf.is_file():                                   # poses are cheap: load them for routing
            d = json.loads(kf.read_text(encoding="utf-8"))
            for k, v in (d.get("pose_history") or {}).items():
                p = _pose4(v)
                if p is not None:
                    rec.pose_history[str(k)] = p
            rec.alias_stamp = {str(k): int(v) for k, v in (d.get("alias_stamp") or {}).items()}
            for item in d.get("keyframes") or []:
                p = _pose4(item.get("pose"))
                if p is None:
                    continue
                rec.keyframes[str(item["mesh_id"])] = KeyframeSnapshot(str(item["mesh_id"]), int(item.get("timestamp_ns", 0)),
                                                                        p, int(item.get("n_points", 0)), str(item.get("cloud_ref", item["mesh_id"])))
        return rec

    def _ensure_loaded(self, rec: FrozenSession) -> None:
        """Objects, frame index, clouds and vectors from disk on first use (keep cap evicts them again)."""
        if rec.payload_loaded or not rec.on_disk:
            return
        folder = self.dir / rec.key
        try:
            obj = folder / "objects.json"
            if obj.is_file():
                rec.objects = json.loads(obj.read_text(encoding="utf-8"))
                for o in rec.objects:
                    if isinstance(o.get("xyz_world"), list):
                        o["xyz_world"] = np.asarray(o["xyz_world"], dtype=np.float32)
            fi = folder / "frame_index.json"
            if fi.is_file():
                rec.frame_to_objects = {str(k): list(v) for k, v in json.loads(fi.read_text(encoding="utf-8")).items()}
            cl = folder / "clouds.npz"
            if cl.is_file():
                with np.load(cl) as z:
                    clouds: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
                    for name in z.files:
                        if name.startswith("positions_"):
                            mid = name[len("positions_"):]
                            clouds[mid] = (z[name], z[f"colors_{mid}"])
                    rec.clouds = clouds
            ve = folder / "vectors.npz"
            if ve.is_file():
                with np.load(ve) as z:
                    rec.vectors = ([str(x) for x in z["ids"].tolist()], np.asarray(z["embs"], dtype=np.float32))
            rec.payload_loaded = True
        except Exception as e:  # noqa: BLE001
            logger.error("[sessions] loading %s failed: %s", rec.key, e)
        self._enforce_keep_cap_except(rec)

    def _enforce_keep_cap_except(self, keep: FrozenSession) -> None:
        loaded = [r for r in self.frozen if r.clouds is not None and r is not keep]
        for rec in loaded[self.cfg.keep_in_memory:] if self.cfg.keep_in_memory > 0 else loaded:
            if rec.on_disk:
                self._evict_payload(rec)
