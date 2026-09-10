"""
Working Memory (WM)

Authoritative in-memory store for *live* objects (proto + confirmed).
- Owns lifecycle: create, update, merge, confirm (promote), expire.
- Holds embeddings (mean + small gallery), label EWMA, stability, pose, timestamps.
- Mirrors spatial membership via an injected ObjectIndex (proximity index).
- Prepares compact payloads to upsert into Milvus (LTM) when objects are ready.

"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Iterable, Any, Callable, Protocol
import numpy as np
import time
import uuid
import threading
import heapq
import logging

logger = logging.getLogger(__name__)


POSE_STALE_AFTER_S_DEFAULT = 0.5


def resolve_pose_stale_after_s(cfg: Any) -> float:
    """``robot_pose.stale_after_s`` -> float seconds (finite, > 0; default
    0.5 = 2.5 nominal periods at a 5 Hz phone stream, so the DIAGNOSTIC flag
    does not flicker on ordinary WiFi jitter yet flips within half a second of
    the stream stopping). Raises ValueError so the runners can fail through
    parser.error before any model loads."""
    rp = (cfg.get("robot_pose") or {}) if isinstance(cfg, dict) else {}
    if not isinstance(rp, dict):
        raise ValueError(f"robot_pose must be a mapping of settings (robot_pose: {{stale_after_s: 0.5}}); got {rp!r}")
    raw = rp.get("stale_after_s", POSE_STALE_AFTER_S_DEFAULT)
    if raw is None:
        return POSE_STALE_AFTER_S_DEFAULT
    if isinstance(raw, bool):
        raise ValueError(f"robot_pose.stale_after_s must be a number of seconds > 0; got {raw!r}")
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"robot_pose.stale_after_s must be a number of seconds > 0; got {raw!r}")
    if not (v > 0) or v != v or v == float("inf"):
        raise ValueError(f"robot_pose.stale_after_s must be a finite number > 0; got {raw!r}")
    return v

# --- type aliases ---
Vec3 = np.ndarray  # shape (3,), float32
Emb = np.ndarray   # shape (D,), float32 L2-normalized unless stated

# --- helpers ---

def _l2norm(v: Emb) -> Emb:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def _cos(a: Emb, b: Emb) -> float:
    return float(np.dot(a, b))


def _now_wall_utc() -> float:
    return time.time()


def _compress_crop_jpeg(crop: np.ndarray, quality: int = 75) -> bytes:
    """Compress an HxWx3 uint8 crop to JPEG bytes (224 embedding crops
    or native-res judgment crops alike).

    Args:
        crop: BGR image array (H, W, 3) uint8 — crops come straight from
            the BGR ingest frame (io/websocket.decode_rgb contract), which
            is exactly what cv2.imencode expects. The old code assumed RGB
            and flipped, storing every snapshot with red/blue swapped
            (found 2026-08-15: a red Coca-Cola can served as blue).
        quality: JPEG quality (1-100)

    Returns:
        JPEG-encoded bytes, or empty bytes if compression fails
    """
    import cv2
    if crop is None or crop.size == 0:
        return b''
    try:
        ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            return bytes(buf)
    except Exception:
        pass
    return b''


def _view_bin_id(view_dir_cam: Optional[np.ndarray], AZ_BINS: int, EL_BINS: int) -> Optional[int]:
    if view_dir_cam is None:
        return None
    v = view_dir_cam.astype(np.float32)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    v = v / n
    # camera +Z forward, +X right, +Y down (typical pinhole cam frame)
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    az = np.arctan2(x, z)                    # [-pi, pi]
    el = np.arctan2(-y, np.hypot(x, z))      # [-pi/2, pi/2]
    az_i = int(np.floor((az + np.pi) / (2*np.pi) * AZ_BINS))
    el_i = int(np.floor((el + np.pi/2) / np.pi    * EL_BINS))
    az_i = max(0, min(AZ_BINS-1, az_i))
    el_i = max(0, min(EL_BINS-1, el_i))
    return el_i * AZ_BINS + az_i


# --- minimal observation contract (duck-typed) ---
# Association should pass an object with these attributes. A simple dataclass works too.
#   obs.p_world: Vec3 (world meters)                  [required]
#   obs.emb_vis: Emb (float32, L2)                    [required]
#   obs.view_dir_cam: np.ndarray, shape (3,) or None  [optional]
#   obs.centroid_px: tuple[int,int] or None           [optional]
#   obs.label_topk: list[tuple[str,float]] or None    [optional]
#   obs.depth_valid: float in [0,1]                   [optional]
#   obs.quality: float in [0,1]                       [optional]


# ------------------------- object state -------------------------
@dataclass(slots=True)
class ObjectState:
    id: str
    xyz_world: Vec3
    cov_world: Vec3                      # diag variance (m^2), shape (3,)

    emb_mean: Emb                        # float32 L2-normalized
    emb_gallery: np.ndarray              # float16, shape (N,D)

    view_bins: Dict[int, Emb]            # bin_id -> mean emb (float32 L2)

    label_scores: Dict[str, float]       # EWMA label scores
    label_primary: Optional[str]

    stability: float                     # [0,1]
    hits: int

    confirmed: bool

    created_mono: float
    created_wall_utc: float

    last_seen_mono: float
    last_seen_wall_utc: float
    last_seen_px: Optional[Tuple[float, float]]

    last_upsert_wall_utc: float          # 0 if never upserted
    last_upsert_mono: float              # 0 if never upserted (monotonic seconds)
    last_upsert_emb: Optional[Emb]
    last_upsert_xyz: Optional[Vec3]

    # RGB crop gallery (JPEG-compressed bytes, most recent last)
    image_crops: List[bytes]

    # Frame tracking for precise pose corrections
    last_update_frame_id: Optional[str]

    # cache
    _dim: int

# ------------------------- Proximity index interface -------------------------

class ProximityIndexLike(Protocol):
    """Protocol for the minimal methods WorkingMemory needs from the ProximityIndex.
    
    API:
      - insert: insert an object into the index
      - update: update an object in the index
      - remove: remove an object from the index
    """

    def insert(self, oid: str, xyz_world: Vec3, wm_lookup: Optional[Callable[[str], Optional[Tuple[bool, float, float]]]] = None) -> None: ...

    def update(self, oid: str, old_xyz_world: Vec3, new_xyz_world: Vec3, wm_lookup: Optional[Callable[[str], Optional[Tuple[bool, float, float]]]] = None) -> None: ...

    def remove(self, oid: str, last_xyz_world: Optional[Vec3] = None) -> None: ...

# ------------------------- Working Memory -------------------------

class WorkingMemory:
    def __init__(self, cfg: Dict[str, Any], *, index: Optional[ProximityIndexLike] = None,
                 clock: Optional[Any] = None) -> None:
        self.cfg = cfg
        self.index = index  # ObjectIndex-like: insert/update/remove
        # Ingest clock (rtsm/core/clock.py): every memory-timing read — object
        # create/update stamps, proto TTL expiry, LTM upsert scheduling — goes
        # through it so the stamper and the expirer share one clock. Wall by
        # default; the runner injects a SensorClock under --replay so memory
        # timing follows the frames' own timestamps, not replay speed.
        if clock is None:
            from rtsm.core.clock import WallClock
            clock = WallClock()
        self._clock = clock

        self._map: Dict[str, ObjectState] = {}
        self._lock = threading.RLock()
        # Latest robot pose: a MAILBOX (Gate 4.5 plan, P1 task 4). Passthrough
        # from the sensor, written at receive time by the ONE receiver in this
        # process (websocket / replay / zeromq); the pipeline does not write
        # it. (frame_epoch, sensor_ts_ns) is a regression detector, not a
        # gate -- see update_robot_pose.
        self._latest_pose: Optional[Dict[str, Any]] = None
        # Process-monotonic arrival time of the stored pose (-> age_s).
        self._latest_pose_arrival_mono: float = 0.0
        self._pose_writes_accepted: int = 0
        self._pose_rejects: Dict[str, int] = {self.POSE_REJECT_OLDER_EPOCH: 0, self.POSE_REJECT_EPOCHLESS: 0}
        self._pose_consecutive_rejects: int = 0
        self._pose_regressions: int = 0             # accepted writes whose stamp went backwards in-epoch
        self._pose_last_regression_warn_mono: float = float("-inf")
        # robot_pose.stale_after_s: DIAGNOSTIC threshold for the payload's
        # `stale` flag (age_s above it). The RC-car agent's stale_abort_s is
        # the safety bound and lives in the agent; RTSM never acts on this.
        # Validated (finite, > 0); the runners call the same resolver before
        # any model loads so a typo fails through parser.error.
        self._pose_stale_after_s: float = resolve_pose_stale_after_s(cfg)
        # Latest forward-clearance summary from the depth stream (meters of
        # open space ahead of the camera). Written by the websocket receiver's
        # clearance_sink at receive time (rtsm/io/websocket.py step 8b), and
        # only when io.clearance.enable is true; consumed by agents as a wall
        # guard before blind motion. The flag also decides whether stats()
        # carries the forward_clearance key at all, so a default build
        # publishes exactly the pre-feature /stats contract.
        self.clearance_enabled: bool = bool(
            ((cfg.get("io") or {}).get("clearance") or {}).get("enable", False)
        )
        self._latest_clearance: Optional[Dict[str, Any]] = None
        # Rolling per-label detector tally (2026-08-28): with a PROMPTED
        # backend (grounded_sam2 + roster vocab) the detector's label is
        # meaningful, and confirmation/indexing lag can hide detections
        # from semantic search for minutes — this answers "is the
        # detector seeing X right now?" independently of memory state.
        # label -> {n, last_seen_wall, last_score, max_score}
        self._label_detections: Dict[str, Dict[str, Any]] = {}
        # Reverse index: frame_id -> set of object IDs last updated on that frame
        self._frame_to_objects: Dict[str, set] = {}
        # Min-heap of (deadline_mono, oid) for proto expiry (lazy re-schedule on matches)
        self._proto_heap: List[Tuple[float, str]] = []

        # Min-heap of (due_mono, oid) for LTM upsert scheduling (lazy duplicates OK)
        self._ltm_heap: List[Tuple[float, str]] = []

        # counters / telemetry
        self._upsert_count_total: int = 0

        # configs (with defaults)
        obj_cfg = cfg.get("object", {})
        self.proto_ttl_s: float = float(obj_cfg.get("proto_ttl_s", 10.0))
        self.promote_hits: int = int(obj_cfg.get("promote_hits", 2))
        self.stability_promote: float = float(obj_cfg.get("stability_promote", 0.50))
        self.require_view_bins: int = int(obj_cfg.get("require_view_bins", 2))
        self.stab_k: float = float(obj_cfg.get("stab_k", 0.45))
        self.miss_decay: float = float(obj_cfg.get("miss_decay", 0.92))

        self.az_bins: int = int(cfg.get("view", {}).get("az_bins", 8))
        self.el_bins: int = int(cfg.get("view", {}).get("el_bins", 3))

        pose_cfg = cfg.get("pose", {})
        self.meas_var_xyz_cm2 = np.array(pose_cfg.get("meas_var_xyz_cm2", [1.5, 1.5, 3.0]), dtype=np.float32) / 1e4
        self.proc_var_xyz_cm2 = np.array(pose_cfg.get("proc_var_xyz_cm2", [0.2, 0.2, 0.4]), dtype=np.float32) / 1e4
        # Threshold (meters) above which a pose correction demotes confirmed objects
        # back to proto so they must re-earn confirmation from good-pose frames.
        self.pose_demote_thresh_m: float = float(pose_cfg.get("demote_thresh_m", 0.30))

        ltm_cfg = cfg.get("ltm", {})
        self.reupsert_cos_max: float = float(ltm_cfg.get("reupsert_cos_max", 0.995))
        self.reupsert_pos_m: float = float(ltm_cfg.get("reupsert_pos_m", 0.05))
        # LTM view-diversity gate. Defaults to object.require_view_bins: an
        # object that confirms with N view bins must also reach the vector
        # store with N, otherwise it is confirmed-but-unsearchable forever
        # (with a stationary camera most objects only ever occupy one bin).
        self.ltm_min_view_bins: int = int(ltm_cfg.get("ltm_min_view_bins", self.require_view_bins))
        if self.ltm_min_view_bins > self.require_view_bins:
            logger.warning(
                f"[WM] ltm.ltm_min_view_bins={self.ltm_min_view_bins} > "
                f"object.require_view_bins={self.require_view_bins}: confirmed objects "
                f"with fewer view bins will never be upserted to the vector store "
                f"and stay invisible to /search/semantic"
            )
        self.ltm_min_period_s: float = float(ltm_cfg.get("min_period_s", 1.0))
        self.ltm_force_period_s: float = float(ltm_cfg.get("force_period_s", 10.0))

        self.max_gallery: int = int(obj_cfg.get("max_gallery", 6))
        self.gallery_dupe_cos: float = float(obj_cfg.get("gallery_dupe_cos", 0.995))


    # ---------- CRUD ----------

    def exists(self, oid: str) -> bool:
        with self._lock:
            return oid in self._map

    def get(self, oid: str) -> Optional[ObjectState]:
        with self._lock:
            return self._map.get(oid)

    def lookup_min(self, oid: str) -> Optional[Tuple[bool, float, float]]:
        """Tiny tuple used by ProximityIndex eviction ranking: (confirmed, stability, last_seen_mono)."""
        with self._lock:
            o = self._map.get(oid)
            if o is None:
                return None
            return (o.confirmed, o.stability, o.last_seen_mono)

    def iter_objects(self) -> Iterable[ObjectState]:
        with self._lock:
            return list(self._map.values())

    # ---------- create / spawn ----------

    def create_object(self, p_world: Vec3, emb_vis: Emb, *, t_mono: Optional[float] = None,
                      label_topk: Optional[List[Tuple[str, float]]] = None,
                      view_dir_cam: Optional[np.ndarray] = None,
                      centroid_px: Optional[Tuple[float, float]] = None,
                      crop: Optional[np.ndarray] = None,
                      frame_id: Optional[str] = None) -> Optional[str]:
        """Spawn a new proto object. Index is updated here as well.

        Returns:
            Object ID if created, None if rejected (e.g., out of bounds)
        """
        t_mono = self._clock.now_mono() if t_mono is None else t_mono
        wall_now = _now_wall_utc()
        emb_vis = emb_vis.astype(np.float32)
        D = int(emb_vis.shape[0])

        # Position bounds validation (optional)
        bounds_cfg = self.cfg.get("object", {}).get("position_bounds_m", None)
        if bounds_cfg is not None:
            x_bounds = bounds_cfg.get("x", [-100, 100])
            y_bounds = bounds_cfg.get("y", [-100, 100])
            z_bounds = bounds_cfg.get("z", [-100, 100])
            px, py, pz = float(p_world[0]), float(p_world[1]), float(p_world[2])
            if not (x_bounds[0] <= px <= x_bounds[1] and
                    y_bounds[0] <= py <= y_bounds[1] and
                    z_bounds[0] <= pz <= z_bounds[1]):
                logger.warning(
                    f"[WM] create_object rejected: position out of bounds "
                    f"xyz=[{px:.2f},{py:.2f},{pz:.2f}] "
                    f"bounds=x{x_bounds} y{y_bounds} z{z_bounds}"
                )
                return None

        oid = uuid.uuid4().hex[:16]
        emb_mean = emb_vis.copy()
        gallery = emb_vis.astype(np.float16)[None, :]  # (1,D)
        view_bins: Dict[int, Emb] = {}
        b = _view_bin_id(view_dir_cam, self.az_bins, self.el_bins)
        if b is not None:
            view_bins[b] = emb_vis.copy()

        label_scores: Dict[str, float] = {}
        if label_topk:
            for lbl, sc in label_topk:
                label_scores[lbl] = max(label_scores.get(lbl, 0.0), float(sc))
        label_primary = max(label_scores.items(), key=lambda kv: kv[1])[0] if label_scores else None

        # Compress and store initial crop
        image_crops: List[bytes] = []
        if crop is not None:
            jpeg_quality = int(self.cfg.get("object", {}).get("crop_jpeg_quality", 75))
            jpeg_bytes = _compress_crop_jpeg(crop, quality=jpeg_quality)
            if jpeg_bytes:
                image_crops.append(jpeg_bytes)

        o = ObjectState(
            id=oid,
            xyz_world=p_world.astype(np.float32),
            cov_world=np.array([0.02, 0.02, 0.04], dtype=np.float32),  # loose init
            emb_mean=emb_mean,
            emb_gallery=gallery,
            view_bins=view_bins,
            label_scores=label_scores,
            label_primary=label_primary,
            stability=0.25,
            hits=1,
            confirmed=False,
            created_mono=t_mono,
            created_wall_utc=wall_now,
            last_seen_mono=t_mono,
            last_seen_wall_utc=wall_now,
            last_seen_px=centroid_px,
            last_upsert_wall_utc=0.0,
            last_upsert_mono=0.0,
            last_upsert_emb=None,
            last_upsert_xyz=None,
            image_crops=image_crops,
            last_update_frame_id=frame_id,
            _dim=D,
        )
        with self._lock:
            self._map[oid] = o
            # Register in frame → objects reverse index
            if frame_id is not None:
                self._frame_to_objects.setdefault(frame_id, set()).add(oid)
            # schedule proto expiry (confirmed objects are never scheduled here)
            self._schedule_proto(oid, o)
        if self.index is not None:
            self.index.insert(oid, o.xyz_world, wm_lookup=self.lookup_min)
        logger.debug(
            f"[WM] create oid={oid} label={label_primary if label_primary else '-'} "
            f"xyz=[{p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f}]"
        )
        return oid


    def update_object(self, oid: str, obs: Any, *, dt_s: Optional[float] = None) -> None:
        """Update state from a matched observation. Association guarantees `obs.p_world` & `obs.emb_vis`.
        Optional fields used if present: view_dir_cam, centroid_px, label_topk, depth_valid, quality.
        """
        with self._lock:
            o = self._map.get(oid)
            if o is None:
                return
            old_xyz = o.xyz_world.copy()

        # --- timestamps & deltas ---
        now_m = self._clock.now_mono()
        now_w = _now_wall_utc()
        dt_s = float(dt_s if dt_s is not None else max(1e-3, now_m - o.last_seen_mono))

        # --- pose EMA (keyframe-dominant) ---
        depth_valid = float(getattr(obs, "depth_valid", 1.0) or 0.0)
        quality = float(getattr(obs, "quality", 1.0) or 0.0)
        is_kf = bool(getattr(obs, "is_keyframe", False))
        if is_kf:
            # Keyframe: near-full trust in new measurement
            w = float(np.clip(0.9 + 0.09 * depth_valid * quality, 0.9, 0.99))
        else:
            # Non-keyframe: minimal influence, preserve keyframe position
            w = float(np.clip(0.01 + 0.09 * depth_valid * quality, 0.01, 0.1))
        z_world = obs.p_world.astype(np.float32)
        xyz_new = (1.0 - w) * o.xyz_world + w * z_world
        # diag covariance update (simple):
        R = self.meas_var_xyz_cm2
        o_cov = (1.0 - w) ** 2 * o.cov_world + (w ** 2) * R
        o_cov = o_cov + self.proc_var_xyz_cm2 * dt_s

        # --- embeddings (gallery, mean, view bin) ---
        e = obs.emb_vis.astype(np.float32)
        # gallery: only add if not near-duplicate
        add_to_gallery = True
        if o.emb_gallery.shape[0] > 0:
            cos_max = float(np.max((o.emb_gallery.astype(np.float32) @ e).astype(np.float32)))
            add_to_gallery = cos_max < self.gallery_dupe_cos or o.emb_gallery.shape[0] < 1
        if add_to_gallery:
            if o.emb_gallery.shape[0] < self.max_gallery:
                o.emb_gallery = np.vstack([o.emb_gallery, e.astype(np.float16)])
            else:
                # FIFO: drop oldest (row 0)
                o.emb_gallery = np.vstack([o.emb_gallery[1:], e.astype(np.float16)])
        # mean
        emb_mean = _l2norm(o.emb_mean * o.hits + e)  # simple running mean in L2 space (approx)

        # view-bin update
        b = _view_bin_id(getattr(obs, "view_dir_cam", None), self.az_bins, self.el_bins)
        if b is not None:
            prev = o.view_bins.get(b)
            o.view_bins[b] = e if prev is None else _l2norm(0.5 * prev + 0.5 * e)

        # --- labels (EWMA) ---
        topk = getattr(obs, "label_topk", None)
        if topk:
            for lbl, sc in topk:
                s_old = o.label_scores.get(lbl, 0.0)
                # EWMA toward score; smaller beta keeps memory of history
                beta = 0.5
                o.label_scores[lbl] = (1 - beta) * s_old + beta * float(sc)
        # primary
        if o.label_scores:
            o.label_primary = max(o.label_scores.items(), key=lambda kv: kv[1])[0]

        # --- image crop gallery (FIFO, max 6) ---
        crop = getattr(obs, 'crop', None)
        if crop is not None:
            jpeg_quality = int(self.cfg.get("object", {}).get("crop_jpeg_quality", 75))
            jpeg_bytes = _compress_crop_jpeg(crop, quality=jpeg_quality)
            if jpeg_bytes:
                max_crops = int(self.cfg.get("object", {}).get("max_image_crops", 6))
                o.image_crops.append(jpeg_bytes)
                # FIFO: keep only most recent max_crops
                if len(o.image_crops) > max_crops:
                    o.image_crops = o.image_crops[-max_crops:]

        # --- stability ---
        # Build a simple gain from geometry + appearance (association can pass cos/dist/px if desired).
        cos_sim = float(getattr(obs, "cos_sim", 0.9))
        dist_m = float(getattr(obs, "dist_m", 0.0))
        gate = float(self.cfg.get("assoc", {}).get("gate_dist_base_m", 0.20))
        cos_n = max(0.0, min(1.0, (cos_sim - 0.5) / 0.5))
        dist_n = 1.0 - min(1.0, dist_m / max(1e-6, gate))
        quality_n = quality
        gain = max(0.0, 0.6 * cos_n + 0.3 * dist_n + 0.1 * quality_n)
        prev_stab = float(o.stability)
        prev_hits = int(o.hits)
        stab = min(1.0, o.stability + self.stab_k * gain * (1.0 - o.stability))

        # --- write back (under lock), and index move if needed ---
        new_frame_id = getattr(obs, "frame_id", None)
        with self._lock:
            o.xyz_world = xyz_new.astype(np.float32)
            o.cov_world = o_cov.astype(np.float32)
            o.emb_mean = emb_mean
            o.hits += 1
            o.stability = stab
            o.last_seen_mono = now_m
            o.last_seen_wall_utc = now_w
            o.last_seen_px = getattr(obs, "centroid_px", None)
            # Update frame → objects reverse index
            if new_frame_id is not None:
                old_frame_id = o.last_update_frame_id
                if old_frame_id is not None and old_frame_id != new_frame_id:
                    old_set = self._frame_to_objects.get(old_frame_id)
                    if old_set is not None:
                        old_set.discard(oid)
                        if not old_set:
                            del self._frame_to_objects[old_frame_id]
                self._frame_to_objects.setdefault(new_frame_id, set()).add(oid)
                o.last_update_frame_id = new_frame_id
            # view_bins, label_scores, label_primary already updated on o
            # If still proto, push a fresh deadline (lazy heap pattern tolerates duplicates)
            if not o.confirmed:
                self._schedule_proto(oid, o)

        if self.index is not None and np.any(self.index.grid.cell(old_xyz) != self.index.grid.cell(o.xyz_world)):
            self.index.update(oid, old_xyz, o.xyz_world, wm_lookup=self.lookup_min)

        # --- logging: match update (DEBUG level to reduce noise) ---

        lbl = getattr(o, 'label_primary', None)
        logger.debug(
            f"[WM] match oid={oid} hits={prev_hits}->{prev_hits+1} "
            f"stab={prev_stab:.3f}->{stab:.3f} cos={cos_sim:.3f} dist_m={dist_m:.3f} "
            f"label={lbl if lbl is not None else '-'}"
        )


    # ---------- miss / decay (call for unmatched objects) ----------

    def decay_unmatched(self, dt_s: float) -> None:
        """Decay stability for all objects when they weren't observed this frame.
        Call once per frame with dt from previous frame in *monotonic* seconds.
        """
        if dt_s <= 0:
            return
        decay = float(self.miss_decay ** max(1.0, dt_s * 30.0))  # approx per-30fps frames
        with self._lock:
            for o in self._map.values():
                o.stability *= decay

    # ---------- promotion & readiness ----------

    def maybe_promote(self, oid: str) -> None:
        with self._lock:
            o = self._map.get(oid)
            if o is None or o.confirmed:
                return
            if o.hits >= self.promote_hits and o.stability >= self.stability_promote and len(o.view_bins) >= self.require_view_bins:
                o.confirmed = True
                # Schedule immediate LTM eligibility check
                top_lbl = o.label_primary
                conf = (o.label_scores.get(top_lbl, 0.0) if top_lbl else 0.0)
                logger.info(
                    f"[WM] promote oid={oid} label={top_lbl if top_lbl else '-'} "
                    f"conf={conf:.3f} hits={o.hits} stab={o.stability:.3f}"
                )
                heapq.heappush(self._ltm_heap, (self._clock.now_mono(), oid))

    def collect_ready_for_upsert(self, force_all: bool = False) -> List[Dict[str, Any]]:
        """Collect confirmed objects that should be (re)upserted to LTM now.
        Returns a list of dict payloads; caller performs the actual DB write.
        Uses a due-time heap (monotonic seconds) to avoid scanning the entire map each time.

        Args:
            force_all: If True, skip all timing/change checks and upsert ALL
                       confirmed objects. Used by demo mode to ensure all
                       confirmed objects are searchable after replay completes.
        """
        out: List[Dict[str, Any]] = []
        m_now = self._clock.now_mono()
        wall_now = _now_wall_utc()

        def _schedule_next_due(o: ObjectState, now_m: float) -> None:
            # Next regular check after min_period based on monotonic timestamp
            last_m = float(o.last_upsert_mono or 0.0)
            next_regular = max(now_m, last_m + self.ltm_min_period_s)
            heapq.heappush(self._ltm_heap, (next_regular, o.id))

        with self._lock:
            # Force-flush: upsert ALL confirmed objects, skip timing/change checks
            if force_all:
                for o in self._map.values():
                    if not o.confirmed or o.emb_mean is None:
                        continue
                    label_topk = sorted(o.label_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    out.append({
                        "object_id": o.id,
                        "emb": o.emb_mean.astype(np.float32),
                        "xyz": o.xyz_world.astype(np.float32),
                        "label_primary": o.label_primary,
                        "label_confidence": (o.label_scores.get(o.label_primary, 0.0) if o.label_primary else 0.0),
                        "label_topk": [k for k, _ in label_topk],
                        "label_scores": [float(v) for _, v in label_topk],
                        "stability": float(o.stability),
                        "created_at": o.created_wall_utc,
                        "created_mono": o.created_mono,
                    })
                    o.last_upsert_mono = m_now
                    o.last_upsert_emb = o.emb_mean.copy()
                    o.last_upsert_xyz = o.xyz_world.copy()
                    self._upsert_count_total += 1
                return out

            # Drain heap for entries due now (lazy duplicates tolerated)
            while self._ltm_heap and self._ltm_heap[0][0] <= m_now:
                _, oid = heapq.heappop(self._ltm_heap)
                o = self._map.get(oid)
                if o is None or not o.confirmed:
                    continue  # stale or not eligible
                # diversity requirement
                if len(o.view_bins) < max(self.ltm_min_view_bins, 1):
                    # re-check later
                    heapq.heappush(self._ltm_heap, (m_now + self.ltm_min_period_s, oid))
                    continue
                # time since last upsert
                elapsed_m = m_now - float(o.last_upsert_mono or 0.0)
                if elapsed_m < self.ltm_min_period_s:
                    # not yet; push to the min-period boundary
                    heapq.heappush(self._ltm_heap, (float(o.last_upsert_mono or 0.0) + self.ltm_min_period_s, oid))
                    continue
                # change tests
                changed = True
                if o.last_upsert_emb is not None:
                    cos_same = _cos(o.emb_mean, o.last_upsert_emb)
                    ref_xyz = o.last_upsert_xyz if o.last_upsert_xyz is not None else o.xyz_world
                    pos_delta = float(np.linalg.norm(o.xyz_world - ref_xyz))
                    changed = (cos_same <= self.reupsert_cos_max) or (pos_delta >= self.reupsert_pos_m) or (elapsed_m >= self.ltm_force_period_s)
                if not changed:
                    # schedule sooner of next min-period or force window
                    remaining_to_force = max(0.0, (float(o.last_upsert_mono or m_now) + self.ltm_force_period_s) - m_now)
                    delay = min(self.ltm_min_period_s, remaining_to_force)
                    heapq.heappush(self._ltm_heap, (m_now + delay, oid))
                    continue

                # build compact record (no huge blobs)
                label_topk = sorted(o.label_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
                payload = {
                    "object_id": o.id,
                    "emb": o.emb_mean.astype(np.float32),
                    "xyz": o.xyz_world.astype(np.float32),
                    "label_primary": o.label_primary,
                    "label_confidence": (o.label_scores.get(o.label_primary, 0.0) if o.label_primary else 0.0),
                    "label_topk": [k for k, _ in label_topk],
                    "label_scores": [float(v) for _, v in label_topk],
                    "stability": float(o.stability),
                    "created_at": o.created_wall_utc,
                    "created_mono": o.created_mono,
                    "updated_at": wall_now,
                }
                out.append(payload)
                is_first = o.last_upsert_emb is None
                # mark last upsert snapshot
                o.last_upsert_wall_utc = wall_now
                o.last_upsert_mono = m_now
                o.last_upsert_emb = o.emb_mean.copy()
                o.last_upsert_xyz = o.xyz_world.copy()
                # telemetry & logging
                self._upsert_count_total += 1

                reason = "first_upsert" if is_first else (
                    "force_period" if elapsed_m >= self.ltm_force_period_s else (
                        "emb_changed" if cos_same <= self.reupsert_cos_max else "pos_changed"
                    )
                )
                logger.debug(
                    f"[WM] upsert oid={o.id} label={o.label_primary if o.label_primary else '-'} "
                    f"views={len(o.view_bins)} stab={o.stability:.3f} reason={reason} total={self._upsert_count_total}"
                )

                # schedule next routine check
                _schedule_next_due(o, m_now)
        return out

    def mark_upsert_failed(self, object_ids: Iterable[str]) -> int:
        """Roll back upsert bookkeeping for payloads the caller failed to
        write to the vector store, and re-queue them for the next flush.

        collect_ready_for_upsert() snapshots last_upsert_* optimistically at
        collection time; without this rollback a failed store write leaves the
        objects looking freshly upserted, so they would not retry until the
        force period — or never, if their heap entries were consumed.

        Returns the number of objects re-queued.
        """
        now_m = self._clock.now_mono()
        requeued = 0
        with self._lock:
            for oid in object_ids:
                o = self._map.get(oid)
                if o is None:
                    continue
                o.last_upsert_mono = 0.0
                o.last_upsert_wall_utc = 0.0
                o.last_upsert_emb = None
                o.last_upsert_xyz = None
                heapq.heappush(self._ltm_heap, (now_m, oid))
                requeued += 1
        return requeued

    # ---------- expiry / pruning ----------

    def expire_timeouts(self) -> List[str]:
        """Expire proto objects past TTL using a min-heap. Returns list of removed IDs."""
        now_m = self._clock.now_mono()
        removed: List[str] = []
        with self._lock:
            while self._proto_heap and self._proto_heap[0][0] <= now_m:
                _, oid = heapq.heappop(self._proto_heap)
                o = self._map.get(oid)
                if o is None or o.confirmed:
                    continue  # stale heap entry
                # recompute the true current deadline (may have been extended by matches)
                true_deadline = o.last_seen_mono + self.proto_ttl_s
                if true_deadline > now_m:
                    # deadline extended; push a fresh entry (lazy heap pattern)
                    heapq.heappush(self._proto_heap, (true_deadline, oid))
                    continue
                # Clean up frame → objects reverse index
                if o.last_update_frame_id is not None:
                    fset = self._frame_to_objects.get(o.last_update_frame_id)
                    if fset is not None:
                        fset.discard(oid)
                        if not fset:
                            del self._frame_to_objects[o.last_update_frame_id]
                # really expired
                removed.append(oid)
                del self._map[oid]
        if self.index is not None:
            for oid in removed:
                self.index.remove(oid, None)
        return removed

    # ---------- internal: proto scheduling ----------
    def _schedule_proto(self, oid: str, o: ObjectState) -> None:
        """Push a (deadline, oid) for a proto object into the heap. Lock must be held."""
        deadline = o.last_seen_mono + self.proto_ttl_s
        heapq.heappush(self._proto_heap, (deadline, oid))

    # ---------- demotion (bad-pose recovery) ----------

    def _demote_object(self, o: ObjectState) -> None:
        """Demote a confirmed object back to proto. Lock must be held.

        Resets hits and stability so the object must re-earn confirmation
        from future good-pose frames.  Schedules it for proto TTL expiry
        so it gets cleaned up if not re-observed.
        """
        was_confirmed = o.confirmed
        o.confirmed = False
        o.hits = 0
        o.stability = 0.0
        self._schedule_proto(o.id, o)
        if was_confirmed:
            logger.info(
                f"[WM] demoted oid={o.id} label={o.label_primary or '-'} "
                f"back to proto (large pose correction)"
            )

    # ---------- pose corrections (loop closure) ----------

    def apply_pose_corrections(
        self,
        frame_corrections: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> int:
        """Apply retroactive pose corrections from SLAM loop closure.

        Uses the frame_id → object_ids reverse index for precise correction:
        objects linked to a corrected frame get that frame's exact delta.
        Objects not linked to any corrected frame fall back to the delta
        from the spatially nearest corrected camera position.

        Args:
            frame_corrections: dict mapping frame_id to
                (old_cam_pos_3, delta_R_3x3, delta_t_3).
                delta transforms a world point: p_new = delta_R @ p_old + delta_t

        Returns:
            Number of objects corrected.
        """
        if not frame_corrections:
            return 0

        corrected_oids: set = set()
        demoted_oids: list = []
        corrected = 0
        thresh = self.pose_demote_thresh_m

        with self._lock:
            # Phase 1: Direct frame_id → object lookup
            for frame_id, (_, delta_R, delta_t) in frame_corrections.items():
                linked_oids = self._frame_to_objects.get(frame_id, set())
                for oid in linked_oids:
                    o = self._map.get(oid)
                    if o is None:
                        continue
                    old_xyz = o.xyz_world.copy()
                    new_xyz = (delta_R @ old_xyz + delta_t).astype(np.float32)
                    shift_m = float(np.linalg.norm(new_xyz - old_xyz))
                    o.xyz_world = new_xyz
                    corrected += 1
                    corrected_oids.add(oid)

                    # Demote if the correction is large — the frame that last
                    # updated this object had a bad pose, so the observations
                    # that drove promotion are untrustworthy.
                    if shift_m >= thresh:
                        self._demote_object(o)
                        demoted_oids.append(oid)

                    if self.index is not None:
                        old_cell = self.index.grid.cell(old_xyz)
                        new_cell = self.index.grid.cell(new_xyz)
                        if old_cell != new_cell:
                            self.index.update(oid, old_xyz, new_xyz, wm_lookup=self.lookup_min)

            # Phase 2: Spatial fallback for objects not linked to any corrected frame
            uncorrected = [o for o in self._map.values() if o.id not in corrected_oids]
            if uncorrected:
                cam_positions = np.array(
                    [v[0] for v in frame_corrections.values()], dtype=np.float32
                )
                deltas_list = list(frame_corrections.values())
                for o in uncorrected:
                    old_xyz = o.xyz_world.copy()
                    diffs = cam_positions - old_xyz[None, :]
                    dists = np.linalg.norm(diffs, axis=1)
                    nearest_idx = int(np.argmin(dists))
                    _, delta_R, delta_t = deltas_list[nearest_idx]
                    new_xyz = (delta_R @ old_xyz + delta_t).astype(np.float32)
                    shift_m = float(np.linalg.norm(new_xyz - old_xyz))
                    o.xyz_world = new_xyz
                    corrected += 1

                    if shift_m >= thresh:
                        self._demote_object(o)
                        demoted_oids.append(o.id)

                    if self.index is not None:
                        old_cell = self.index.grid.cell(old_xyz)
                        new_cell = self.index.grid.cell(new_xyz)
                        if old_cell != new_cell:
                            self.index.update(o.id, old_xyz, new_xyz, wm_lookup=self.lookup_min)

        if corrected > 0:
            direct = len(corrected_oids)
            fallback = corrected - direct
            logger.info(
                f"[WM] Applied pose corrections to {corrected} objects "
                f"({direct} direct, {fallback} fallback) "
                f"from {len(frame_corrections)} frame deltas"
                f"{f', demoted {len(demoted_oids)} back to proto' if demoted_oids else ''}"
            )
        return corrected

    # ---------- utilities ----------

    # Robot-pose mailbox rejection reasons (keys of robot_pose.rejected_by_reason).
    # Both can only come from a SECOND writer: every shipped receiver writes at
    # receive time with its own epoch, and the pipeline no longer writes at all.
    POSE_REJECT_OLDER_EPOCH = "older_epoch"           # a frame from a previous streaming session
    POSE_REJECT_EPOCHLESS = "epochless_into_epoch"    # writer knows no epoch, stored pose has one
    _POSE_REGRESSION_WARN_EVERY_S = 60.0

    def update_robot_pose(
        self,
        t_wc: np.ndarray,
        q_wc_xyzw: np.ndarray,
        timestamp: float,
        frame_epoch: Optional[int] = None,
        *,
        sensor_ts_ns: Optional[int] = None,
        pose_clock: Optional[str] = None,
    ) -> bool:
        """Latest-value MAILBOX for the robot pose (passthrough from the sensor;
        RTSM never computes or filters pose). Returns True when accepted.

        ONE writer per process: the receiver, at receive time, for every
        tracking-normal frame (websocket / replay / zeromq). The pipeline does
        not write the pose (P1 task 4 removed its dequeue-time write, which
        was the only reason a compare-and-set ever had to arbitrate).

        The key ``(frame_epoch, sensor_ts_ns)`` is therefore a REGRESSION
        DETECTOR, not a gate: a same-epoch write whose stamp went backwards is
        ACCEPTED (the sender says this is the pose now -- a looped replay, a
        bag loop, a restarted bridge, a stepped clock) and counted in
        ``sensor_ts_regressions`` with a WARNING at most once a minute.
        ``timestamp`` (the wall value shown to agents) is the fallback key
        when a side has no sensor stamp; a stamp <= 0 counts as none.

        Two writes ARE rejected, both impossible from the single shipped
        writer and kept as a safety net for embedders that add one:
          writer epoch < stored epoch            -> older_epoch
          writer epoch None, stored epoch known  -> epochless_into_epoch
        A writer epoch above the stored one (new streaming session) and a
        first epoch-bearing write into an epoch-less mailbox are accepted.

        ``pose_clock`` tags where ``timestamp`` came from: ``sender`` (the
        sender's own wall clock) or ``server`` (this process's time.time());
        None keeps the stored tag.
        """
        ts = float(timestamp)
        s_ns = int(sensor_ts_ns) if (sensor_ts_ns is not None and int(sensor_ts_ns) > 0) else None
        epoch = int(frame_epoch) if frame_epoch is not None else None    # numpy ints -> JSON-safe ints
        now_mono = time.monotonic()
        with self._lock:
            lp = self._latest_pose
            reason = self._pose_reject_reason(lp, epoch)
            if reason is not None:
                self._pose_rejects[reason] += 1
                self._pose_consecutive_rejects += 1
                n = self._pose_consecutive_rejects
                if n in (1, 100, 1000, 10000) or (n > 10000 and n % 100000 == 0):
                    logger.warning(
                        "[WM] robot_pose: write rejected (%s; %d consecutive): writer epoch=%s vs stored epoch=%s. "
                        "Only a second writer can produce this -- every shipped receiver writes at receive "
                        "time with its own epoch.", reason, n, epoch, lp.get("frame_epoch"),
                    )
                return False
            self._pose_consecutive_rejects = 0
            if lp is not None and self._pose_is_regression(lp, epoch, ts, s_ns):
                self._pose_regressions += 1
                if now_mono - self._pose_last_regression_warn_mono >= self._POSE_REGRESSION_WARN_EVERY_S:
                    self._pose_last_regression_warn_mono = now_mono
                    logger.warning(
                        "[WM] robot_pose: pose key went backwards within epoch %s (sensor stamp %s -> %s, "
                        "timestamp %.3f -> %.3f; %d regression(s) so far): the sender restarted or stepped its "
                        "clock (looped replay, bag loop, bridge restart). The pose follows the sender; a new "
                        "session would bump frame_epoch instead.",
                        epoch, lp.get("sensor_ts_ns"), s_ns, float(lp["timestamp"]), ts, self._pose_regressions,
                    )
            self._pose_writes_accepted += 1
            clock = pose_clock if pose_clock is not None else (lp.get("pose_clock") if lp else None)
            self._latest_pose = {
                "xyz": t_wc.tolist() if hasattr(t_wc, 'tolist') else list(t_wc),
                "quaternion_xyzw": q_wc_xyzw.tolist() if hasattr(q_wc_xyzw, 'tolist') else list(q_wc_xyzw),
                "timestamp": ts,
                "frame_epoch": epoch,
                "sensor_ts_ns": s_ns,
                "pose_clock": clock,
            }
            self._latest_pose_arrival_mono = now_mono
            return True

    @classmethod
    def _pose_reject_reason(cls, lp: Optional[Dict[str, Any]], epoch: Optional[int]) -> Optional[str]:
        """Second-writer safety net; None = accept."""
        if lp is None:
            return None
        lp_epoch = lp.get("frame_epoch")
        if epoch is None:
            return cls.POSE_REJECT_EPOCHLESS if lp_epoch is not None else None
        if lp_epoch is not None and epoch < lp_epoch:
            return cls.POSE_REJECT_OLDER_EPOCH
        return None

    @staticmethod
    def _pose_is_regression(lp: Dict[str, Any], epoch: Optional[int], ts: float, s_ns: Optional[int]) -> bool:
        """Same epoch (or both epoch-less) and the ordering key went backwards."""
        lp_epoch = lp.get("frame_epoch")
        if epoch != lp_epoch:
            return False            # a new session (epoch above) or the first epoch: a restart, not a regression
        lp_s = lp.get("sensor_ts_ns")
        if s_ns is not None and lp_s is not None:
            return s_ns < int(lp_s)
        return ts < float(lp["timestamp"])

    def _robot_pose_payload(self) -> Optional[Dict[str, Any]]:
        """The stored pose plus read-time diagnostics: age_s on this process's
        clock since the last accepted write (never wall-now minus a sender
        stamp), the stale flag, and the mailbox counters. Built under the WM
        lock (no torn read); a fresh dict each call (callers may mutate)."""
        with self._lock:
            lp = self._latest_pose
            if lp is None:
                return None
            age = max(0.0, time.monotonic() - self._latest_pose_arrival_mono)
            out = dict(lp)
            out["xyz"] = list(lp["xyz"])
            out["quaternion_xyzw"] = list(lp["quaternion_xyzw"])
            out["age_s"] = round(age, 3)
            out["stale"] = bool(age > self._pose_stale_after_s)
            out["stale_after_s"] = self._pose_stale_after_s
            out["writes_accepted"] = int(self._pose_writes_accepted)
            out["sensor_ts_regressions"] = int(self._pose_regressions)
            out["rejected_writes"] = int(sum(self._pose_rejects.values()))
            out["rejected_by_reason"] = dict(self._pose_rejects)
            return out

    def get_robot_pose(self) -> Optional[Dict[str, Any]]:
        """The latest robot pose with diagnostics (see _robot_pose_payload), or
        None if no pose has arrived yet. Written at receive time by every
        receiver (P1 tasks 2 and 4); the pipeline does not write it."""
        return self._robot_pose_payload()

    def set_forward_clearance(self, clearance_m: float, valid_frac: float,
                              timestamp: float) -> None:
        """Store the latest depth-derived forward clearance (meters of open
        space ahead of the camera). clearance_m = 0.0 means blocked or
        unmeasurable (fail-closed)."""
        with self._lock:
            self._latest_clearance = {
                "clearance_m": float(clearance_m),
                "valid_frac": float(valid_frac),
                "timestamp": float(timestamp),
            }

    def get_forward_clearance(self) -> Optional[Dict[str, Any]]:
        return self._latest_clearance

    def note_label_detections(self, labels, scores=None) -> None:
        """Tally one frame's detector labels (see _label_detections).
        Bounded: at most 64 distinct labels are tracked (vocab-prompted
        backends emit a handful; the guard is for open-vocab defaults)."""
        if not labels:
            return
        now = time.time()
        with self._lock:
            for i, label in enumerate(labels):
                if label is None:
                    continue
                key = str(label)
                rec = self._label_detections.get(key)
                if rec is None:
                    if len(self._label_detections) >= 64:
                        continue
                    rec = {"n": 0, "last_seen_wall": 0.0,
                           "last_score": None, "max_score": None}
                    self._label_detections[key] = rec
                rec["n"] += 1
                rec["last_seen_wall"] = now
                s = None
                if scores is not None:
                    try:
                        s = float(scores[i])
                    except (IndexError, TypeError, ValueError):
                        s = None
                if s is not None:
                    rec["last_score"] = round(s, 4)
                    if rec["max_score"] is None or s > rec["max_score"]:
                        rec["max_score"] = round(s, 4)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self._map)
            c = sum(1 for o in self._map.values() if o.confirmed)
            avg_hits = (sum(o.hits for o in self._map.values()) / n) if n else 0.0
            # Confirmed objects that have never reached the vector store —
            # should hover near 0 (flush latency only); a large steady value
            # means semantic retrieval is starving.
            never_upserted = sum(
                1 for o in self._map.values()
                if o.confirmed and float(o.last_upsert_mono or 0.0) == 0.0
            )
            out = {
                "objects": n,
                "confirmed": c,
                "avg_hits": avg_hits,
                "upserts_total": int(self._upsert_count_total),
                "ltm_never_upserted": never_upserted,
                "robot_pose": self._robot_pose_payload(),
                "detections_by_label": {
                    k: dict(v) for k, v in self._label_detections.items()
                },
            }
            # Published only when receive-time clearance sensing is enabled
            # (io.clearance.enable): None until the first depth frame, then
            # {clearance_m, valid_frac, timestamp}. Flag off => key absent,
            # so /stats matches a build without the feature (the rc_car_agent
            # client treats absent and None alike: no sample).
            if self.clearance_enabled:
                out["forward_clearance"] = self._latest_clearance
            return out

    def clear(self) -> Dict[str, int]:
        """
        Clear all objects from working memory.

        Clears object map, scheduling heaps, and resets counters.
        Also clears the attached spatial index if present.

        Returns dict with counts of what was cleared.
        """
        with self._lock:
            obj_count = len(self._map)
            confirmed_count = sum(1 for o in self._map.values() if o.confirmed)
            proto_count = obj_count - confirmed_count

            # Clear object map
            self._map.clear()

            # Clear scheduling heaps and reverse index
            self._proto_heap.clear()
            self._ltm_heap.clear()
            self._frame_to_objects.clear()

            # Reset counters
            self._upsert_count_total = 0
            self._label_detections = {}

            # Clear attached spatial index if present
            if self.index is not None:
                self.index.clear()

            # Reset the robot-pose mailbox so re-fed older stamps (replaying a
            # recording after a /reset, a rig restart) are accepted again, and
            # zero its counters.
            self._latest_pose = None
            self._latest_pose_arrival_mono = 0.0
            self._pose_writes_accepted = 0
            for k in self._pose_rejects:
                self._pose_rejects[k] = 0
            self._pose_consecutive_rejects = 0
            self._pose_regressions = 0
            self._pose_last_regression_warn_mono = float("-inf")
            # Drop the last clearance sample too (stale after a reset).
            self._latest_clearance = None

            logger.info(f"[WM] Cleared {obj_count} objects ({confirmed_count} confirmed, {proto_count} proto)")

            return {
                "objects_cleared": obj_count,
                "confirmed_cleared": confirmed_count,
                "proto_cleared": proto_count,
            }
