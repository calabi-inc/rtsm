"""
External detections: the contract, the adapter protocol and the standard
``vision_msgs`` adapters (P3, the detections adapter).

A customer who already runs a detector feeds its output to RTSM and gets the
memory and the report on *their* detections. Everything here is pure NumPy /
plain Python; nothing imports a model.

Contract (DETECTIONS_CONTRACT 1, additive like the ingest contracts)
---------------------------------------------------------------------
``Detections`` carries, per frame, boxes in PIXELS AT THE RGB RESOLUTION, the
top label per detection with every hypothesis behind it, optional scores,
optional track ids and optional masks. **``scores`` is None when the detector
reports none** -- it is never a vector of made-up constants; a message in which
only some detections are scored carries NaN for the unscored ones. The 2026
VLM detectors (LocateAnything, Qwen3-VL, Gemma 4, Moondream) emit no
confidence at all, so the unscored case is first class: the ``external``
segmentation backend and the pipeline's label merge treat it explicitly (the
customer's label is stored with a configured prior and the observation ledger
records ``score: null``), and the report header says ``scores: absent``.

Adapters
--------
An adapter converts one message into a ``Detections``. The registry maps a
message type to its adapter; the bag reader asks ``adapter_for(msgtype)`` for
the detections topic it discovered. Third-party adapters register with
``register_detections_adapter`` (an entry-point group can follow when one
exists outside this repository).

* ``VisionMsgs2DAdapter`` -- ``vision_msgs/msg/Detection2DArray`` in both the
  ROS 2 layout (``results[].hypothesis.class_id`` / ``.score``,
  ``bbox.center.position.x`` / ``.y``, ``bbox.center.theta``) and the ROS 1
  layout (``results[].id`` int64 / ``.score``, ``bbox.center.x`` / ``.y`` /
  ``.theta``, ``source_img`` with the image size the boxes refer to). rosbags
  reports both under the same ``vision_msgs/msg/...`` name; the shape is
  told apart by the fields. A rotated box (theta != 0) becomes the bounding
  box of the rotated rectangle. Boxes are rescaled when the message's source
  image size (ROS 1 ``source_img``, or the ``image_hw`` option) differs from
  the paired RGB.
* ``VisionMsgs3DAdapter`` -- ``vision_msgs/msg/Detection3DArray``: the box's
  eight corners go from the detection header's frame to the camera optical
  frame through the caller's TF lookup, are projected with the RGB
  intrinsics, and the 2-D box is the bounding box of the projected corners.
  A detection whose frame cannot be resolved, that lies behind the camera or
  entirely outside the image is dropped and counted -- never guessed.

Masks
-----
``boxes_to_masks`` turns boxes into the instance masks the pipeline's
heuristics need: inside each box, the pixels within ``band_m`` of the box's
median depth, largest connected component (so the wall behind a mug leaves
the mask); the full box when depth is missing, too sparse, or
``mask_from="box"``. Depth may be at a lower resolution than the RGB (the
Lens path: 256x192 under 1920x1440): the band is computed at depth resolution
and resampled nearest-neighbour into the box.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DETECTIONS_CONTRACT = 1
SOURCE_EXTERNAL = "external"
DETECTION_2D = "vision_msgs/msg/Detection2DArray"
DETECTION_3D = "vision_msgs/msg/Detection3DArray"

# why a detection of a message was dropped (counted in Detections.n_dropped and the bag stats)
DROP_DEGENERATE = "degenerate_box"
DROP_OUTSIDE = "outside_image"
DROP_BEHIND = "behind_camera"
DROP_UNRESOLVED_FRAME = "unresolved_frame"
DROP_NO_INTRINSICS = "no_intrinsics"

Hypothesis = Tuple[str, Optional[float]]


# ───────────────────────────── contract ─────────────────────────────

@dataclass
class Detections:
    """One frame's external detections (see the module docstring)."""
    boxes_xyxy: np.ndarray                                   # [N,4] float32, pixels at the RGB resolution
    scores: Optional[np.ndarray]                             # [N] float32; None = the detector reports none; NaN = this one unscored
    labels: List[Optional[str]]                              # [N] the top hypothesis (None when the detection has no hypothesis)
    hypotheses: List[List[Hypothesis]]                       # [N] every hypothesis, best first: (label, score | None)
    ids: Optional[List[Optional[str]]] = None                # [N] track / detection ids when the source emits them
    masks: Optional[np.ndarray] = None                       # [N,H,W] bool when the source provides masks
    source: str = SOURCE_EXTERNAL                            # detector name / topic
    msgtype: Optional[str] = None
    t_sensor_ns: Optional[int] = None
    frame_id: Optional[str] = None
    n_dropped: Dict[str, int] = field(default_factory=dict)  # reason -> count, for the detections of this message

    @property
    def count(self) -> int:
        return int(self.boxes_xyxy.shape[0]) if self.boxes_xyxy is not None else 0

    @property
    def n_scored(self) -> int:
        if self.scores is None:
            return 0
        return int(np.isfinite(self.scores).sum())

    @property
    def scoring(self) -> str:
        """``absent`` | ``present`` | ``mixed`` for this message."""
        if self.count == 0:
            return "absent" if self.scores is None else "present"
        n = self.n_scored
        return "absent" if n == 0 else "present" if n == self.count else "mixed"

    @staticmethod
    def empty(**kw: Any) -> "Detections":
        return Detections(boxes_xyxy=np.zeros((0, 4), dtype=np.float32), scores=None, labels=[], hypotheses=[], **kw)


class DetectionsAdapter(Protocol):
    """One message -> one ``Detections``."""
    name: str
    msgtypes: Tuple[str, ...]

    def convert(self, msg: Any, msgtype: str, *, rgb_hw: Tuple[int, int], intrinsics: Any = None,
                tf_lookup: Optional[Callable[[str, int], np.ndarray]] = None, t_sensor_ns: Optional[int] = None,
                source: str = SOURCE_EXTERNAL, image_hw: Optional[Tuple[int, int]] = None) -> Detections: ...


_ADAPTERS: Dict[str, DetectionsAdapter] = {}


def register_detections_adapter(adapter: DetectionsAdapter) -> None:
    for mt in adapter.msgtypes:
        _ADAPTERS[mt] = adapter


def adapter_for(msgtype: str) -> Optional[DetectionsAdapter]:
    return _ADAPTERS.get(msgtype)


def detection_msgtypes() -> Tuple[str, ...]:
    return tuple(_ADAPTERS)


# ───────────────────────────── helpers ─────────────────────────────

def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _hypotheses(results: Sequence[Any]) -> List[Hypothesis]:
    """ROS 2: results[].hypothesis.class_id / .score; ROS 1: results[].id (int64) / .score.
    Best first (a scored hypothesis beats an unscored one; ties keep the message order)."""
    out: List[Hypothesis] = []
    for r in results or ():
        hyp = getattr(r, "hypothesis", None)
        if hyp is not None:
            label = getattr(hyp, "class_id", None)
            score = _finite(getattr(hyp, "score", None))
        else:
            raw = getattr(r, "id", None)
            label = (str(raw) if raw is not None else None)
            score = _finite(getattr(r, "score", None))
        label = (str(label) if label not in (None, "") else None)
        if label is None and score is None:
            continue
        out.append((label if label is not None else "", score))
    out.sort(key=lambda h: (0 if h[1] is not None else 1, -(h[1] if h[1] is not None else 0.0)))
    return out


def _stamp_ns(header: Any) -> Optional[int]:
    st = getattr(header, "stamp", None)
    if st is None:
        return None
    sec = getattr(st, "sec", None); nsec = getattr(st, "nanosec", None)
    if nsec is None:
        nsec = getattr(st, "nsec", 0)
    try:
        v = int(sec) * 1_000_000_000 + int(nsec)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _rotated_box_aabb(cx: float, cy: float, w: float, h: float, theta: float) -> Tuple[float, float, float, float]:
    if not theta:
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    hw, hh = (w * c + h * s) / 2, (w * s + h * c) / 2
    return cx - hw, cy - hh, cx + hw, cy + hh


def _finish(boxes: List[Tuple[float, float, float, float]], hyps: List[List[Hypothesis]], ids: List[Optional[str]],
            dropped: Dict[str, int], rgb_hw: Tuple[int, int], scale: Tuple[float, float], **meta: Any) -> Detections:
    """Scale, clip and drop; build the arrays."""
    H, W = int(rgb_hw[0]), int(rgb_hw[1])
    sx, sy = scale
    keep_boxes: List[List[float]] = []; keep_h: List[List[Hypothesis]] = []; keep_ids: List[Optional[str]] = []
    for (x0, y0, x1, y1), hs, did in zip(boxes, hyps, ids):
        x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
            dropped[DROP_DEGENERATE] = dropped.get(DROP_DEGENERATE, 0) + 1
            continue
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
        if cx1 <= cx0 or cy1 <= cy0:
            dropped[DROP_OUTSIDE] = dropped.get(DROP_OUTSIDE, 0) + 1
            continue
        keep_boxes.append([cx0, cy0, cx1, cy1]); keep_h.append(hs); keep_ids.append(did)
    n = len(keep_boxes)
    arr = np.asarray(keep_boxes, dtype=np.float32).reshape(n, 4)
    labels = [(hs[0][0] if hs and hs[0][0] != "" else None) for hs in keep_h]
    top_scores = [(hs[0][1] if hs else None) for hs in keep_h]
    if n and any(s is not None for s in top_scores):
        scores: Optional[np.ndarray] = np.asarray([(s if s is not None else np.nan) for s in top_scores], dtype=np.float32)
    else:
        scores = None
    return Detections(boxes_xyxy=arr, scores=scores, labels=labels, hypotheses=keep_h,
                      ids=(keep_ids if any(i is not None for i in keep_ids) else None), n_dropped=dropped, **meta)


# ───────────────────────────── vision_msgs 2-D ─────────────────────────────

class VisionMsgs2DAdapter:
    name = "vision_msgs_2d"
    msgtypes = (DETECTION_2D,)

    def convert(self, msg: Any, msgtype: str, *, rgb_hw: Tuple[int, int], intrinsics: Any = None,
                tf_lookup: Optional[Callable[[str, int], np.ndarray]] = None, t_sensor_ns: Optional[int] = None,
                source: str = SOURCE_EXTERNAL, image_hw: Optional[Tuple[int, int]] = None) -> Detections:
        header = getattr(msg, "header", None)
        dets = list(getattr(msg, "detections", None) or [])
        boxes: List[Tuple[float, float, float, float]] = []; hyps: List[List[Hypothesis]] = []; ids: List[Optional[str]] = []
        dropped: Dict[str, int] = {}
        src_hw: Optional[Tuple[int, int]] = image_hw
        for d in dets:
            bb = getattr(d, "bbox", None)
            center = getattr(bb, "center", None)
            pos = getattr(center, "position", None)               # ROS 2 Pose2D(position: Point2D, theta)
            if pos is not None:
                cx, cy = _finite(getattr(pos, "x", None)), _finite(getattr(pos, "y", None))
            else:                                                 # ROS 1 geometry_msgs/Pose2D(x, y, theta)
                cx, cy = _finite(getattr(center, "x", None)), _finite(getattr(center, "y", None))
            theta = _finite(getattr(center, "theta", 0.0)) or 0.0
            w, h = _finite(getattr(bb, "size_x", None)), _finite(getattr(bb, "size_y", None))
            if None in (cx, cy, w, h):
                dropped[DROP_DEGENERATE] = dropped.get(DROP_DEGENERATE, 0) + 1
                continue
            if src_hw is None:                                    # ROS 1 Detection2D.source_img carries the image the boxes refer to
                img = getattr(d, "source_img", None)
                ih, iw = int(getattr(img, "height", 0) or 0), int(getattr(img, "width", 0) or 0)
                if ih > 0 and iw > 0:
                    src_hw = (ih, iw)
            boxes.append(_rotated_box_aabb(cx, cy, w, h, theta))
            hyps.append(_hypotheses(getattr(d, "results", None)))
            did = getattr(d, "id", None)
            ids.append(str(did) if did not in (None, "") else None)
        scale = (1.0, 1.0)
        if src_hw is not None and tuple(src_hw) != tuple(rgb_hw) and src_hw[0] > 0 and src_hw[1] > 0:
            scale = (rgb_hw[1] / float(src_hw[1]), rgb_hw[0] / float(src_hw[0]))
        return _finish(boxes, hyps, ids, dropped, rgb_hw, scale, source=source, msgtype=msgtype,
                       t_sensor_ns=(t_sensor_ns if t_sensor_ns is not None else _stamp_ns(header)),
                       frame_id=(str(getattr(header, "frame_id", "")) or None))


# ───────────────────────────── vision_msgs 3-D ─────────────────────────────

def _quat_to_R(q: Any) -> np.ndarray:
    x, y, z, w = (float(getattr(q, k, 0.0)) for k in ("x", "y", "z", "w"))
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)


def _intrinsics_tuple(intr: Any) -> Optional[Tuple[float, float, float, float]]:
    if intr is None:
        return None
    try:
        return float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)
    except AttributeError:
        try:
            fx, fy, cx, cy = intr
            return float(fx), float(fy), float(cx), float(cy)
        except (TypeError, ValueError):
            return None


class VisionMsgs3DAdapter:
    """Detection3DArray -> 2-D boxes through TF (``tf_lookup(frame_id, t_ns) -> T_cam_from_frame`` 4x4) and the intrinsics."""
    name = "vision_msgs_3d"
    msgtypes = (DETECTION_3D,)

    def convert(self, msg: Any, msgtype: str, *, rgb_hw: Tuple[int, int], intrinsics: Any = None,
                tf_lookup: Optional[Callable[[str, int], np.ndarray]] = None, t_sensor_ns: Optional[int] = None,
                source: str = SOURCE_EXTERNAL, image_hw: Optional[Tuple[int, int]] = None) -> Detections:
        header = getattr(msg, "header", None)
        stamp = t_sensor_ns if t_sensor_ns is not None else _stamp_ns(header)
        frame = str(getattr(header, "frame_id", "")) or None
        dets = list(getattr(msg, "detections", None) or [])
        dropped: Dict[str, int] = {}
        K = _intrinsics_tuple(intrinsics)
        meta = dict(source=source, msgtype=msgtype, t_sensor_ns=stamp, frame_id=frame)
        if K is None:
            dropped[DROP_NO_INTRINSICS] = len(dets)
            return Detections.empty(n_dropped=dropped, **meta)
        fx, fy, cx, cy = K
        H, W = int(rgb_hw[0]), int(rgb_hw[1])
        T_cache: Dict[str, Optional[np.ndarray]] = {}

        def T_for(fid: Optional[str]) -> Optional[np.ndarray]:
            key = fid or ""
            if key not in T_cache:
                T: Optional[np.ndarray] = None
                if tf_lookup is not None and fid and stamp is not None:
                    try:
                        T = np.asarray(tf_lookup(fid, int(stamp)), dtype=float).reshape(4, 4)
                    except Exception as e:  # noqa: BLE001 -- a missing frame is a counted drop
                        logger.debug("[detections] frame %r unresolved at %s: %s", fid, stamp, e)
                        T = None
                T_cache[key] = T
            return T_cache[key]

        boxes: List[Tuple[float, float, float, float]] = []; hyps: List[List[Hypothesis]] = []; ids: List[Optional[str]] = []
        for d in dets:
            dh = getattr(d, "header", None)
            fid = (str(getattr(dh, "frame_id", "")) or None) or frame
            T = T_for(fid)
            if T is None:
                dropped[DROP_UNRESOLVED_FRAME] = dropped.get(DROP_UNRESOLVED_FRAME, 0) + 1
                continue
            bb = getattr(d, "bbox", None)
            center = getattr(bb, "center", None); size = getattr(bb, "size", None)
            pos = getattr(center, "position", None); ori = getattr(center, "orientation", None)
            try:
                c = np.array([float(pos.x), float(pos.y), float(pos.z)]); sz = np.array([float(size.x), float(size.y), float(size.z)])
            except (AttributeError, TypeError, ValueError):
                dropped[DROP_DEGENERATE] = dropped.get(DROP_DEGENERATE, 0) + 1
                continue
            if not (np.all(np.isfinite(c)) and np.all(np.isfinite(sz)) and np.all(sz > 0)):
                dropped[DROP_DEGENERATE] = dropped.get(DROP_DEGENERATE, 0) + 1
                continue
            R = _quat_to_R(ori) if ori is not None else np.eye(3)
            half = sz / 2.0
            corners = np.array([[sx, sy, sz_] for sx in (-half[0], half[0]) for sy in (-half[1], half[1]) for sz_ in (-half[2], half[2])])
            pts = (R @ corners.T).T + c                               # in the detection frame
            cam = (T[:3, :3] @ pts.T).T + T[:3, 3]                     # camera optical frame
            if np.any(cam[:, 2] <= 1e-6):
                dropped[DROP_BEHIND] = dropped.get(DROP_BEHIND, 0) + 1
                continue
            u = fx * cam[:, 0] / cam[:, 2] + cx; v = fy * cam[:, 1] / cam[:, 2] + cy
            boxes.append((float(u.min()), float(v.min()), float(u.max()), float(v.max())))
            hyps.append(_hypotheses(getattr(d, "results", None)))
            did = getattr(d, "id", None)
            ids.append(str(did) if did not in (None, "") else None)
        return _finish(boxes, hyps, ids, dropped, (H, W), (1.0, 1.0), **meta)


register_detections_adapter(VisionMsgs2DAdapter())
register_detections_adapter(VisionMsgs3DAdapter())


# ───────────────────────────── masks ─────────────────────────────

def boxes_to_masks(boxes_xyxy: np.ndarray, rgb_hw: Tuple[int, int], depth_m: Optional[np.ndarray] = None, *,
                   band_m: float = 0.20, mask_from: str = "depth_band", min_valid_frac: float = 0.2) -> Tuple[np.ndarray, Dict[str, int]]:
    """Instance masks [N,H,W] bool for the boxes (pixels at the RGB resolution).
    ``depth_band``: inside each box, the pixels within ``band_m`` of the box's
    median depth, largest connected component containing the median pixel
    (depth at its own resolution, resampled nearest into the box); the box
    itself when depth is missing / too sparse (``min_valid_frac``) or when
    ``mask_from == "box"``. Returns the masks and {"depth_band": n, "box": n}."""
    H, W = int(rgb_hw[0]), int(rgb_hw[1])
    n = int(boxes_xyxy.shape[0]) if boxes_xyxy is not None else 0
    masks = np.zeros((n, H, W), dtype=bool)
    how = {"depth_band": 0, "box": 0}
    if n == 0:
        return masks, how
    use_depth = (mask_from == "depth_band" and depth_m is not None and getattr(depth_m, "ndim", 0) == 2 and depth_m.size > 0)
    dH, dW = (depth_m.shape[:2] if use_depth else (H, W))
    sy, sx = dH / float(H), dW / float(W)
    for i in range(n):
        x0, y0, x1, y1 = boxes_xyxy[i]
        X0, Y0 = int(max(0, math.floor(x0))), int(max(0, math.floor(y0)))
        X1, Y1 = int(min(W, math.ceil(x1))), int(min(H, math.ceil(y1)))
        if X1 <= X0 or Y1 <= Y0:
            continue
        if use_depth:
            dx0, dy0 = int(math.floor(X0 * sx)), int(math.floor(Y0 * sy))
            dx1, dy1 = int(max(dx0 + 1, math.ceil(X1 * sx))), int(max(dy0 + 1, math.ceil(Y1 * sy)))
            dx1, dy1 = min(dW, dx1), min(dH, dy1)
            patch = np.asarray(depth_m[dy0:dy1, dx0:dx1], dtype=np.float32)
            valid = np.isfinite(patch) & (patch > 0)
            if patch.size and valid.mean() >= min_valid_frac:
                # The reference depth is the median of the CENTRAL half of the box: a detector's box is centred
                # on its object, while a loose box can hold more background than object (the median of the whole
                # box would then pick the wall). Falls back to the whole box when the centre has no depth.
                ph, pw = patch.shape
                cy0, cy1 = ph // 4, max(ph // 4 + 1, (3 * ph + 3) // 4)
                cx0, cx1 = pw // 4, max(pw // 4 + 1, (3 * pw + 3) // 4)
                centre = patch[cy0:cy1, cx0:cx1]; cvalid = valid[cy0:cy1, cx0:cx1]
                med = float(np.median(centre[cvalid])) if cvalid.any() else float(np.median(patch[valid]))
                band = valid & (np.abs(patch - med) <= float(band_m))
                band = _largest_component(band, patch, med)
                if band.any():
                    # exact nearest-neighbour mapping of every RGB pixel of the box onto its depth pixel
                    rows = np.clip((np.arange(Y0, Y1) * sy).astype(int) - dy0, 0, band.shape[0] - 1)
                    cols = np.clip((np.arange(X0, X1) * sx).astype(int) - dx0, 0, band.shape[1] - 1)
                    masks[i, Y0:Y1, X0:X1] = band[rows[:, None], cols[None, :]]
                    how["depth_band"] += 1
                    continue
        masks[i, Y0:Y1, X0:X1] = True
        how["box"] += 1
    return masks, how


def _largest_component(band: np.ndarray, patch: np.ndarray, med: float) -> np.ndarray:
    """The connected component of ``band`` that contains the pixel closest to the
    median depth (the object itself), else the largest component."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover -- scipy is a core dependency
        return band
    lab, n = ndimage.label(band)
    if n <= 1:
        return band
    diff = np.where(band, np.abs(patch - med), np.inf)
    seed = np.unravel_index(int(np.argmin(diff)), diff.shape)
    seed_lab = int(lab[seed])
    if seed_lab > 0:
        return lab == seed_lab
    sizes = ndimage.sum(band, lab, index=range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)
