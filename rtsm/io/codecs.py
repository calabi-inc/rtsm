"""
Codec layer (Gate 4.5 plan, P3 task 0.5): encoding-keyed pure functions.

Everything that turns transport bytes into engine arrays lives here, once:
  * RGB payload -> BGR uint8 (the channel-order contract: FramePacket.rgb is
    BGR; a source that delivers RGB natively converts HERE, keyed on its
    declared encoding -- never in the pipeline, never by pixel statistics);
  * depth payload -> float32 metres (NaN = invalid, except the RTAB-Map bridge
    convention ``png_uint16_raw`` which keeps zeros, as that receiver always
    did);
  * confidence map, confidence filter, intrinsics rescale;
  * pose parse (ARKit matrix / quaternion, RTAB-Map Euler, prepared) and the
    camera-convention normalisation (ARKit -> OpenCV, applied once at ingest).

An unknown encoding / format / convention raises ``UnsupportedEncoding``
(a ValueError): explicit, never a silent default.

The functions moved from ``rtsm/io/websocket.py`` keep their names and
behaviour; that module re-exports them so existing imports work.
"""
from __future__ import annotations

import logging
import struct
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from rtsm.core.datamodel import PinholeIntrinsics, PoseStamped
from rtsm.utils.transforms import euler_to_quat_xyzw, rotmat_to_quat_xyzw

logger = logging.getLogger(__name__)

RGB_ENCODINGS = ("jpeg", "png", "bgra", "nv12", "raw_bgr")
DEPTH_ENCODINGS = ("uint16_mm", "float32_m", "png_uint16", "png_uint16_raw", "raw_depth_m")
POSE_FORMATS = ("matrix4x4_col_major", "quat_translation", "rtabmap_euler", "prepared")
CONVENTIONS = ("arkit", "opencv")

# ARKit camera (Y-up, Z-toward-viewer) -> OpenCV camera (Y-down, Z-forward).
# Right-multiplying T_wc by this converts camera columns from ARKit to OpenCV
# convention. diag(1, -1, -1, 1) is its own inverse.
_ARKIT_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


class UnsupportedEncoding(ValueError):
    """An encoding / pose format / convention the codec layer does not know."""


# ───────────────────────────── RGB ─────────────────────────────

def decode_rgb(raw: Any, fmt: str, width: int, height: int) -> np.ndarray:
    """Decode an RGB payload into an (H, W, 3) uint8 BGR array (OpenCV order).

    fmt: ``jpeg`` | ``png`` | ``bgra`` | ``nv12`` | ``raw_bgr`` (already an array).
    Raises ValueError on a decode failure, UnsupportedEncoding on a format
    this layer does not know.
    """
    if fmt == "raw_bgr":
        if not isinstance(raw, np.ndarray):
            raise ValueError("raw_bgr expects an ndarray")
        return raw
    if fmt.startswith("ros:"):                       # bag sources: "ros:<sensor_msgs/Image encoding>", row padding already stripped
        return ros_image_to_bgr(fmt[4:], raw, width, height)
    if fmt.startswith("rosc:"):                      # bag sources: "rosc:<CompressedImage format>"
        return ros_compressed_to_bgr(fmt[5:], raw)
    if fmt in ("jpeg", "png"):
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imdecode failed for {fmt} ({len(raw)} bytes)")
        return img
    if fmt == "bgra":
        expected = height * width * 4
        if len(raw) != expected:
            raise ValueError(
                f"bgra buffer size mismatch: got {len(raw)}, "
                f"expected {expected} ({width}x{height}x4)"
            )
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if fmt == "nv12":
        # NV12: Y plane (H*W) + interleaved UV plane (H/2 * W); total H * W * 3 / 2
        expected = height * width * 3 // 2
        if len(raw) != expected:
            raise ValueError(
                f"nv12 buffer size mismatch: got {len(raw)}, "
                f"expected {expected} ({width}x{height} * 1.5)"
            )
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    raise UnsupportedEncoding(f"Unsupported rgb_format: {fmt!r}")


# ───────────────────────────── depth ─────────────────────────────

def decode_depth(
    raw: Any,
    fmt: Optional[str],
    width: int,
    height: int,
    depth_scale: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Decode a depth payload into (H, W) float32 metres with NaN for invalid
    pixels (wire convention 0 = invalid). ``png_uint16_raw`` is the RTAB-Map
    bridge convention: PNG uint16 scaled to metres with zeros KEPT (that
    receiver never masked them; changing it would change every packet).

    Returns None for no depth (``fmt`` None or an empty payload), and -- as
    before this layer existed -- None with a warning when a PNG payload fails
    to decode. Raises UnsupportedEncoding on an unknown format.
    """
    if fmt is None or raw is None:
        return None
    if fmt == "raw_depth_m":
        return raw if isinstance(raw, np.ndarray) else None
    if len(raw) == 0:
        return None
    if depth_scale is None:
        depth_scale = 0.001  # default: millimetres

    if fmt == "uint16_mm":
        depth_u16 = np.frombuffer(raw, dtype=np.uint16).reshape(height, width)
        depth_m = depth_u16.astype(np.float32) * depth_scale
        depth_m[depth_u16 == 0] = np.nan
        return depth_m
    if fmt == "float32_m":
        depth_m = np.frombuffer(raw, dtype=np.float32).reshape(height, width).copy()
        depth_m[depth_m == 0.0] = np.nan
        return depth_m
    if fmt in ("png_uint16", "png_uint16_raw"):
        buf = np.frombuffer(raw, dtype=np.uint8)
        depth_u16 = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            logger.warning("[codecs] failed to decode PNG depth")
            return None
        depth_m = depth_u16.astype(np.float32) * float(depth_scale)
        if fmt == "png_uint16":
            depth_m[depth_u16 == 0] = np.nan
        return depth_m
    raise UnsupportedEncoding(f"Unsupported depth_format: {fmt!r}")


def depth_valid_fraction(depth_m: Optional[np.ndarray]) -> Optional[float]:
    """Fraction of finite depth pixels (the P2 pose-ledger field for dropped
    frames). None when there is no depth."""
    if depth_m is None or getattr(depth_m, "size", 0) == 0:
        return None
    return float(np.isfinite(depth_m).mean())


def forward_clearance_from_depth(depth_m: Optional[np.ndarray],
                                 min_valid_frac: float = 0.2) -> Tuple[float, float]:
    """Metres of open space ahead of the camera, from one decoded depth frame.
    RECEIVE-TIME wall-guard sensing (2026-08-16): frame-packet level, before
    the ingest queue / gate and any GPU work, so it updates at stream rate.

    Returns (clearance_m, valid_frac). Central band of the image (rows 30-55 %,
    cols 33-66 %); clearance is the 10th percentile of valid depths. Fail-
    closed: a mostly-invalid band returns 0.0."""
    if depth_m is None or depth_m.size == 0:
        return 0.0, 0.0
    h, w = depth_m.shape[:2]
    band = depth_m[int(h * 0.30):int(h * 0.55), int(w * 0.33):int(w * 0.66)]
    if band.size == 0:
        return 0.0, 0.0
    valid = band[np.isfinite(band) & (band > 0.05)]
    frac = float(valid.size) / float(band.size)
    if frac < min_valid_frac:
        return 0.0, frac
    return float(np.percentile(valid, 10)), frac


# ───────────────────────────── confidence ─────────────────────────────

def decode_confidence(raw: Optional[Any], width: int, height: int) -> Optional[np.ndarray]:
    """An ARKit-style uint8 confidence map (0 / 1 / 2), or None when the payload
    is missing or its size does not match the declared dimensions (as the
    websocket receiver always treated it)."""
    if raw is None or width <= 0 or height <= 0:
        return None
    if isinstance(raw, np.ndarray):
        return raw if raw.shape[:2] == (height, width) else None
    if len(raw) != width * height:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)


def apply_confidence_filter(depth_m: Optional[np.ndarray], confidence: Optional[np.ndarray],
                            threshold: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """NaN-mask depth pixels whose confidence is below ``threshold`` (0 = off).
    The map is resized (nearest) to the depth resolution when they differ.
    Mutates ``depth_m``; returns ``(depth_m, confidence_used)`` where the
    second is the resized map when a resize happened -- the FramePacket
    carries THAT map, as the websocket receiver always did.
    """
    if confidence is None or depth_m is None or threshold <= 0:
        return depth_m, confidence
    dep_h, dep_w = depth_m.shape[:2]
    conf_h, conf_w = confidence.shape[:2]
    conf = confidence
    if conf_h != dep_h or conf_w != dep_w:
        conf = cv2.resize(confidence, (dep_w, dep_h), interpolation=cv2.INTER_NEAREST)
    depth_m[conf < threshold] = np.nan
    return depth_m, conf


# ───────────────────────────── intrinsics ─────────────────────────────

def rescale_intrinsics(fx: float, fy: float, cx: float, cy: float, *, from_wh: Tuple[int, int],
                       to_wh: Tuple[int, int]) -> PinholeIntrinsics:
    """Intrinsics declared at ``from_wh`` expressed at ``to_wh`` (the RGB
    resolution the engine works in). A zero / missing dimension leaves the
    values unchanged (the websocket receiver's rule)."""
    iw, ih = int(from_wh[0]), int(from_wh[1])
    tw, th = int(to_wh[0]), int(to_wh[1])
    if iw > 0 and ih > 0 and tw > 0 and th > 0 and (iw != tw or ih != th):
        sx, sy = tw / iw, th / ih
        fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy
    return PinholeIntrinsics(width=tw, height=th, fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


# ───────────────────────────── pose ─────────────────────────────

def parse_arkit_pose(T_wc_data: list, pose_format: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an ARKit pose into canonical (t_wc, q_xyzw) float32 arrays.
    ``matrix4x4_col_major``: 16 floats; ``quat_translation``: 7 floats
    [qx, qy, qz, qw, tx, ty, tz]. Raises ValueError on a wrong element count,
    UnsupportedEncoding on an unknown format."""
    if pose_format == "quat_translation":
        if len(T_wc_data) != 7:
            raise ValueError(f"quat_translation expects 7 elements, got {len(T_wc_data)}")
        qx, qy, qz, qw, tx, ty, tz = (float(v) for v in T_wc_data)
        return (np.array([tx, ty, tz], dtype=np.float32),
                np.array([qx, qy, qz, qw], dtype=np.float32))
    if pose_format == "matrix4x4_col_major":
        if len(T_wc_data) != 16:
            raise ValueError(f"matrix4x4_col_major expects 16 elements, got {len(T_wc_data)}")
        mat = np.array(T_wc_data, dtype=np.float64).reshape(4, 4, order="F")
        t_wc = mat[:3, 3].astype(np.float32)
        q_xyzw = rotmat_to_quat_xyzw(mat[:3, :3].astype(np.float32))
        return t_wc, q_xyzw
    raise UnsupportedEncoding(f"Unsupported pose_format: {pose_format!r}")


def parse_rtabmap_euler(T_wc: list, pose_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """RTAB-Map bridge pose [x, y, z, roll, pitch, yaw] (radians) -> (t_wc, q_xyzw),
    translation scaled by ``pose_scale`` (metres per unit)."""
    x, y, z = float(T_wc[0]), float(T_wc[1]), float(T_wc[2])
    roll, pitch, yaw = float(T_wc[3]), float(T_wc[4]), float(T_wc[5])
    t_wc = np.array([x, y, z], dtype=np.float32) * float(pose_scale)
    return t_wc, euler_to_quat_xyzw(roll, pitch, yaw)


def parse_pose(raw: Any, pose_format: str, *, pose_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch on the declared pose format. ``prepared`` = the adapter already
    produced (t_wc, q_wc_xyzw)."""
    if pose_format == "prepared":
        t, q = raw
        return np.asarray(t, dtype=np.float32), np.asarray(q, dtype=np.float32)
    if pose_format in ("matrix4x4_col_major", "quat_translation"):
        return parse_arkit_pose(raw, pose_format)
    if pose_format == "rtabmap_euler":
        return parse_rtabmap_euler(raw, pose_scale)
    raise UnsupportedEncoding(f"Unsupported pose_format: {pose_format!r}")


def normalize_pose_convention(t_wc: np.ndarray, q_xyzw: np.ndarray, convention: str
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Bring a camera pose into the engine's OpenCV camera convention.
    ``arkit``: right-multiply T_wc by the ARKit->OpenCV flip (applied once at
    ingest so every consumer sees one convention); ``opencv``: identity."""
    if convention == "opencv":
        return t_wc, q_xyzw
    if convention == "arkit":
        T_wc_mat = PoseStamped(stamp_ns=0, frame_id="", t_wc=t_wc, q_wc_xyzw=q_xyzw).T_wc() @ _ARKIT_TO_OPENCV
        return (T_wc_mat[:3, 3].astype(np.float32),
                rotmat_to_quat_xyzw(T_wc_mat[:3, :3].astype(np.float32)))
    raise UnsupportedEncoding(f"Unsupported pose convention: {convention!r}")


def normalize_matrix_convention(T_wc: np.ndarray, convention: str) -> np.ndarray:
    """The same flip for a 4x4 (pose corrections)."""
    if convention == "opencv":
        return T_wc
    if convention == "arkit":
        return T_wc @ _ARKIT_TO_OPENCV
    raise UnsupportedEncoding(f"Unsupported pose convention: {convention!r}")


# ───────────────────────────── ROS image / depth encodings (bag sources, P3 task 1) ─────────────────────────────
# The channel-order rule (execution plan, P3): FramePacket.rgb is BGR; a reader converts keyed on the message
# `encoding` field, never on pixel statistics. Depth stays in WIRE terms (uint16_mm / float32_m) so the bag path
# decodes through the same `decode_depth` as the receivers.

_ROS_COLOR_CHANNELS = {"rgb8": 3, "bgr8": 3, "8uc3": 3, "rgba8": 4, "bgra8": 4, "8uc4": 4, "mono8": 1, "8uc1": 1}
_ROS_DEPTH_WIRE = {"16uc1": ("uint16_mm", 2, 0.001), "mono16": ("uint16_mm", 2, 0.001), "32fc1": ("float32_m", 4, 1.0)}


def ros_image_to_bgr(encoding: str, data: Any, width: int, height: int, *, step: Optional[int] = None,
                     is_bigendian: int = 0) -> np.ndarray:
    """A raw ``sensor_msgs/Image`` colour payload -> (H, W, 3) uint8 BGR.
    rgb8 / rgba8 are swapped, bgr8 / bgra8 / 8UC3 taken as they are, mono8 /
    8UC1 replicated. Row padding (``step`` > width * channels) is dropped.
    Raises UnsupportedEncoding for other encodings and for big-endian images."""
    if int(is_bigendian or 0):
        raise UnsupportedEncoding("big-endian images are not supported")
    enc = str(encoding).lower()
    ch = _ROS_COLOR_CHANNELS.get(enc)
    if ch is None:
        raise UnsupportedEncoding(f"Unsupported ROS image encoding: {encoding!r}")
    w, h = int(width), int(height)
    row = int(step) if step else w * ch
    buf = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=np.uint8).reshape(-1)
    if row < w * ch or buf.size < row * h:
        raise ValueError(f"image buffer too small: {buf.size} bytes for {w}x{h}x{ch} (step {row})")
    img = buf[: row * h].reshape(h, row)[:, : w * ch].reshape(h, w, ch)
    if enc == "rgb8":
        return np.ascontiguousarray(img[..., ::-1])
    if enc in ("bgr8", "8uc3"):
        return np.ascontiguousarray(img)
    if enc == "rgba8":
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGBA2BGR)
    if enc in ("bgra8", "8uc4"):
        return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(np.ascontiguousarray(img.reshape(h, w)), cv2.COLOR_GRAY2BGR)


def ros_compressed_to_bgr(fmt: str, data: Any) -> np.ndarray:
    """A ``sensor_msgs/CompressedImage`` colour payload -> BGR. ``format`` is
    ``"jpeg"``, ``"png"`` or image_transport's ``"<orig>; <codec> compressed <order>"``
    (e.g. ``"rgb8; jpeg compressed bgr8"``): the codec's own byte order is the
    part after ``compressed`` (default bgr8, what cv2.imdecode returns); ``rgb8``
    there means the compressed bytes carry RGB order and are swapped."""
    f = str(fmt or "").lower()
    if "compresseddepth" in f:
        raise UnsupportedEncoding("compressedDepth is a depth payload (ros_compressed_depth)")
    if not any(c in f for c in ("jpeg", "jpg", "png")):
        raise UnsupportedEncoding(f"Unsupported CompressedImage format: {fmt!r}")
    buf = np.frombuffer(data, dtype=np.uint8) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=np.uint8).reshape(-1)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imdecode failed for CompressedImage {fmt!r} ({buf.size} bytes)")
    order = f.split("compressed", 1)[1].strip() if "compressed" in f else "bgr8"
    if order.startswith("rgb"):
        img = np.ascontiguousarray(img[..., ::-1])
    return img


def ros_depth_wire(encoding: str, data: Any, width: int, height: int, *, is_bigendian: int = 0) -> Tuple[Any, str, float]:
    """A raw ``sensor_msgs/Image`` depth payload -> ``(bytes, wire_encoding, scale)``
    for an ``EncodedImage``: 16UC1 / mono16 -> ``uint16_mm`` (0 = invalid, REP 118),
    32FC1 -> ``float32_m`` (NaN or 0.0 = invalid -- both conventions exist in the
    wild: TUM writes NaN, the RealSense D455 writes 0). Decoding then goes through
    ``decode_depth`` like every receiver frame. Raises UnsupportedEncoding."""
    if int(is_bigendian or 0):
        raise UnsupportedEncoding("big-endian depth images are not supported")
    enc = str(encoding).lower()
    spec = _ROS_DEPTH_WIRE.get(enc)
    if spec is None:
        raise UnsupportedEncoding(f"Unsupported ROS depth encoding: {encoding!r} (16UC1 | mono16 | 32FC1)")
    wire, bpp, scale = spec
    raw = bytes(data) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=np.uint8).tobytes()
    need = int(width) * int(height) * bpp
    if len(raw) < need:
        raise ValueError(f"depth buffer too small: {len(raw)} bytes for {width}x{height}x{bpp}")
    return raw[:need], wire, scale


def ros_compressed_depth(fmt: str, data: Any) -> Tuple[Any, str, float]:
    """``sensor_msgs/CompressedImage`` from compressed_depth_image_transport
    (format ``"16UC1; compressedDepth png"`` / ``"32FC1; compressedDepth"``): a
    12-byte header (int32 format, float32 depthQuantA, float32 depthQuantB) then a
    16-bit PNG. 16UC1 -> uint16 millimetres verbatim; 32FC1 -> the quantised
    inverse depth is turned back into metres (``A / (q - B)``, 0 = invalid) and
    returned as ``float32_m``. ``rvl`` -> UnsupportedEncoding (explicit, as planned)."""
    f = str(fmt or "").lower()
    if "rvl" in f:
        raise UnsupportedEncoding("compressedDepth rvl is not supported (png only)")
    raw = bytes(data) if not isinstance(data, np.ndarray) else np.asarray(data, dtype=np.uint8).tobytes()
    if len(raw) < 12:
        raise ValueError("compressedDepth payload shorter than its 12-byte header")
    _fmt_enum, quant_a, quant_b = struct.unpack("<iff", raw[:12])
    png = np.frombuffer(raw[12:], dtype=np.uint8)
    arr = cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError("compressedDepth PNG failed to decode")
    if arr.dtype != np.uint16:
        raise UnsupportedEncoding(f"compressedDepth PNG dtype {arr.dtype} unexpected (uint16)")
    if f.startswith("32fc1"):
        q = arr.astype(np.float32)
        depth = np.zeros_like(q, dtype=np.float32)
        valid = arr > 0
        depth[valid] = np.float32(quant_a) / (q[valid] - np.float32(quant_b))
        return depth.tobytes(), "float32_m", 1.0
    return np.ascontiguousarray(arr).tobytes(), "uint16_mm", 0.001
