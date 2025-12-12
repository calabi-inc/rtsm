from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from typing import Literal, List, Dict, Any
from dataclasses import field


BBox = Tuple[int, int, int, int]
StructType = Literal["floor", "wall", "ceiling"]

Vec3 = NDArray[np.float32]
Quat = NDArray[np.float32]       # xyzw (unit)
Mat3 = NDArray[np.float32]
Mat4 = NDArray[np.float32]
RGB  = NDArray[np.uint8]         # (H,W,3) uint8, BGR or RGB (document in cfg)
Depth = NDArray[np.float32]      # (H,W) float32 meters, NaN = invalid

__all__ = ["ClipTopK", "Observation", "ObjectState", "BBox", "StructType"]

# ------------------------- CLIP top-k entry -------------------------

@dataclass(slots=True)
class ClipTopK:
    class_id: str
    score: float
    class_idx: int

# ------------------------- Per-keyframe observation -------------------------

@dataclass(slots=True)
class Observation:
    """
    Ephemeral, per-keyframe detection that flows through association → memory.
    """
    temp_id: str                          # e.g., "kf123_m7"
    bbox_xyxy: BBox
    area_px: int
    is_partial: bool
    centroid_px: Tuple[float, float]
    centroid_cam: Optional[NDArray[np.float32]] = None   # (3,) in camera frame
    label: str = "unknown"
    score: float = 0.0
    clip_vec: Optional[NDArray[np.float32]] = None       # (512,), L2-normalized
    topk: List[ClipTopK] = field(default_factory=list)
    struct_type: Optional[StructType] = None             # floor | wall | ceiling | None
    mean_z_m: Optional[float] = None
    std_z_m: Optional[float] = None
    valid_z_pct: Optional[float] = None

# ------------------------- Persistent object state -------------------------

@dataclass(slots=True)
class ObjectState:
    """
    Persistent, fused state for an object tracked across views.
    """
    id: str
    centroid_w: NDArray[np.float32]                   # (3,) in world frame
    size: Dict[str, Any]                              # {"type":"sphere","r":float} or {"type":"aabb","s":(sx,sy,sz)}
    clip_vec: NDArray[np.float32]                     # (512,), L2-normalized
    last_seen: float                                  # timestamp (s)
    primary_bucket: Tuple[int, int, int]              # spatial index (ix,iy,iz)
    label: str = "unknown"

    
Vec3 = NDArray[np.float32]
Quat = NDArray[np.float32]       # xyzw (unit)
Mat3 = NDArray[np.float32]
Mat4 = NDArray[np.float32]
RGB  = NDArray[np.uint8]         # (H,W,3) uint8, BGR or RGB (document in cfg)
Depth = NDArray[np.float32]      # (H,W) float32 meters, NaN = invalid

# ---------- time bundle ----------

@dataclass(slots=True)
class TimeBundle:
    """All the clocks we care about for one frame."""
    t_mono_s: float                 # time.monotonic(); durations/TTLs
    t_wall_utc_s: float             # time.time(); logs/DB
    t_sensor_ns: Optional[int]      # ROS header.stamp (ns), None if not available
    seq: Optional[int] = None       # ROS seq if available

# ---------- pose ----------

@dataclass(slots=True)
class PoseStamped:
    """Camera pose at t_sensor_ns: T_world_cam = [R|t]."""
    stamp_ns: Optional[int]         # mirror TimeBundle.t_sensor_ns
    frame_id: Optional[str]
    t_wc: Vec3                      # translation (m)
    q_wc_xyzw: Quat                 # unit quaternion (x,y,z,w)

    # cached lazily if you want; here we just compute on demand
    def R_wc(self) -> Mat3:
        x, y, z, w = self.q_wc_xyzw.astype(np.float32)
        # quaternion -> rotation (right-handed, xyzw)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        return np.array([
            [1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],
            [2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],
            [2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)]
        ], dtype=np.float32)

    def T_wc(self) -> Mat4:
        R = self.R_wc()
        T = np.eye(4, dtype=np.float32)
        T[:3,:3] = R
        T[:3, 3] = self.t_wc
        return T

# ---------- camera model (intrinsics) ----------

@dataclass(slots=True)
class PinholeIntrinsics:
    width: int
    height: int
    fx: float; fy: float; cx: float; cy: float

    def K(self) -> Mat3:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float32)

# ---------- canonical frame packet ----------

@dataclass(slots=True)
class FramePacket:
    """
    Canonical, ROS-free payload the CORE consumes.
    - rgb:  HxWx3 uint8
    - depth_m: HxW float32 meters with NaNs for invalids (or None if not provided)
    - pose:  PoseStamped at t_sensor_ns (or None)
    - intr:  camera intrinsics used to interpret rgb/depth
    """
    time: TimeBundle
    rgb: RGB
    depth_m: Optional[Depth]
    pose: Optional[PoseStamped]
    intr: Optional[PinholeIntrinsics]
    is_keyframe: bool = False

    # convenience helpers
    @property
    def size(self) -> Tuple[int,int]:
        h, w = int(self.rgb.shape[0]), int(self.rgb.shape[1])
        return h, w

    def world_from_cam(self, p_cam: Vec3) -> Optional[Vec3]:
        if self.pose is None or p_cam is None: return None
        R = self.pose.R_wc(); t = self.pose.t_wc
        return (R @ p_cam.astype(np.float32) + t).astype(np.float32)

    def cell_of_camera(self, grid_m: float = 0.25) -> Tuple[int,int,int]:
        """Camera origin’s grid cell (for IO staleness)."""
        assert self.pose is not None, "pose required"
        c = self.pose.t_wc
        return (int(np.floor(c[0]/grid_m)),
                int(np.floor(c[1]/grid_m)),
                int(np.floor(c[2]/grid_m)))