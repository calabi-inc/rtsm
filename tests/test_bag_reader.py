"""
P3 task 1 -- the bag reader on synthetic bags (written in-test with rosbags'
Writer) and, when present, on the external corpus (recordings/external/).

Covers: discovery rules on the corpus' topic sets, RGB/depth pairing with
offsets and gaps, every colour and depth encoding, a moving + static TF chain,
an odometry pose source, relative topic names and leading-slash frames, zero
stamps, a bag without message definitions, a bare MCAP, and every refusal.
"""
from __future__ import annotations

import shutil
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")

from rosbags.rosbag2 import CompressionFormat, CompressionMode, StoragePlugin, Writer
from rosbags.typesys import Stores, get_typestore

from rtsm.io import codecs
from rtsm.io.bag_reader import (
    BagInfo, BagRefusal, BagStats, TopicInfo, iter_bag_frames, open_bag, probe_bag, resolve_topics,
)

REPO = Path(__file__).resolve().parents[1]
EXTERNAL = REPO / "recordings" / "external"
S = 1_000_000_000
TS = get_typestore(Stores.ROS2_HUMBLE)
T = TS.types


def _hdr(stamp_ns: int, frame: str):
    return T["std_msgs/msg/Header"](stamp=T["builtin_interfaces/msg/Time"](sec=stamp_ns // S, nanosec=stamp_ns % S), frame_id=frame)


def _image(stamp_ns, frame, arr: np.ndarray, encoding: str, step=None):
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    bpp = arr.dtype.itemsize
    return T["sensor_msgs/msg/Image"](header=_hdr(stamp_ns, frame), height=h, width=w, encoding=encoding, is_bigendian=0,
                                      step=step or w * ch * bpp, data=np.frombuffer(np.ascontiguousarray(arr).tobytes(), dtype=np.uint8))


def _compressed(stamp_ns, frame, fmt: str, payload: bytes):
    return T["sensor_msgs/msg/CompressedImage"](header=_hdr(stamp_ns, frame), format=fmt, data=np.frombuffer(payload, dtype=np.uint8))


def _info(stamp_ns, frame, w, h, fx, fy, cx, cy):
    ROI = T["sensor_msgs/msg/RegionOfInterest"]
    return T["sensor_msgs/msg/CameraInfo"](header=_hdr(stamp_ns, frame), height=h, width=w, distortion_model="plumb_bob",
                                           d=np.zeros(5), k=np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1.0]), r=np.eye(3).reshape(-1),
                                           p=np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1.0, 0]), binning_x=0, binning_y=0,
                                           roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))


def _tf(stamp_ns, parent, child, t, q):
    TSt, Tf, V3, Q = T["geometry_msgs/msg/TransformStamped"], T["geometry_msgs/msg/Transform"], T["geometry_msgs/msg/Vector3"], T["geometry_msgs/msg/Quaternion"]
    return T["tf2_msgs/msg/TFMessage"](transforms=[TSt(header=_hdr(stamp_ns, parent), child_frame_id=child,
                                                     transform=Tf(translation=V3(x=t[0], y=t[1], z=t[2]), rotation=Q(x=q[0], y=q[1], z=q[2], w=q[3])))])


def _odom(stamp_ns, parent, child, t, q):
    P, Pt, Q, PwC, TwC, Tw = (T["geometry_msgs/msg/Pose"], T["geometry_msgs/msg/Point"], T["geometry_msgs/msg/Quaternion"],
                              T["geometry_msgs/msg/PoseWithCovariance"], T["geometry_msgs/msg/TwistWithCovariance"], T["geometry_msgs/msg/Twist"])
    V3 = T["geometry_msgs/msg/Vector3"]
    return T["nav_msgs/msg/Odometry"](header=_hdr(stamp_ns, parent), child_frame_id=child,
                                     pose=PwC(pose=P(position=Pt(x=t[0], y=t[1], z=t[2]), orientation=Q(x=q[0], y=q[1], z=q[2], w=q[3])), covariance=np.zeros(36)),
                                     twist=TwC(twist=Tw(linear=V3(x=0, y=0, z=0), angular=V3(x=0, y=0, z=0)), covariance=np.zeros(36)))


IDENT = (0.0, 0.0, 0.0, 1.0)


def q_z(deg):
    a = np.deg2rad(deg) / 2
    return (0.0, 0.0, float(np.sin(a)), float(np.cos(a)))


def bgr_frame(i: int, w=8, h=6) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200 - i                    # B
    img[..., 1] = np.arange(w, dtype=np.uint8)[None, :] * 20
    img[..., 2] = 10 + i                     # R  (B != R)
    return img


def depth_mm(i: int, w=8, h=6) -> np.ndarray:
    d = (np.arange(h * w, dtype=np.uint16).reshape(h, w) * 37 + 500 + i).astype(np.uint16)
    d[0, :2] = 0
    return d


def write_bag(path: Path, messages, *, storage=StoragePlugin.MCAP, chunk_zstd=False):
    """messages: iterable of (topic, msgtype, log_ns, msg), written in the given order."""
    conns = {}
    w = Writer(path, version=9, storage_plugin=storage)
    if chunk_zstd:
        w.set_compression(CompressionMode.STORAGE, CompressionFormat.ZSTD)
    with w:
        for topic, msgtype, log_ns, msg in messages:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, msgtype, typestore=TS)
            w.write(conns[topic], int(log_ns), TS.serialize_cdr(msg, msgtype))
    return path


def simple_bag_messages(n=6, *, rgb_encoding="rgb8", depth_encoding="16UC1", topic_prefix="/camera", camera="cam_optical",
                        world="map", depth_offset_ns=1_000_000, tf_every=1, gaps=(), moving=True, relative=False,
                        odom=False, seq_topic=False, tracking=None, info_size=None, static_hop=True):
    """A small RGB-D + pose bag. Poses: world -> base moving along +x (0.1 m per frame) with a yaw ramp; base -> camera
    static +0.05 m. Depth stamps are offset by ``depth_offset_ns``; ``gaps`` lists RGB indices with no depth."""
    pre = topic_prefix if not relative else topic_prefix.strip("/")
    rgb_t, depth_t, info_t = f"{pre}/color/image_raw", f"{pre}/depth/image_rect_raw", f"{pre}/color/camera_info"
    msgs = []
    for i in range(n):
        st = S + i * 100_000_000
        log = 1_700_000_000 * S + i * 100_000_000
        bgr = bgr_frame(i)
        if rgb_encoding == "rgb8":
            msgs.append((rgb_t, "sensor_msgs/msg/Image", log, _image(st, camera, np.ascontiguousarray(bgr[..., ::-1]), "rgb8")))
        elif rgb_encoding == "bgr8":
            msgs.append((rgb_t, "sensor_msgs/msg/Image", log, _image(st, camera, bgr, "bgr8")))
        elif rgb_encoding == "rgba8":
            msgs.append((rgb_t, "sensor_msgs/msg/Image", log, _image(st, camera, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA), "rgba8")))
        elif rgb_encoding == "mono8":
            msgs.append((rgb_t, "sensor_msgs/msg/Image", log, _image(st, camera, bgr[..., 1].copy(), "mono8")))
        elif rgb_encoding == "png_compressed":
            ok, png = cv2.imencode(".png", bgr)
            msgs.append((rgb_t, "sensor_msgs/msg/CompressedImage", log, _compressed(st, camera, "bgr8; png compressed bgr8", png.tobytes())))
        else:
            raise ValueError(rgb_encoding)
        d = depth_mm(i)
        dst = st + depth_offset_ns
        if i not in gaps:
            if depth_encoding == "16UC1":
                msgs.append((depth_t, "sensor_msgs/msg/Image", log + 1, _image(dst, camera, d, "16UC1")))
            elif depth_encoding == "32FC1":
                f = d.astype(np.float32) * 0.001
                f[d == 0] = np.nan
                msgs.append((depth_t, "sensor_msgs/msg/Image", log + 1, _image(dst, camera, f, "32FC1")))
            elif depth_encoding == "32FC1_zero":
                f = d.astype(np.float32) * 0.001
                msgs.append((depth_t, "sensor_msgs/msg/Image", log + 1, _image(dst, camera, f, "32FC1")))
            elif depth_encoding == "compressedDepth":
                ok, png = cv2.imencode(".png", d)
                msgs.append((depth_t, "sensor_msgs/msg/CompressedImage", log + 1,
                             _compressed(dst, camera, "16UC1; compressedDepth png", struct.pack("<iff", 0, 0.0, 0.0) + png.tobytes())))
            elif depth_encoding == "rvl":
                msgs.append((depth_t, "sensor_msgs/msg/CompressedImage", log + 1,
                             _compressed(dst, camera, "16UC1; compressedDepth rvl", struct.pack("<iff", 0, 0.0, 0.0) + b"\x00" * 16)))
            else:
                raise ValueError(depth_encoding)
        iw, ih = info_size or (8, 6)
        msgs.append((info_t, "sensor_msgs/msg/CameraInfo", log + 2, _info(st, camera, iw, ih, 5.0 * iw / 8, 5.0 * ih / 6, 4.0 * iw / 8, 3.0 * ih / 6)))
        if seq_topic:
            msgs.append((f"{pre}/frame_seq", "std_msgs/msg/UInt32", log, T["std_msgs/msg/UInt32"](data=100 + i)))
        if tracking is not None:
            msgs.append((f"{pre}/tracking_state", "std_msgs/msg/String", log, T["std_msgs/msg/String"](data=tracking[i % len(tracking)])))
    # poses: TF samples at every frame stamp (plus one before and one after so interpolation brackets everything)
    stamps = [S - 100_000_000] + [S + i * 100_000_000 for i in range(n)] + [S + n * 100_000_000]
    for k, st in enumerate(stamps):
        if not moving:
            break
        x = 0.1 * (k - 1)
        pose_t, pose_q = (x, 0.0, 1.0), q_z(5.0 * (k - 1))
        if odom:
            msgs.append(("/odom", "nav_msgs/msg/Odometry", 1_700_000_000 * S + (k - 1) * 100_000_000 - 5, _odom(st, world, "base", pose_t, pose_q)))
        elif k % tf_every == 0:
            msgs.append(("/tf", "tf2_msgs/msg/TFMessage", 1_700_000_000 * S + (k - 1) * 100_000_000 - 5, _tf(st, world, "base", pose_t, pose_q)))
    if static_hop:
        msgs.append(("/tf_static", "tf2_msgs/msg/TFMessage", 1_700_000_000 * S - 10, _tf(0, "base", camera, (0.05, 0.0, 0.0), IDENT)))
    msgs.sort(key=lambda m: m[2])
    return msgs


def expected_pose(i):
    """world -> base (0.1 i, 0, 1, yaw 5 i deg) composed with base -> cam (+0.05 x)."""
    from rtsm.io.tf_buffer import make_T, split_T
    Tw = make_T((0.1 * i, 0.0, 1.0), q_z(5.0 * i)) @ make_T((0.05, 0.0, 0.0), IDENT)
    return split_T(Tw)


def _frames(path, **kw):
    st = BagStats()
    frames = list(iter_bag_frames(path, stats=st, **kw))
    return frames, st


def _decoded(fr):
    h = fr.header
    bgr = codecs.decode_rgb(h.rgb.data, h.rgb.encoding, h.rgb.width, h.rgb.height)
    depth = codecs.decode_depth(h.depth.data, h.depth.encoding, h.depth.width, h.depth.height, h.depth.scale)
    return bgr, depth


# ───────────────────────────── discovery on the corpus' topic sets ─────────────────────────────

def _info_from(topics: dict, kind="rosbag2") -> BagInfo:
    return BagInfo(path="x", kind=kind, storage="mcap", has_typedefs=True, message_count=0, duration_s=0.0, typestore="humble",
                   topics={"/" + n.lstrip("/"): TopicInfo(name="/" + n.lstrip("/"), msgtype=t, count=1, raw_name=n) for n, t in topics.items()})


def test_discovery_layout_v1_tum_and_r2b():
    v1 = _info_from({"/camera/color/image_raw": "sensor_msgs/msg/Image", "/camera/depth/image_rect_raw": "sensor_msgs/msg/Image",
                     "/camera/confidence/image_raw": "sensor_msgs/msg/Image", "/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
                     "/tf": "tf2_msgs/msg/TFMessage", "/arkit/frame_seq": "std_msgs/msg/UInt32", "/arkit/tracking_state": "std_msgs/msg/String"})
    tm = resolve_topics(v1)
    assert (tm.rgb, tm.depth, tm.confidence, tm.rgb_info, tm.depth_info, tm.tf, tm.seq, tm.tracking) == (
        "/camera/color/image_raw", "/camera/depth/image_rect_raw", "/camera/confidence/image_raw", "/camera/color/camera_info", None, "/tf",
        "/arkit/frame_seq", "/arkit/tracking_state")
    tum = _info_from({"/camera/rgb/image_color": "sensor_msgs/msg/Image", "/camera/depth/image": "sensor_msgs/msg/Image",
                      "/camera/rgb/camera_info": "sensor_msgs/msg/CameraInfo", "/camera/depth/camera_info": "sensor_msgs/msg/CameraInfo",
                      "/tf": "tf/msg/tfMessage", "/imu": "sensor_msgs/msg/Imu", "/cortex_marker_array": "visualization_msgs/msg/MarkerArray"}, kind="ros1")
    tm = resolve_topics(tum)
    assert (tm.rgb, tm.depth, tm.rgb_info, tm.depth_info, tm.tf) == (
        "/camera/rgb/image_color", "/camera/depth/image", "/camera/rgb/camera_info", "/camera/depth/camera_info", "/tf")
    r2b = _info_from({"d455_1_rgb_image": "sensor_msgs/msg/Image", "d455_1_depth_image": "sensor_msgs/msg/Image",
                      "d455_1_left_ir_image": "sensor_msgs/msg/Image", "d455_1_right_ir_image": "sensor_msgs/msg/Image",
                      "hawk_0_left_rgb_image": "sensor_msgs/msg/Image", "hawk_0_right_rgb_image": "sensor_msgs/msg/Image",
                      "d455_1_rgb_camera_info": "sensor_msgs/msg/CameraInfo", "d455_1_depth_camera_info": "sensor_msgs/msg/CameraInfo",
                      "d455_1_left_ir_camera_info": "sensor_msgs/msg/CameraInfo", "hawk_0_left_rgb_camera_info": "sensor_msgs/msg/CameraInfo",
                      "/tf_static": "tf2_msgs/msg/TFMessage", "pandar_xt_32_0_lidar": "sensor_msgs/msg/PointCloud2"})
    tm = resolve_topics(r2b)
    assert (tm.rgb, tm.depth, tm.rgb_info, tm.depth_info, tm.tf, tm.tf_static) == (
        "/d455_1_rgb_image", "/d455_1_depth_image", "/d455_1_rgb_camera_info", "/d455_1_depth_camera_info", None, "/tf_static")
    # overrides win and are validated
    tm = resolve_topics(r2b, {"rgb": "hawk_0_left_rgb_image"})
    assert tm.rgb == "/hawk_0_left_rgb_image" and tm.rules["rgb"] == "override"
    with pytest.raises(BagRefusal) as e:
        resolve_topics(r2b, {"depth": "/nope"})
    assert e.value.codes == ["no_depth_topic"]
    with pytest.raises(ValueError, match="unknown roles"):
        resolve_topics(r2b, {"lidar": "x"})


# ───────────────────────────── synthetic bags ─────────────────────────────

def test_round_trip_rgb8_16uc1_with_tf_chain(tmp_path):
    bag = write_bag(tmp_path / "b", simple_bag_messages(6, seq_topic=True, tracking=["normal", "normal", "limited"]))
    info = open_bag(bag)
    assert info.kind == "rosbag2" and info.storage == "mcap" and info.has_typedefs
    frames, st = _frames(bag)
    assert [f.header.seq for f in frames] == [100, 101, 102, 103, 104, 105]
    assert st.paired == 6 and st.unpaired_rgb == 0 and st.unpaired_depth == 0 and st.pose_missing == 0
    assert st.pose_kind == "tf" and st.tf_chain == ["map -> base", "base -> cam_optical"] and st.pair_dt_ms_max == pytest.approx(1.0)
    assert [f.header.tracking_state for f in frames] == ["normal", "normal", "limited", "normal", "normal", "limited"]
    for i, fr in enumerate(frames):
        h = fr.header
        assert h.t_sensor_ns == S + i * 100_000_000 and h.pose_convention == "opencv" and h.pose_frame_id == "map"
        bgr, depth = _decoded(fr)
        assert np.array_equal(bgr, bgr_frame(i))
        ref = depth_mm(i).astype(np.float32) * 0.001
        ref[depth_mm(i) == 0] = np.nan
        assert np.array_equal(np.isnan(depth), np.isnan(ref)) and np.allclose(np.nan_to_num(depth), np.nan_to_num(ref))
        t, q = expected_pose(i)
        assert np.allclose(h.pose_raw[0], t, atol=1e-6) and np.allclose(np.abs(h.pose_raw[1]), np.abs(q), atol=1e-6)
        assert (h.intrinsics.width, h.intrinsics.height, h.intrinsics.fx, h.intrinsics.cx) == (8, 6, 5.0, 4.0)
        assert h.t_wall_utc_s == pytest.approx(1_700_000_000 + i * 0.1, abs=1e-6)


@pytest.mark.parametrize("rgb_enc", ["bgr8", "rgba8", "mono8", "png_compressed"])
def test_colour_encodings_all_land_in_bgr(tmp_path, rgb_enc):
    bag = write_bag(tmp_path / rgb_enc, simple_bag_messages(2, rgb_encoding=rgb_enc))
    frames, st = _frames(bag)
    assert len(frames) == 2
    bgr, _ = _decoded(frames[0])
    if rgb_enc == "mono8":
        assert bgr.shape == (6, 8, 3) and np.array_equal(bgr[..., 0], bgr_frame(0)[..., 1])
    else:
        assert np.array_equal(bgr, bgr_frame(0))
    if rgb_enc == "png_compressed":
        assert frames[0].header.rgb.encoding.startswith("rosc:") and frames[0].header.intrinsics.width == 8   # size from CameraInfo


@pytest.mark.parametrize("depth_enc", ["32FC1", "32FC1_zero", "compressedDepth"])
def test_depth_encodings(tmp_path, depth_enc):
    bag = write_bag(tmp_path / depth_enc, simple_bag_messages(2, depth_encoding=depth_enc))
    frames, st = _frames(bag)
    assert len(frames) == 2
    _, depth = _decoded(frames[1])
    ref = depth_mm(1).astype(np.float32) * 0.001
    ref[depth_mm(1) == 0] = np.nan
    assert np.array_equal(np.isnan(depth), np.isnan(ref)) and np.allclose(np.nan_to_num(depth), np.nan_to_num(ref), atol=1e-6)


def test_rvl_depth_is_refused(tmp_path):
    bag = write_bag(tmp_path / "rvl", simple_bag_messages(2, depth_encoding="rvl"))
    with pytest.raises(BagRefusal) as e:
        _frames(bag)
    assert e.value.codes == ["unsupported_depth_encoding"] and "rvl" in str(e.value)


def test_pairing_offsets_gaps_and_tolerance(tmp_path):
    msgs = simple_bag_messages(8, depth_offset_ns=15_000_000, gaps=(2, 5))
    bag = write_bag(tmp_path / "pair", msgs)
    frames, st = _frames(bag)
    assert [f.header.seq for f in frames] == [0, 1, 3, 4, 6, 7]                # index-based seq: gaps keep their numbers
    assert st.frames_seen == 8 and st.paired == 6 and st.unpaired_rgb == 2 and st.unpaired_depth == 0
    assert st.pair_dt_ms_max == pytest.approx(15.0)
    frames, st = _frames(bag, pair_tolerance_s=0.010)                           # 15 ms offsets no longer pair
    assert frames == [] and st.unpaired_rgb == 8 and st.unpaired_depth == 6


def test_relative_topic_names_and_odometry_pose(tmp_path):
    bag = write_bag(tmp_path / "odom", simple_bag_messages(4, relative=True, odom=True))
    info = open_bag(bag)
    assert "/camera/color/image_raw" in info.topics and info.topics["/camera/color/image_raw"].raw_name == "camera/color/image_raw"
    frames, st = _frames(bag)
    assert len(frames) == 4 and st.pose_kind == "odometry" and st.tf_chain == ["map -> base", "base -> cam_optical"]
    t, q = expected_pose(2)
    assert np.allclose(frames[2].header.pose_raw[0], t, atol=1e-6)


def test_tf_interpolation_between_sparse_samples(tmp_path):
    bag = write_bag(tmp_path / "sparse", simple_bag_messages(6, tf_every=2))
    frames, st = _frames(bag)
    assert len(frames) == 6 and st.pose_missing == 0
    for i in (1, 3):                                                             # stamps between TF samples -> interpolated
        t, q = expected_pose(i)
        assert np.allclose(frames[i].header.pose_raw[0], t, atol=1e-6)
        assert np.allclose(np.abs(frames[i].header.pose_raw[1]), np.abs(q), atol=1e-6)


def test_world_and_camera_frame_overrides(tmp_path):
    bag = write_bag(tmp_path / "frames", simple_bag_messages(3))
    frames, st = _frames(bag, world_frame="map", camera_frame="cam_optical")
    assert st.world_frame == "map" and st.tf_chain == ["map -> base", "base -> cam_optical"] and len(frames) == 3
    # a world frame below the moving hop leaves only static hops: a constant pose is refused, not silently accepted
    with pytest.raises(BagRefusal) as e:
        _frames(bag, world_frame="base")
    assert e.value.codes == ["no_pose_source"] and "only static hops" in str(e.value)
    with pytest.raises(BagRefusal) as e:
        _frames(bag, world_frame="mars")
    assert e.value.codes == ["no_pose_source"]


def test_refusals_no_depth_no_pose_unaligned(tmp_path):
    msgs = [m for m in simple_bag_messages(3) if "depth" not in m[0]]
    with pytest.raises(BagRefusal) as e:
        _frames(write_bag(tmp_path / "nodepth", msgs))
    assert e.value.codes == ["no_depth_topic"]
    st = probe_bag(tmp_path / "nodepth")
    assert st.refusal and st.refusal[0][0] == "no_depth_topic"
    # static hops only: no moving pose
    msgs = simple_bag_messages(3, moving=False)
    with pytest.raises(BagRefusal) as e:
        _frames(write_bag(tmp_path / "static", msgs))
    assert e.value.codes == ["no_pose_source"] and "only static hops" in str(e.value)
    # unaligned: a depth CameraInfo whose K differs by 5 %
    msgs = simple_bag_messages(3)
    msgs.append(("/camera/depth/camera_info", "sensor_msgs/msg/CameraInfo", 1_700_000_000 * S + 3, _info(S, "depth_optical", 8, 6, 5.25, 5.25, 4.2, 3.15)))
    bag = write_bag(tmp_path / "unaligned", sorted(msgs, key=lambda m: m[2]))
    with pytest.raises(BagRefusal) as e:
        _frames(bag)
    assert e.value.codes == ["unaligned_depth"]
    frames, st = _frames(bag, assume_aligned=True)
    assert len(frames) == 3 and "assumed aligned" in st.registration
    # both problems reported together
    msgs = [m for m in msgs if m[0] not in ("/tf",)]
    with pytest.raises(BagRefusal) as e:
        _frames(write_bag(tmp_path / "both", msgs))
    assert set(e.value.codes) == {"no_pose_source", "unaligned_depth"}


def test_zero_stamps_are_skipped_and_counted(tmp_path):
    msgs = simple_bag_messages(3)
    msgs.append(("/camera/color/image_raw", "sensor_msgs/msg/Image", 1_700_000_000 * S + 999, _image(0, "cam_optical", bgr_frame(9)[..., ::-1].copy(), "rgb8")))
    frames, st = _frames(write_bag(tmp_path / "zero", sorted(msgs, key=lambda m: m[2])))
    assert len(frames) == 3 and st.skipped_zero_stamp == 1


def test_camera_info_at_other_resolution_is_rescaled(tmp_path):
    bag = write_bag(tmp_path / "rescale", simple_bag_messages(2, info_size=(16, 12)))
    frames, _ = _frames(bag)
    i = frames[0].header.intrinsics
    assert (i.width, i.height) == (8, 6) and i.fx == pytest.approx(5.0) and i.cx == pytest.approx(4.0)


def test_bare_mcap_and_sqlite3_storage(tmp_path):
    bag = write_bag(tmp_path / "chunked", simple_bag_messages(3), chunk_zstd=True)
    mcap_file = next(bag.glob("*.mcap"))
    bare = tmp_path / "bare.mcap"
    shutil.copy(mcap_file, bare)
    info = open_bag(bare)
    assert info.kind == "mcap" and info.storage == "mcap" and info.message_count > 0
    frames, st = _frames(bare)
    assert len(frames) == 3 and st.bag["kind"] == "mcap"
    assert np.array_equal(_decoded(frames[1])[0], bgr_frame(1))
    sq = write_bag(tmp_path / "sq", simple_bag_messages(3), storage=StoragePlugin.SQLITE3)
    assert open_bag(sq).storage == "sqlite3"
    frames, _ = _frames(sq)
    assert len(frames) == 3


def test_ros2idl_mcap_is_refused(tmp_path):
    from mcap.writer import Writer as McapWriter
    p = tmp_path / "idl.mcap"
    with open(p, "wb") as fh:
        w = McapWriter(fh)
        w.start(profile="ros2")
        sid = w.register_schema(name="sensor_msgs/msg/Image", encoding="ros2idl", data=b"module sensor_msgs {}")
        cid = w.register_channel(topic="/camera/color/image_raw", message_encoding="cdr", schema_id=sid)
        w.add_message(channel_id=cid, log_time=1, publish_time=1, data=b"\x00\x01\x00\x00")
        w.finish()
    with pytest.raises(BagRefusal) as e:
        open_bag(p)
    assert e.value.codes == ["unsupported_schema_encoding"] and "ros2idl" in str(e.value)
    assert probe_bag(p).refusal[0][0] == "unsupported_schema_encoding"


def test_not_a_bag_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_bag(tmp_path / "missing")
    (tmp_path / "x.db3").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="DIRECTORY"):
        open_bag(tmp_path / "x.db3")


# ───────────────────────────── the corpus (skipped when absent) ─────────────────────────────

@pytest.mark.skipif(not (EXTERNAL / "rgbd_dataset_freiburg1_desk.bag").is_file(), reason="TUM fr1/desk bag not on this box")
def test_tum_fr1_desk_reads_through_discovery_alone():
    st = BagStats()
    frames = list(iter_bag_frames(EXTERNAL / "rgbd_dataset_freiburg1_desk.bag", stats=st, max_frames=60))
    assert len(frames) == 60 and st.pose_missing == 0
    assert st.tf_chain == ["world -> kinect", "kinect -> openni_camera", "openni_camera -> openni_rgb_frame", "openni_rgb_frame -> openni_rgb_optical_frame"]
    assert st.topics["rgb"] == "/camera/rgb/image_color" and st.topics["depth"] == "/camera/depth/image" and st.topics["rgb_info"] == "/camera/rgb/camera_info"
    assert "matches rgb K" in st.registration and st.bag["kind"] == "ros1"
    h = frames[0].header
    assert h.rgb.encoding == "ros:rgb8" and h.depth.encoding == "float32_m" and h.tracking_state == "normal" and h.confidence is None
    assert h.intrinsics.fx == pytest.approx(525.0) and (h.intrinsics.width, h.intrinsics.height) == (640, 480)
    bgr, depth = _decoded(frames[0])
    assert bgr.shape == (480, 640, 3) and 0.1 < float(np.isnan(depth).mean()) < 0.3
    assert 0.5 < float(np.linalg.norm(h.pose_raw[0])) < 5.0                     # a mocap pose a metre or two from the origin


@pytest.mark.skipif(not (EXTERNAL / "r2b_cafe" / "metadata.yaml").is_file(), reason="r2b_cafe not on this box")
def test_r2b_cafe_is_refused_with_both_reasons():
    st = probe_bag(EXTERNAL / "r2b_cafe")
    assert st.refusal and {c for c, _ in st.refusal} == {"no_pose_source", "unaligned_depth"}
    assert st.topics["rgb"] == "/d455_1_rgb_image" and st.topics["depth"] == "/d455_1_depth_image" and st.bag["storage"] == "sqlite3"
    st = probe_bag(EXTERNAL / "r2b_cafe", assume_aligned=True)
    assert [c for c, _ in st.refusal] == ["no_pose_source"]


@pytest.mark.skipif(not (EXTERNAL / "r2b_hope" / "metadata.yaml").is_file(), reason="r2b_hope not on this box")
def test_r2b_hope_is_refused():
    st = probe_bag(EXTERNAL / "r2b_hope")
    assert st.refusal and {c for c, _ in st.refusal} == {"no_depth_topic", "no_camera_info"}
