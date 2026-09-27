"""
P3 task 0 -- Lens recording -> rosbag2/MCAP converter parity (CPU).

The bag path must reproduce what the websocket receiver produced from
messages.bin: per-frame RGB (BGR contract), depth bytes, confidence map and
intrinsics bit-identical, the pose equal after the receiver's ARKit flip --
the pose-math test CLAUDE.md requires. A tiny synthetic recording ships (built
here, deterministic); the first 20 frames of recordings/session1 are compared
too whenever the recording is on the box.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rosbags", reason="the [eval] extra is not installed")

from rtsm.evaluation import recording_mcap as rm
from rtsm.io import codecs
from rtsm.io.codecs import UnsupportedEncoding
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, IngestFrontEnd
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import parse_lens_message

REPO = Path(__file__).resolve().parents[2]
SESSION1 = REPO / "recordings" / "session1"


# ───────────────────────────── synthetic recording ─────────────────────────────

def _rotmat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64); axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _lens_message(*, frame_id, timestamp_ns, unix_ts, rng, w=32, h=24, dw=8, dh=6, tracking="normal",
                  depth_format="uint16_mm", pose=None, conf=True, session_id="SYN-1"):
    nv12 = rng.integers(0, 256, size=h * w * 3 // 2, dtype=np.uint8).tobytes()
    depth = rng.integers(200, 4000, size=(dh, dw), dtype=np.uint16)
    depth[0, :3] = 0                                                       # invalid pixels
    if depth_format == "float32_m":
        depth_bytes = (depth.astype(np.float32) * 0.001).tobytes()
    else:
        depth_bytes = depth.tobytes()
    if pose is None:
        R = _rotmat([0.3, 1.0, 0.2], 0.4 + 0.1 * frame_id)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0.1 * frame_id, -0.2, 1.5]
        pose = T.flatten(order="F").tolist()
    header = {
        "frame_id": frame_id, "timestamp_ns": timestamp_ns, "unix_timestamp": unix_ts, "tracking_state": tracking,
        "rgb_format": "nv12", "rgb_width": w, "rgb_height": h,
        "depth_format": depth_format, "depth_width": dw, "depth_height": dh, "depth_scale": 0.001,
        # intrinsics declared at 2x the RGB size -> exercises the rescale
        "fx": 40.0 + frame_id, "fy": 41.0, "cx": 32.5, "cy": 24.25, "intrinsics_width": 2 * w, "intrinsics_height": 2 * h,
        "pose_format": "matrix4x4_col_major", "T_wc": pose, "session_id": session_id,
    }
    parts = b""
    if conf:
        c = rng.integers(0, 3, size=(dh, dw), dtype=np.uint8)
        header.update({"confidence_format": "uint8", "confidence_width": dw, "confidence_height": dh})
        cb = c.tobytes(); parts = struct.pack("<I", len(cb)) + cb
    hj = json.dumps(header).encode("utf-8")
    return struct.pack("<I", len(hj)) + hj + struct.pack("<I", len(nv12)) + nv12 + struct.pack("<I", len(depth_bytes)) + depth_bytes + parts


def write_recording(d: Path, messages, session_id="SYN-1") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    off = 0
    with open(d / "messages.bin", "wb") as fb, open(d / "index.jsonl", "w", encoding="utf-8") as fi:
        for i, m in enumerate(messages):
            fb.write(m)
            fi.write(json.dumps({"seq": i, "offset": off, "length": len(m), "t_mono_s": round(0.1 * i, 6)}) + "\n")
            off += len(m)
    (d / "meta.json").write_text(json.dumps({"format_version": 1, "session_id": session_id, "device_name": "synthetic",
                                             "total_binary_messages": len(messages)}), encoding="utf-8")
    return d


@pytest.fixture
def synthetic(tmp_path):
    rng = np.random.default_rng(7)
    msgs = []
    fid = 0
    for i in range(6):
        fid += 1 if i != 3 else 3                                          # a gap in the source frame ids
        msgs.append(_lens_message(frame_id=fid, timestamp_ns=1_000_000_000 + 200_000_000 * i,
                                  unix_ts=1_700_000_000.25 + 0.2 * i, rng=rng,
                                  tracking=("limited" if i == 4 else "normal"), conf=(i != 5)))
    return write_recording(tmp_path / "rec", msgs), msgs


def _assert_frames_equal(a: rm.DecodedFrame, b: rm.DecodedFrame):
    assert a.seq == b.seq and a.t_sensor_ns == b.t_sensor_ns and a.tracking_state == b.tracking_state
    assert abs((a.t_wall_utc_s or 0) - (b.t_wall_utc_s or 0)) < 1e-6
    assert a.rgb_bgr.dtype == b.rgb_bgr.dtype == np.uint8 and np.array_equal(a.rgb_bgr, b.rgb_bgr)
    assert a.depth_raw == b.depth_raw and a.depth_encoding == b.depth_encoding
    assert (a.depth_width, a.depth_height) == (b.depth_width, b.depth_height)
    if a.depth_encoding == "uint16_mm":
        assert a.depth_scale == b.depth_scale                       # 32FC1 is metres: the wire scale is irrelevant there
    assert (a.confidence is None) == (b.confidence is None)
    if a.confidence is not None:
        assert np.array_equal(a.confidence, b.confidence)
    ia, ib = a.intrinsics, b.intrinsics
    assert (ia.width, ia.height, ia.fx, ia.fy, ia.cx, ia.cy) == (ib.width, ib.height, ib.fx, ib.fy, ib.cx, ib.cy)
    assert a.t_wc.dtype == b.t_wc.dtype == np.float32 and np.array_equal(a.t_wc, b.t_wc)
    assert np.array_equal(a.q_wc_xyzw, b.q_wc_xyzw)
    da, db = a.depth_m(), b.depth_m()
    assert np.array_equal(np.isnan(da), np.isnan(db)) and np.array_equal(np.nan_to_num(da), np.nan_to_num(db))


# ───────────────────────────── tests ─────────────────────────────

def test_synthetic_round_trip_is_bit_identical(synthetic, tmp_path):
    rec, msgs = synthetic
    out = tmp_path / "bag"
    s = rm.convert_recording_to_mcap(rec, out, apply_camera_flip=True)
    assert s.frames == 6 and s.skipped == [] and s.arkit_flip_baked and s.compression == "zstd"
    assert s.topics[rm.TOPIC_RGB] == s.topics[rm.TOPIC_DEPTH] == s.topics[rm.TOPIC_INFO] == s.topics[rm.TOPIC_TF] == 6
    assert s.topics[rm.TOPIC_CONF] == 5 and s.topics[rm.TOPIC_SEQ] == 6 and s.topics[rm.TOPIC_TRACKING] == 6
    assert (out / "metadata.yaml").is_file() and list(out.glob("*.mcap"))
    custom = rm.read_bag_custom_data(out)
    assert custom["rtsm_bag_layout"] == "1" and custom["session_id"] == "SYN-1" and custom["arkit_flip_baked"] == "true"
    assert custom["pose_convention"] == "opencv" and custom["frames"] == "6"
    got = list(rm.iter_bag_frames(out))
    want = [rm.decode_recording_frame(m, apply_camera_flip=True, session_id="SYN-1") for m in msgs]
    assert len(got) == 6
    for g, w in zip(got, want):
        _assert_frames_equal(g, w)
    assert [g.seq for g in got] == [1, 2, 3, 6, 7, 8]                  # the gap survives
    assert [g.tracking_state for g in got][4] == "limited"
    assert got[5].confidence is None and got[0].confidence is not None
    assert got[0].intrinsics.fx == 20.5 and got[0].intrinsics.width == 32          # (40 + frame_id 1) / 2: rescaled from the declared 64x48
    assert got[0].session_id == "SYN-1"


def test_channel_order_trap(synthetic, tmp_path):
    """rgb8 on the wire, BGR in the packet: the read-back must equal the receiver's
    BGR array, not its mirror (a converter that forgot the swap would match the mirror)."""
    rec, msgs = synthetic
    out = tmp_path / "bag"
    rm.convert_recording_to_mcap(rec, out)
    g = next(iter(rm.iter_bag_frames(out)))
    w = rm.decode_recording_frame(msgs[0], apply_camera_flip=True)
    assert not np.array_equal(w.rgb_bgr[..., 0], w.rgb_bgr[..., 2])   # the trap is armed (B != R)
    assert np.array_equal(g.rgb_bgr, w.rgb_bgr)
    assert not np.array_equal(g.rgb_bgr, w.rgb_bgr[..., ::-1])


def test_flip_is_baked_once_and_only_when_asked(synthetic, tmp_path):
    rec, msgs = synthetic
    flipped = list(rm.iter_bag_frames(rm.convert_recording_to_mcap(rec, tmp_path / "on", apply_camera_flip=True).out_dir))
    plain = list(rm.iter_bag_frames(rm.convert_recording_to_mcap(rec, tmp_path / "off", apply_camera_flip=False).out_dir))
    raw_t, raw_q = codecs.parse_pose(json.loads(msgs[0][4:4 + struct.unpack("<I", msgs[0][:4])[0]])["T_wc"], "matrix4x4_col_major")
    exp_t, exp_q = codecs.normalize_pose_convention(raw_t, raw_q, "arkit")
    assert np.array_equal(plain[0].t_wc, raw_t.astype(np.float32)) and np.array_equal(plain[0].q_wc_xyzw, raw_q.astype(np.float32))
    assert np.array_equal(flipped[0].t_wc, exp_t.astype(np.float32)) and np.array_equal(flipped[0].q_wc_xyzw, exp_q.astype(np.float32))
    assert not np.array_equal(flipped[0].q_wc_xyzw, plain[0].q_wc_xyzw)  # the flip changed the rotation
    assert rm.read_bag_custom_data(tmp_path / "off")["pose_convention"] == "arkit"


def test_unsupported_depth_encoding_is_an_explicit_error(tmp_path):
    rng = np.random.default_rng(1)
    msgs = [_lens_message(frame_id=0, timestamp_ns=10, unix_ts=1.0, rng=rng)]
    # patch the header's depth_format to something the layout does not carry
    n = struct.unpack("<I", msgs[0][:4])[0]
    hdr = json.loads(msgs[0][4:4 + n]); hdr["depth_format"] = "png_uint16"
    hj = json.dumps(hdr).encode(); msgs[0] = struct.pack("<I", len(hj)) + hj + msgs[0][4 + n:]
    rec = write_recording(tmp_path / "rec", msgs)
    with pytest.raises(UnsupportedEncoding, match="png_uint16"):
        rm.convert_recording_to_mcap(rec, tmp_path / "bag")


def test_float32_depth_and_truncated_message(tmp_path):
    rng = np.random.default_rng(2)
    good = _lens_message(frame_id=0, timestamp_ns=10, unix_ts=1.0, rng=rng, depth_format="float32_m")
    bad = _lens_message(frame_id=1, timestamp_ns=20, unix_ts=1.1, rng=rng)[:100]      # truncated -> skipped, listed
    rec = write_recording(tmp_path / "rec", [good, bad])
    s = rm.convert_recording_to_mcap(rec, tmp_path / "bag", compression=None)
    assert s.frames == 1 and len(s.skipped) == 1 and s.skipped[0][0] == 1 and "LensFramingError" in s.skipped[0][1]
    g = list(rm.iter_bag_frames(tmp_path / "bag"))[0]
    w = rm.decode_recording_frame(good, apply_camera_flip=True)
    assert g.depth_encoding == "float32_m" and g.depth_scale == 1.0
    _assert_frames_equal(g, w)
    with pytest.raises(FileExistsError):
        rm.convert_recording_to_mcap(rec, tmp_path / "bag")
    assert rm.convert_recording_to_mcap(rec, tmp_path / "bag", overwrite=True, compression=None).frames == 1


def test_mcap_library_cross_reads_the_file(synthetic, tmp_path):
    pytest.importorskip("mcap")
    from mcap.reader import make_reader
    rec, _ = synthetic
    out = rm.convert_recording_to_mcap(rec, tmp_path / "bag").out_dir
    with open(next(Path(out).glob("*.mcap")), "rb") as fh:
        r = make_reader(fh)
        assert r.get_header().profile == "ros2"
        summ = r.get_summary()
        assert {s.name for s in summ.schemas.values()} == {"sensor_msgs/msg/Image", "sensor_msgs/msg/CameraInfo",
                                                           "tf2_msgs/msg/TFMessage", "std_msgs/msg/UInt32", "std_msgs/msg/String"}
        assert {c.topic for c in summ.channels.values()} == {rm.TOPIC_RGB, rm.TOPIC_DEPTH, rm.TOPIC_CONF, rm.TOPIC_INFO,
                                                              rm.TOPIC_TF, rm.TOPIC_SEQ, rm.TOPIC_TRACKING}
        assert summ.statistics.message_count == 6 * 7 - 1


def test_cli_wrapper(synthetic, tmp_path, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("recording_to_mcap", REPO / "scripts" / "recording_to_mcap.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rec, _ = synthetic
    assert mod.main([str(rec), str(tmp_path / "cli_bag"), "--max-frames", "2", "--no-compression"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["frames"] == 2 and out["compression"] is None and Path(out["out_dir"]).is_dir()


# ───────────────────────────── the real thing ─────────────────────────────

@pytest.mark.skipif(not (SESSION1 / "messages.bin").is_file(), reason="recordings/session1 not on this box")
def test_session1_first_20_frames_match_the_receiver_packets(tmp_path):
    """The receiver's FramePackets (through the ingest front-end, no throttle,
    no confidence filter) vs the bag's frames: RGB, depth, confidence,
    intrinsics, pose, stamps and seq identical; the confidence filter the
    deployed config applies (threshold 2) gives the same masked depth."""
    n = 20
    s = rm.convert_recording_to_mcap(SESSION1, tmp_path / "s1", apply_camera_flip=True, max_frames=n)
    assert s.frames == n and s.skipped == []
    bag = list(rm.iter_bag_frames(tmp_path / "s1"))
    fe = IngestFrontEnd(source="websocket", policy=WEBSOCKET_POLICY, ingest_queue=IngestQueue(64), throttle_clock="sensor",
                        keyframe_every_n=30, nonkf_min_interval_s=0.0, require_tracking_normal=False, confidence_threshold=0)
    pkts = []
    for i, (_e, data) in enumerate(rm.iter_recording(SESSION1)):
        if i >= n:
            break
        pkt = parse_lens_message(fe, data, apply_camera_flip=True)
        assert pkt is not None
        pkts.append(pkt)
    assert len(bag) == n
    for g, p in zip(bag, pkts):
        assert g.seq == p.time.seq and g.t_sensor_ns == p.time.t_sensor_ns
        assert abs(g.t_wall_utc_s - p.time.t_wall_utc_s) < 1e-6
        assert np.array_equal(g.rgb_bgr, p.rgb)
        assert np.array_equal(g.confidence, p.confidence)
        d = g.depth_m()
        assert np.array_equal(np.isnan(d), np.isnan(p.depth_m)) and np.array_equal(np.nan_to_num(d), np.nan_to_num(p.depth_m))
        assert (g.intrinsics.width, g.intrinsics.height, g.intrinsics.fx, g.intrinsics.fy, g.intrinsics.cx, g.intrinsics.cy) == \
               (p.intr.width, p.intr.height, p.intr.fx, p.intr.fy, p.intr.cx, p.intr.cy)
        assert np.array_equal(g.t_wc, p.pose.t_wc) and np.array_equal(g.q_wc_xyzw, p.pose.q_wc_xyzw)
        # the deployed confidence filter (io.websocket.confidence_threshold: 2) on the bag's depth == on the receiver's
        filt, _ = codecs.apply_confidence_filter(g.depth_m(), g.confidence, 2)
        ref, _ = codecs.apply_confidence_filter(p.depth_m.copy(), p.confidence, 2)
        assert np.array_equal(np.isnan(filt), np.isnan(ref)) and np.array_equal(np.nan_to_num(filt), np.nan_to_num(ref))
    assert bag[0].rgb_bgr.shape == (1440, 1920, 3) and bag[0].confidence.shape == (192, 256)
    assert len({(f.intrinsics.fx, f.intrinsics.cx) for f in bag}) > 1            # ARKit intrinsics vary per frame
