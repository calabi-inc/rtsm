"""
Unit tests for WebSocket receiver data conversion functions and
depth coordinate scaling in mask_staging.
"""

from __future__ import annotations
import json
import struct
import math
import numpy as np
import cv2
import torch
import pytest

from rtsm.io.websocket import decode_rgb, decode_depth, parse_arkit_pose
from rtsm.utils.transforms import rotmat_to_quat_xyzw, euler_to_quat_xyzw, euler_to_rotation_matrix
from rtsm.utils.mask_staging import _depth_coords


# ─────────────────── decode_rgb ───────────────────


class TestDecodeRGB:
    def test_jpeg(self):
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[10:30, 10:50] = [0, 128, 255]
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result = decode_rgb(buf.tobytes(), "jpeg", 64, 48)
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8

    def test_png(self):
        img = np.ones((32, 32, 3), dtype=np.uint8) * 127
        _, buf = cv2.imencode(".png", img)
        result = decode_rgb(buf.tobytes(), "png", 32, 32)
        assert result.shape == (32, 32, 3)
        np.testing.assert_array_equal(result, img)

    def test_bgra(self):
        bgra = np.zeros((16, 16, 4), dtype=np.uint8)
        bgra[:, :, 0] = 10   # B
        bgra[:, :, 1] = 20   # G
        bgra[:, :, 2] = 30   # R
        bgra[:, :, 3] = 255  # A
        result = decode_rgb(bgra.tobytes(), "bgra", 16, 16)
        assert result.shape == (16, 16, 3)
        assert result[0, 0, 0] == 10  # B
        assert result[0, 0, 1] == 20  # G
        assert result[0, 0, 2] == 30  # R

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported rgb_format"):
            decode_rgb(b"\x00", "tiff", 1, 1)

    def test_bgra_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="bgra buffer size mismatch"):
            decode_rgb(b"\x00" * 10, "bgra", 16, 16)


# ─────────────────── decode_depth ───────────────────


class TestDecodeDepth:
    def test_uint16_mm(self):
        depth_u16 = np.array([[1500, 0], [2000, 3000]], dtype=np.uint16)
        raw = depth_u16.tobytes()
        result = decode_depth(raw, "uint16_mm", 2, 2, depth_scale=0.001)
        assert result is not None
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        np.testing.assert_almost_equal(result[0, 0], 1.5)
        assert np.isnan(result[0, 1]), "zero should become NaN"
        np.testing.assert_almost_equal(result[1, 0], 2.0)
        np.testing.assert_almost_equal(result[1, 1], 3.0, decimal=5)

    def test_float32_m(self):
        depth_f32 = np.array([[1.5, 0.0], [2.0, 3.0]], dtype=np.float32)
        raw = depth_f32.tobytes()
        result = decode_depth(raw, "float32_m", 2, 2, depth_scale=1.0)
        assert result is not None
        np.testing.assert_almost_equal(result[0, 0], 1.5)
        assert np.isnan(result[0, 1]), "zero should become NaN"

    def test_png_uint16(self):
        depth_u16 = np.array([[500, 0], [1000, 0]], dtype=np.uint16)
        _, buf = cv2.imencode(".png", depth_u16)
        result = decode_depth(buf.tobytes(), "png_uint16", 2, 2, depth_scale=0.001)
        assert result is not None
        np.testing.assert_almost_equal(result[0, 0], 0.5)
        assert np.isnan(result[0, 1])

    def test_none_format(self):
        assert decode_depth(b"", None, 0, 0) is None

    def test_empty_bytes(self):
        assert decode_depth(b"", "uint16_mm", 2, 2) is None


# ─────────────────── parse_arkit_pose ───────────────────


class TestParseArkitPose:
    def test_quat_translation(self):
        # qx, qy, qz, qw, tx, ty, tz
        data = [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0]
        t, q = parse_arkit_pose(data, "quat_translation")
        np.testing.assert_array_almost_equal(t, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(q, [0.0, 0.0, 0.0, 1.0])

    def test_matrix4x4_identity(self):
        # Identity 4x4 in column-major order
        mat = np.eye(4, dtype=np.float64)
        data = mat.flatten(order="F").tolist()
        t, q = parse_arkit_pose(data, "matrix4x4_col_major")
        np.testing.assert_array_almost_equal(t, [0.0, 0.0, 0.0])
        # Identity quaternion: [0, 0, 0, 1]
        np.testing.assert_array_almost_equal(np.abs(q), [0.0, 0.0, 0.0, 1.0], decimal=5)

    def test_matrix4x4_with_translation(self):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 3] = [5.0, 6.0, 7.0]
        data = mat.flatten(order="F").tolist()
        t, q = parse_arkit_pose(data, "matrix4x4_col_major")
        np.testing.assert_array_almost_equal(t, [5.0, 6.0, 7.0])

    def test_quat_wrong_count_raises(self):
        with pytest.raises(ValueError, match="7 elements"):
            parse_arkit_pose([1, 2, 3], "quat_translation")

    def test_matrix_wrong_count_raises(self):
        with pytest.raises(ValueError, match="16 elements"):
            parse_arkit_pose([1, 2, 3], "matrix4x4_col_major")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported pose_format"):
            parse_arkit_pose([], "euler_xyz")


# ─────────────────── rotmat_to_quat_xyzw ───────────────────


class TestRotmatToQuat:
    def test_identity(self):
        R = np.eye(3, dtype=np.float32)
        q = rotmat_to_quat_xyzw(R)
        np.testing.assert_array_almost_equal(q, [0, 0, 0, 1], decimal=5)

    def test_90_deg_around_z(self):
        # R_z(90°)
        R = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ], dtype=np.float32)
        q = rotmat_to_quat_xyzw(R)
        # Expected: [0, 0, sin(45°), cos(45°)] = [0, 0, 0.7071, 0.7071]
        np.testing.assert_array_almost_equal(
            np.abs(q), [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
            decimal=4,
        )

    def test_roundtrip_with_euler(self):
        """euler → rotation_matrix → rotmat_to_quat should ≈ euler_to_quat."""
        roll, pitch, yaw = 0.3, -0.2, 1.1
        q_direct = euler_to_quat_xyzw(roll, pitch, yaw)
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        q_via_mat = rotmat_to_quat_xyzw(R)
        # Quaternion sign ambiguity: q and -q represent the same rotation
        if np.dot(q_direct, q_via_mat) < 0:
            q_via_mat = -q_via_mat
        np.testing.assert_array_almost_equal(q_direct, q_via_mat, decimal=4)

    def test_unit_norm(self):
        R = euler_to_rotation_matrix(0.5, -0.3, 2.0)
        q = rotmat_to_quat_xyzw(R)
        np.testing.assert_almost_equal(np.linalg.norm(q), 1.0, decimal=5)


# ─────────────────── _depth_coords ───────────────────


class TestDepthCoords:
    def test_same_shape_noop(self):
        ys = np.array([0, 10, 20])
        xs = np.array([5, 15, 25])
        dy, dx = _depth_coords(ys, xs, (480, 640), (480, 640))
        np.testing.assert_array_equal(dy, ys)
        np.testing.assert_array_equal(dx, xs)

    def test_different_shape_scales(self):
        # Mask is 1440x1920, depth is 192x256
        ys = np.array([0, 720, 1439])
        xs = np.array([0, 960, 1919])
        dy, dx = _depth_coords(ys, xs, (192, 256), (1440, 1920))
        # y=0 → 0, y=720 → 96, y=1439 → 191 (clipped)
        assert dy[0] == 0
        assert 95 <= dy[1] <= 96
        assert dy[2] == 191
        # x=0 → 0, x=960 → 128, x=1919 → 255 (clipped)
        assert dx[0] == 0
        assert 127 <= dx[1] <= 128
        assert dx[2] == 255

    def test_depth_quantiles_with_different_resolution(self):
        """End-to-end: mask at 8x8, depth at 4x4 — should not crash."""
        from rtsm.utils.mask_staging import _depth_quantiles

        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[2:6, 2:6] = True  # 4x4 block of True
        depth = np.ones((4, 4), dtype=np.float32) * 2.0  # 2 meters everywhere
        valid_pct, p50, spread = _depth_quantiles(mask, depth)
        assert valid_pct > 0
        assert p50 is not None
        np.testing.assert_almost_equal(p50, 2.0)

    def test_centroid_cam_with_different_resolution(self):
        """End-to-end: mask at 8x8, depth at 4x4."""
        from rtsm.utils.mask_staging import _centroid_cam

        mask = torch.zeros(8, 8, dtype=torch.bool)
        mask[2:6, 2:6] = True
        depth = np.ones((4, 4), dtype=np.float32) * 1.5
        c = _centroid_cam(mask, depth, fx=100, fy=100, cx=4, cy=4, stride=1)
        assert c is not None
        assert c.shape == (3,)
        # z should be ~1.5
        np.testing.assert_almost_equal(c[2], 1.5, decimal=1)


# ─────────────────── Binary message parsing ───────────────────


def _build_test_frame(**overrides) -> bytes:
    """Build a minimal valid binary frame for testing."""
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:30, 10:50] = [0, 128, 255]
    _, jpeg_buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = jpeg_buf.tobytes()

    depth_u16 = np.ones((48, 64), dtype=np.uint16) * 1500
    depth_u16[0, 0] = 0
    depth_bytes = depth_u16.tobytes()

    header = {
        "session_id": "test-123",
        "frame_id": 1,
        "timestamp_ns": 1000000000,
        "unix_timestamp": 1700000000.0,
        "rgb_format": "jpeg",
        "rgb_width": 64,
        "rgb_height": 48,
        "image_orientation": "landscapeRight",
        "depth_format": "uint16_mm",
        "depth_width": 64,
        "depth_height": 48,
        "depth_scale": 0.001,
        "fx": 50.0,
        "fy": 50.0,
        "cx": 32.0,
        "cy": 24.0,
        "intrinsics_width": 64,
        "intrinsics_height": 48,
        "pose_format": "quat_translation",
        "T_wc": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
        "tracking_state": "normal",
        "tracking_reason": None,
        "pose_source": "arkit_vio",
    }
    header.update(overrides)
    json_bytes = json.dumps(header).encode("utf-8")

    msg = b""
    msg += struct.pack("<I", len(json_bytes))
    msg += json_bytes
    msg += struct.pack("<I", len(jpeg_bytes))
    msg += jpeg_bytes
    msg += struct.pack("<I", len(depth_bytes))
    msg += depth_bytes
    return msg


class TestParseBinaryMessage:
    def _make_receiver(self):
        from rtsm.io.websocket import WebSocketReceiver
        from rtsm.io.ingest_queue import IngestQueue

        q = IngestQueue(maxsize=16)
        recv = WebSocketReceiver(
            ingest_queue=q,
            keyframe_every_n=5,
            nonkf_min_interval_s=0.0,  # disable throttle for testing
        )
        return recv, q

    def test_full_roundtrip(self):
        recv, q = self._make_receiver()
        data = _build_test_frame()
        pkt = recv._parse_binary_message(data)
        assert pkt is not None
        assert pkt.rgb.shape == (48, 64, 3)
        assert pkt.depth_m is not None
        assert pkt.depth_m.shape == (48, 64)
        assert np.isnan(pkt.depth_m[0, 0]), "zero depth should be NaN"
        np.testing.assert_almost_equal(pkt.depth_m[1, 1], 1.5)
        np.testing.assert_array_almost_equal(pkt.pose.t_wc, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(pkt.pose.q_wc_xyzw, [0, 0, 0, 1])
        assert pkt.intr.width == 64
        assert pkt.intr.height == 48
        assert pkt.time.t_sensor_ns == 1000000000
        assert pkt.time.seq == 1
        assert pkt.is_keyframe  # first frame is always keyframe

    def test_tracking_state_filter(self):
        recv, q = self._make_receiver()
        data = _build_test_frame(tracking_state="limited")
        pkt = recv._parse_binary_message(data)
        assert pkt is None, "should drop frames with tracking_state != normal"

    def test_truncated_message(self):
        recv, q = self._make_receiver()
        pkt = recv._parse_binary_message(b"\x00\x00")
        assert pkt is None

    def test_intrinsics_scaling(self):
        recv, q = self._make_receiver()
        data = _build_test_frame(
            intrinsics_width=128,
            intrinsics_height=96,
            rgb_width=64,
            rgb_height=48,
            fx=100.0,
            fy=100.0,
            cx=64.0,
            cy=48.0,
        )
        pkt = recv._parse_binary_message(data)
        assert pkt is not None
        # Scale is 0.5x in both dimensions
        np.testing.assert_almost_equal(pkt.intr.fx, 50.0)
        np.testing.assert_almost_equal(pkt.intr.fy, 50.0)
        np.testing.assert_almost_equal(pkt.intr.cx, 32.0)
        np.testing.assert_almost_equal(pkt.intr.cy, 24.0)

    def test_no_depth(self):
        """Frame with no depth should have depth_m = None."""
        recv, q = self._make_receiver()
        # Build frame with no depth
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        _, jpeg_buf = cv2.imencode(".jpg", rgb)
        jpeg_bytes = jpeg_buf.tobytes()
        header = {
            "session_id": "test",
            "frame_id": 0,
            "timestamp_ns": 100,
            "unix_timestamp": 1700000000.0,
            "rgb_format": "jpeg",
            "rgb_width": 64,
            "rgb_height": 48,
            "depth_format": None,
            "depth_width": None,
            "depth_height": None,
            "depth_scale": None,
            "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0,
            "intrinsics_width": 64, "intrinsics_height": 48,
            "pose_format": "quat_translation",
            "T_wc": [0, 0, 0, 1, 0, 0, 0],
            "tracking_state": "normal",
        }
        json_bytes = json.dumps(header).encode("utf-8")
        msg = struct.pack("<I", len(json_bytes)) + json_bytes
        msg += struct.pack("<I", len(jpeg_bytes)) + jpeg_bytes
        msg += struct.pack("<I", 0)  # depth_len = 0
        pkt = recv._parse_binary_message(msg)
        assert pkt is not None
        assert pkt.depth_m is None


# ─────────────────── H.264 decode support ───────────────────

av = pytest.importorskip("av")
from fractions import Fraction


def _encode_h264_frame(img_bgr: np.ndarray) -> bytes:
    """Encode a BGR numpy array to H.264 IDR frame bytes using PyAV."""
    h, w = img_bgr.shape[:2]
    codec = av.CodecContext.create("libx264", "w")
    codec.width = w
    codec.height = h
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, 30)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency"}
    codec.open()

    frame = av.VideoFrame.from_ndarray(img_bgr, format="bgr24")
    frame.pts = 0
    packets = codec.encode(frame)
    packets += codec.encode()  # flush encoder
    return b"".join(bytes(p) for p in packets)


def _build_test_frame_h264(h264_bytes: bytes, width: int, height: int) -> bytes:
    """Build a binary WebSocket message with H.264 RGB payload."""
    depth_u16 = np.ones((height, width), dtype=np.uint16) * 1500
    depth_u16[0, 0] = 0
    depth_bytes = depth_u16.tobytes()

    header = {
        "session_id": "test-h264",
        "frame_id": 1,
        "timestamp_ns": 1000000000,
        "unix_timestamp": 1700000000.0,
        "rgb_format": "h264",
        "rgb_width": width,
        "rgb_height": height,
        "depth_format": "uint16_mm",
        "depth_width": width,
        "depth_height": height,
        "depth_scale": 0.001,
        "fx": 50.0, "fy": 50.0, "cx": 32.0, "cy": 24.0,
        "intrinsics_width": width,
        "intrinsics_height": height,
        "pose_format": "quat_translation",
        "T_wc": [0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
        "tracking_state": "normal",
    }
    json_bytes = json.dumps(header).encode("utf-8")

    msg = b""
    msg += struct.pack("<I", len(json_bytes))
    msg += json_bytes
    msg += struct.pack("<I", len(h264_bytes))
    msg += h264_bytes
    msg += struct.pack("<I", len(depth_bytes))
    msg += depth_bytes
    return msg


class TestH264Decoder:
    """Tests for H.264 decode support (requires PyAV)."""

    @pytest.fixture
    def h264_idr_frame(self):
        """Generate a valid H.264 IDR frame from a test image."""
        img = np.zeros((48, 64, 3), dtype=np.uint8)
        img[10:30, 10:50] = [0, 128, 255]
        h264_bytes = _encode_h264_frame(img)
        return h264_bytes, img

    def test_decoder_roundtrip(self, h264_idr_frame):
        """H.264 encode → decode roundtrip should produce matching image."""
        from rtsm.io.websocket import H264Decoder

        h264_bytes, original = h264_idr_frame
        decoder = H264Decoder()
        result = decoder.decode(h264_bytes)

        assert result is not None
        assert result.shape == (48, 64, 3)
        assert result.dtype == np.uint8
        # H.264 is lossy — allow deviation from YUV420 chroma subsampling
        np.testing.assert_allclose(
            result.astype(float), original.astype(float), atol=35,
        )

    def test_decoder_returns_none_for_empty(self):
        """Decoder should handle packets that produce no frames gracefully."""
        from rtsm.io.websocket import H264Decoder

        decoder = H264Decoder()
        # SPS/PPS-only packets won't produce a decoded frame
        result = decoder.decode(b"")
        assert result is None

    def test_binary_message_h264(self, h264_idr_frame):
        """Full binary message with rgb_format=h264 should parse correctly."""
        from rtsm.io.websocket import WebSocketReceiver
        from rtsm.io.ingest_queue import IngestQueue

        h264_bytes, _ = h264_idr_frame
        recv = WebSocketReceiver(
            ingest_queue=IngestQueue(16),
            keyframe_every_n=5,
            nonkf_min_interval_s=0.0,
        )

        data = _build_test_frame_h264(h264_bytes, width=64, height=48)
        pkt = recv._parse_binary_message(data)

        assert pkt is not None
        assert pkt.rgb.shape == (48, 64, 3)
        assert pkt.rgb.dtype == np.uint8
        assert pkt.rgb_jpeg is None  # H.264 does not set raw JPEG
        assert pkt.depth_m is not None
        assert pkt.depth_m.shape == (48, 64)
        assert np.isnan(pkt.depth_m[0, 0])
        np.testing.assert_almost_equal(pkt.depth_m[1, 1], 1.5)
        np.testing.assert_array_almost_equal(pkt.pose.t_wc, [1.0, 2.0, 3.0])

    def test_h264_decoder_persists_across_frames(self, h264_idr_frame):
        """Decoder instance should be reused across multiple H.264 frames."""
        from rtsm.io.websocket import WebSocketReceiver
        from rtsm.io.ingest_queue import IngestQueue

        h264_bytes, _ = h264_idr_frame
        recv = WebSocketReceiver(
            ingest_queue=IngestQueue(16),
            keyframe_every_n=999,
            nonkf_min_interval_s=0.0,
        )

        # Parse first frame — creates decoder
        data1 = _build_test_frame_h264(h264_bytes, width=64, height=48)
        pkt1 = recv._parse_binary_message(data1)
        assert pkt1 is not None
        assert recv._h264_decoder is not None
        decoder_id = id(recv._h264_decoder)

        # Parse second frame — should reuse same decoder
        data2 = _build_test_frame_h264(h264_bytes, width=64, height=48)
        pkt2 = recv._parse_binary_message(data2)
        assert pkt2 is not None
        assert id(recv._h264_decoder) == decoder_id

    def test_existing_formats_unaffected(self):
        """JPEG/NV12/BGRA paths should still work after H.264 addition."""
        from rtsm.io.websocket import WebSocketReceiver
        from rtsm.io.ingest_queue import IngestQueue

        recv = WebSocketReceiver(
            ingest_queue=IngestQueue(16),
            keyframe_every_n=5,
            nonkf_min_interval_s=0.0,
        )

        # JPEG frame (existing _build_test_frame uses JPEG)
        data = _build_test_frame()
        pkt = recv._parse_binary_message(data)
        assert pkt is not None
        assert pkt.rgb.shape == (48, 64, 3)
        assert pkt.rgb_jpeg is not None  # JPEG preserves raw bytes
        assert recv._h264_decoder is None  # should NOT create H.264 decoder
