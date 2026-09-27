"""P3 task 1 -- the ROS image / depth helpers of the codec layer: every colour
encoding keyed on the message field (the channel-order rule), both float-depth
invalid conventions, compressedDepth PNG (16UC1 and 32FC1 quantised), explicit
errors for RVL, big-endian and unknown encodings."""
from __future__ import annotations

import struct

import cv2
import numpy as np
import pytest

from rtsm.io import codecs
from rtsm.io.codecs import UnsupportedEncoding


def _bgr(h=4, w=6):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = 200                        # B
    img[..., 1] = np.arange(w, dtype=np.uint8)[None, :] * 10   # G varies
    img[..., 2] = 30                         # R  (B != R: the trap is armed)
    return img


def test_raw_colour_encodings_keyed_on_the_field():
    bgr = _bgr()
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    assert np.array_equal(codecs.ros_image_to_bgr("bgr8", bgr.tobytes(), w, h), bgr)
    assert np.array_equal(codecs.ros_image_to_bgr("rgb8", rgb.tobytes(), w, h), bgr)
    assert np.array_equal(codecs.ros_image_to_bgr("8UC3", bgr.tobytes(), w, h), bgr)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    assert np.array_equal(codecs.ros_image_to_bgr("rgba8", rgba.tobytes(), w, h), bgr)
    assert np.array_equal(codecs.ros_image_to_bgr("bgra8", bgra.tobytes(), w, h), bgr)
    mono = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    out = codecs.ros_image_to_bgr("mono8", mono.tobytes(), w, h)
    assert out.shape == (h, w, 3) and np.array_equal(out[..., 0], mono) and np.array_equal(out[..., 2], mono)
    # the mirror is NOT accepted: a reader that ignored the field would produce it
    assert not np.array_equal(codecs.ros_image_to_bgr("rgb8", rgb.tobytes(), w, h), rgb)


def test_row_padding_and_size_checks():
    bgr = _bgr()
    h, w = bgr.shape[:2]
    padded = np.zeros((h, w * 3 + 4), dtype=np.uint8)
    padded[:, : w * 3] = bgr.reshape(h, w * 3)
    assert np.array_equal(codecs.ros_image_to_bgr("bgr8", padded.tobytes(), w, h, step=w * 3 + 4), bgr)
    with pytest.raises(ValueError, match="too small"):
        codecs.ros_image_to_bgr("bgr8", bgr.tobytes()[:-1], w, h)
    with pytest.raises(UnsupportedEncoding, match="big-endian"):
        codecs.ros_image_to_bgr("bgr8", bgr.tobytes(), w, h, is_bigendian=1)
    with pytest.raises(UnsupportedEncoding, match="yuv422"):
        codecs.ros_image_to_bgr("yuv422", bgr.tobytes(), w, h)


def test_compressed_image_formats():
    bgr = _bgr(16, 24)
    ok, png = cv2.imencode(".png", bgr)                              # lossless: exact round trip
    assert ok
    assert np.array_equal(codecs.ros_compressed_to_bgr("png", png.tobytes()), bgr)
    assert np.array_equal(codecs.ros_compressed_to_bgr("bgr8; png compressed bgr8", png.tobytes()), bgr)
    # image_transport says the compressed bytes are RGB-ordered -> swap
    ok, png_rgb = cv2.imencode(".png", np.ascontiguousarray(bgr[..., ::-1]))
    assert np.array_equal(codecs.ros_compressed_to_bgr("rgb8; png compressed rgb8", png_rgb.tobytes()), bgr)
    ok, jpg = cv2.imencode(".jpg", bgr)
    out = codecs.ros_compressed_to_bgr("jpeg", jpg.tobytes())
    assert out.shape == bgr.shape and out.dtype == np.uint8
    with pytest.raises(UnsupportedEncoding):
        codecs.ros_compressed_to_bgr("16UC1; compressedDepth png", jpg.tobytes())
    with pytest.raises(UnsupportedEncoding):
        codecs.ros_compressed_to_bgr("webp", jpg.tobytes())
    with pytest.raises(ValueError):
        codecs.ros_compressed_to_bgr("jpeg", b"not an image")


def test_depth_wire_mapping_and_both_invalid_conventions():
    d16 = np.array([[0, 500], [1200, 65535]], dtype=np.uint16)
    raw, enc, scale = codecs.ros_depth_wire("16UC1", d16.tobytes(), 2, 2)
    assert (enc, scale) == ("uint16_mm", 0.001) and raw == d16.tobytes()
    dm = codecs.decode_depth(raw, enc, 2, 2, scale)
    assert np.isnan(dm[0, 0]) and dm[0, 1] == pytest.approx(0.5) and dm[1, 1] == pytest.approx(65.535)
    assert codecs.ros_depth_wire("mono16", d16.tobytes(), 2, 2)[1] == "uint16_mm"
    d32 = np.array([[np.nan, 0.0], [1.25, 2.5]], dtype=np.float32)
    raw, enc, scale = codecs.ros_depth_wire("32FC1", d32.tobytes(), 2, 2)
    assert (enc, scale) == ("float32_m", 1.0)
    dm = codecs.decode_depth(raw, enc, 2, 2, scale)
    assert np.isnan(dm[0, 0]) and np.isnan(dm[0, 1]) and dm[1, 0] == 1.25          # NaN (TUM) and 0.0 (D455) both invalid
    with pytest.raises(UnsupportedEncoding):
        codecs.ros_depth_wire("8UC1", b"\x00" * 4, 2, 2)
    with pytest.raises(UnsupportedEncoding, match="big-endian"):
        codecs.ros_depth_wire("16UC1", d16.tobytes(), 2, 2, is_bigendian=1)
    with pytest.raises(ValueError, match="too small"):
        codecs.ros_depth_wire("16UC1", d16.tobytes()[:-2], 2, 2)


def _compressed_depth(arr_u16: np.ndarray, fmt: str, quant_a=0.0, quant_b=0.0) -> bytes:
    ok, png = cv2.imencode(".png", arr_u16)
    assert ok
    return struct.pack("<iff", 0, quant_a, quant_b) + png.tobytes()


def test_compressed_depth_png_16uc1_and_32fc1():
    d16 = np.array([[0, 700], [1500, 3000]], dtype=np.uint16)
    raw, enc, scale = codecs.ros_compressed_depth("16UC1; compressedDepth png", _compressed_depth(d16, "16UC1"))
    assert (enc, scale) == ("uint16_mm", 0.001) and raw == d16.tobytes()
    # 32FC1: q = A / d + B  ->  d = A / (q - B); 0 = invalid
    A, B = 1000.0, 0.5                                             # fine quantisation: <= 0.5 % error at 4 m
    depth = np.array([[0.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    q = np.where(depth > 0, np.round(A / np.maximum(depth, 1e-6) + B), 0).astype(np.uint16)
    raw, enc, scale = codecs.ros_compressed_depth("32FC1; compressedDepth png", _compressed_depth(q, "32FC1", A, B))
    assert (enc, scale) == ("float32_m", 1.0)
    dm = codecs.decode_depth(raw, enc, 2, 2, scale)
    assert np.isnan(dm[0, 0]) and np.allclose(dm[0, 1], 1.0, rtol=1e-2) and np.allclose(dm[1, 1], 4.0, rtol=1e-2)
    with pytest.raises(UnsupportedEncoding, match="rvl"):
        codecs.ros_compressed_depth("16UC1; compressedDepth rvl", b"\x00" * 20)
    with pytest.raises(ValueError):
        codecs.ros_compressed_depth("16UC1; compressedDepth png", b"\x00" * 5)
    with pytest.raises(ValueError):
        codecs.ros_compressed_depth("16UC1; compressedDepth png", struct.pack("<iff", 0, 0, 0) + b"not png")
