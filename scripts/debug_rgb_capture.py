"""
Debug RGB capture — mimics rtsm.io.websocket.WebSocketReceiver.

Same FastAPI/uvicorn server on ws://0.0.0.0:8765/stream, same handshake,
same binary parsing, same NV12→BGR decode.  Instead of building a
FramePacket and enqueuing to the pipeline, it simply saves each decoded
RGB frame as a PNG under tests/debug_rgb/.

Use this to verify that the source image is gravitationally aligned
before any RTSM processing touches it.

Usage:
    python scripts/debug_rgb_capture.py
    python scripts/debug_rgb_capture.py --port 8765 --max-frames 30
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import struct
import sys
import time

import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

PROTOCOL_VERSION = 1


# ── Decode helpers (copied from rtsm.io.websocket) ──────────────


def decode_rgb(raw: bytes, fmt: str, width: int, height: int) -> np.ndarray:
    """Decode RGB payload into (H, W, 3) uint8 BGR array."""
    if fmt in ("jpeg", "png"):
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imdecode failed for {fmt} ({len(raw)} B)")
        return img
    elif fmt == "bgra":
        expected = height * width * 4
        if len(raw) != expected:
            raise ValueError(f"bgra size mismatch: {len(raw)} vs {expected}")
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif fmt == "nv12":
        expected = height * width * 3 // 2
        if len(raw) != expected:
            raise ValueError(f"nv12 size mismatch: {len(raw)} vs {expected}")
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape(height * 3 // 2, width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    else:
        raise ValueError(f"Unsupported rgb_format: {fmt!r}")


# ── Binary frame parser (mirrors WebSocketReceiver._parse_binary_message) ──


def parse_frame(data: bytes):
    """
    Parse binary message: [json_len][json][rgb_len][rgb][depth_len][depth]

    Returns (header_dict, rgb_raw_bytes).  Depth is read but discarded.
    """
    offset = 0
    n = len(data)

    # JSON header
    if n < 4:
        raise ValueError("message too short for json_len")
    (json_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if n < offset + json_len:
        raise ValueError("message truncated at JSON payload")
    header = json.loads(data[offset : offset + json_len].decode("utf-8"))
    offset += json_len

    # RGB payload
    if n < offset + 4:
        raise ValueError("message truncated at rgb_len")
    (rgb_len,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if n < offset + rgb_len:
        raise ValueError("message truncated at RGB payload")
    rgb_bytes = data[offset : offset + rgb_len]
    offset += rgb_len

    # Depth payload (skip)
    if n >= offset + 4:
        (depth_len,) = struct.unpack_from("<I", data, offset)
        offset += 4 + depth_len

    return header, rgb_bytes


# ── Server ───────────────────────────────────────────────────────


def create_app(out_dir: str, max_frames: int) -> FastAPI:
    state = {"count": 0}

    app = FastAPI(title="RTSM Debug RGB Capture")

    @app.get("/health")
    def health():
        return JSONResponse({"status": "ok", "receiver": "debug_rgb"})

    @app.websocket("/stream")
    async def stream_endpoint(ws: WebSocket):
        await ws.accept()
        client_addr = ws.client.host if ws.client else "unknown"
        print(f"[debug_rgb] Client connected from {client_addr}")

        # ── Handshake (same as WebSocketReceiver._handle_stream) ──
        try:
            hello_raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
        except asyncio.TimeoutError:
            print(f"[debug_rgb] Handshake timeout from {client_addr}")
            await ws.close(code=4002, reason="handshake timeout")
            return
        except Exception:
            print(f"[debug_rgb] Connection lost during handshake")
            return

        try:
            hello = json.loads(hello_raw)
        except json.JSONDecodeError:
            await ws.close(code=4001, reason="invalid hello JSON")
            return

        if hello.get("type") != "hello":
            await ws.close(code=4001, reason="expected hello message")
            return

        proto_ver = hello.get("protocol_version", 0)
        if proto_ver != PROTOCOL_VERSION:
            ack = {
                "type": "hello_ack",
                "status": "error",
                "reason": f"unsupported protocol_version: {proto_ver}",
            }
            await ws.send_json(ack)
            await ws.close(code=4001, reason=ack["reason"])
            return

        session_id = hello.get("session_id", "unknown")
        device_name = hello.get("device_name", "unknown")

        ack = {
            "type": "hello_ack",
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
            "server": "debug_rgb",
            "session_id": session_id,
        }
        await ws.send_json(ack)
        print(
            f"[debug_rgb] Handshake OK: session={session_id}, "
            f"device={device_name}, proto_v={proto_ver}"
        )

        # ── Receive binary frames ──
        frames_saved = 0
        try:
            while state["count"] < max_frames:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                if "bytes" in msg and msg["bytes"]:
                    try:
                        header, rgb_raw = parse_frame(msg["bytes"])
                    except Exception as exc:
                        print(f"[debug_rgb] Parse error: {exc}")
                        continue

                    # Decode RGB (handles nv12, jpeg, png, bgra)
                    rgb_fmt = header.get("rgb_format", "jpeg")
                    rgb_w = int(header.get("rgb_width", 0))
                    rgb_h = int(header.get("rgb_height", 0))

                    try:
                        bgr = decode_rgb(rgb_raw, fmt=rgb_fmt, width=rgb_w, height=rgb_h)
                    except Exception as exc:
                        print(f"[debug_rgb] Decode error: {exc}")
                        continue

                    frame_id = header.get("frame_id", state["count"])
                    tracking = header.get("tracking_state", "?")

                    filename = f"frame_{state['count']:04d}_fid{frame_id}.png"
                    filepath = os.path.join(out_dir, filename)
                    cv2.imwrite(filepath, bgr)

                    h, w = bgr.shape[:2]
                    print(
                        f"[debug_rgb] Saved {filename}  ({w}x{h})  "
                        f"fmt={rgb_fmt}  tracking={tracking}"
                    )
                    state["count"] += 1
                    frames_saved += 1

                # Ignore text messages (pose_corrections etc.)

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[debug_rgb] Connection error: {e}")
        finally:
            print(
                f"[debug_rgb] Session {session_id} ended: "
                f"{frames_saved} frames saved"
            )

    return app


def main():
    import uvicorn

    p = argparse.ArgumentParser(
        description="Debug server — saves raw RGB from Calabi Lens for orientation check."
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    p.add_argument("--max-frames", type=int, default=30, help="Frames to capture (default: 30)")
    p.add_argument("--out-dir", default=None, help="Output dir (default: tests/debug_rgb)")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "debug_rgb")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[debug_rgb] Saving frames to: {out_dir}")
    print(f"[debug_rgb] Will capture up to {args.max_frames} frames.\n")

    app = create_app(out_dir, args.max_frames)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        ws_max_size=16 * 1024 * 1024,
    )


if __name__ == "__main__":
    main()
