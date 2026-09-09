"""
Replay Receiver — reads a recorded WebSocket session and feeds it
through the RTSM pipeline at the original recording rate.

Usage:
    uv run rtsm-run --replay path/to/recording
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import List, Optional

from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.websocket import WebSocketReceiver
from rtsm.evaluation.event_log import RX_DROPPED, RX_ENQUEUED, RX_QUEUE_FULL

logger = logging.getLogger(__name__)


class ReplayReceiver:
    """Replays a recorded WebSocket session through the RTSM pipeline.

    Implements the same lifecycle contract as WebSocketReceiver:
    start() launches a daemon thread, stop() signals it to halt.
    Frames are fed at the original recording rate (real-time) so that
    TTL caches and throttles in the pipeline behave identically.
    """

    def __init__(
        self,
        recording_dir: str,
        ingest_queue: IngestQueue,
        *,
        require_tracking_normal: bool = True,
        keyframe_every_n: int = 30,
        nonkf_min_interval_s: float = 0.5,
        confidence_threshold: int = 1,
        apply_camera_flip: bool = False,
        on_keyframe: Optional[callable] = None,
        on_camera_frame: Optional[callable] = None,
        on_pose_corrections: Optional[callable] = None,
        on_pose_corrections_batch: Optional[callable] = None,
        latency_analytics=None,
        replay_speed: float = 1.0,
        event_sink: Optional[callable] = None,
        throttle_clock: str = "wall",
    ) -> None:
        self._recording_dir = os.path.abspath(recording_dir)
        self._ingest_q = ingest_queue
        self._on_keyframe = on_keyframe
        self._on_camera_frame = on_camera_frame

        # Validate recording directory
        bin_path = os.path.join(self._recording_dir, "messages.bin")
        idx_path = os.path.join(self._recording_dir, "index.jsonl")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"Recording missing messages.bin: {bin_path}")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"Recording missing index.jsonl: {idx_path}")

        self._bin_path = bin_path
        self._idx_path = idx_path
        self._txt_path = os.path.join(self._recording_dir, "text_messages.jsonl")

        # Create a WebSocketReceiver as a decoder-only instance (never started).
        # We reuse _parse_binary_message and _handle_text_message directly.
        self._latency_analytics = latency_analytics
        self._decoder = WebSocketReceiver(
            ingest_queue=IngestQueue(1),  # dummy, never used
            require_tracking_normal=require_tracking_normal,
            keyframe_every_n=keyframe_every_n,
            nonkf_min_interval_s=nonkf_min_interval_s,
            confidence_threshold=confidence_threshold,
            apply_camera_flip=apply_camera_flip,
            on_keyframe=None,  # we handle callbacks ourselves
            on_pose_corrections=on_pose_corrections,
            on_pose_corrections_batch=on_pose_corrections_batch,
            latency_analytics=latency_analytics,  # passed to decoder for frame_received/tracking/throttle hooks
            # Frame-flow trace: the decoder emits the parse-side decisions
            # (malformed / parse_error / tracking_state / throttle) tagged
            # source="replay"; the enqueue decisions are emitted below. Depth
            # is read from the REAL ingest queue, not the decoder's dummy.
            event_sink=event_sink,
            event_source="replay",
            trace_queue=ingest_queue,
            # ingest.clock: "sensor" makes the non-KF throttle compare recorded
            # header timestamps, so the admitted set is the same at any speed.
            throttle_clock=throttle_clock,
        )

        self._replay_speed = max(0.1, replay_speed)  # <1 = slower, >1 = faster

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._done = threading.Event()

    def start(self) -> None:
        """Launch the replay loop in a daemon thread."""
        self._thread = threading.Thread(
            target=self._replay_loop, name="replay-receiver", daemon=True
        )
        self._thread.start()
        logger.info(f"[replay] Started from {self._recording_dir}")

    def stop(self) -> None:
        """Signal the replay thread to stop."""
        self._stop_event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until replay finishes. Returns True if completed."""
        return self._done.wait(timeout=timeout)

    # ── Core replay loop ──

    def _replay_loop(self) -> None:
        # Load index
        binary_entries = self._load_index()
        text_entries = self._load_text_messages()

        # Merge into single timeline
        timeline: List[dict] = []
        for entry in binary_entries:
            entry["_kind"] = "binary"
            timeline.append(entry)
        for entry in text_entries:
            entry["_kind"] = "text"
            timeline.append(entry)
        timeline.sort(key=lambda e: e["t_mono_s"])

        total_binary = len(binary_entries)
        frames_enqueued = 0

        if not timeline:
            logger.warning("[replay] Empty recording, nothing to replay")
            self._done.set()
            return

        # Start from the first entry's timestamp to skip initial dead time
        # (e.g. gap between process start and device connection)
        prev_t = timeline[0]["t_mono_s"]

        logger.info(
            f"[replay] Timeline: {total_binary} binary frames, "
            f"{len(text_entries)} text messages"
        )

        with open(self._bin_path, "rb") as bin_f:
            for entry in timeline:
                if self._stop_event.is_set():
                    break

                # Sleep to match original recording rate (adjusted by replay_speed)
                t_mono = entry["t_mono_s"]
                delta = t_mono - prev_t
                if delta > 0:
                    adjusted_delta = delta / self._replay_speed
                    # Use small sleep chunks so stop_event is responsive
                    deadline = time.monotonic() + adjusted_delta
                    while time.monotonic() < deadline:
                        if self._stop_event.is_set():
                            break
                        remaining = deadline - time.monotonic()
                        time.sleep(min(remaining, 0.1))
                prev_t = t_mono

                kind = entry["_kind"]

                if kind == "binary":
                    bin_f.seek(entry["offset"])
                    raw = bin_f.read(entry["length"])

                    pkt = self._decoder._parse_binary_message(raw)
                    if pkt is not None:
                        # Broadcast camera frame for every decoded packet (full frame rate PiP)
                        if self._on_camera_frame is not None:
                            try:
                                self._on_camera_frame(pkt)
                            except Exception as e:
                                logger.error(f"[replay] on_camera_frame callback error: {e}")
                        if self._latency_analytics:
                            self._latency_analytics.sample_queue_depth(self._ingest_q.qsize())
                        ok = self._ingest_q.put(pkt, block=False)
                        if ok:
                            frames_enqueued += 1
                            self._decoder._trace_rx(RX_ENQUEUED, "", pkt=pkt)
                            # (throttle stamp advanced at the decoder's admit decision)
                            if pkt.is_keyframe and self._on_keyframe is not None:
                                try:
                                    self._on_keyframe(pkt)
                                except Exception as e:
                                    logger.error(f"[replay] on_keyframe callback error: {e}")
                        else:
                            if self._latency_analytics:
                                self._latency_analytics.record_queue_drop()
                            logger.warning("[replay] ingest queue full; dropping frame")
                            self._decoder._trace_rx(RX_DROPPED, RX_QUEUE_FULL, pkt=pkt)

                elif kind == "text":
                    try:
                        self._decoder._handle_text_message(entry["payload"])
                    except Exception as e:
                        logger.error(f"[replay] text message error: {e}")

        remaining_q = self._ingest_q.qsize()
        logger.info(
            f"[replay] Complete: {total_binary} frames replayed, "
            f"{frames_enqueued} enqueued, {remaining_q} remaining in queue"
        )
        self._done.set()

    def _load_index(self) -> List[dict]:
        entries = []
        with open(self._idx_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def _load_text_messages(self) -> List[dict]:
        if not os.path.isfile(self._txt_path):
            return []
        entries = []
        with open(self._txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries
