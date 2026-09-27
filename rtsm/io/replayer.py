"""
Replay Receiver — reads a recorded WebSocket session and feeds it
through the RTSM pipeline at the original recording rate.

Usage:
    uv run rtsm-run --replay path/to/recording

Since P3 task 0.5 the replayer is a transport adapter like the live
receiver: it reads the recording's binary messages and text messages and
drives the SAME Lens framing + ingest front-end (rtsm/io/ingest_frontend.py)
on the real ingest queue, tagged ``source="replay"``. The decoder-only
``WebSocketReceiver`` instance with a dummy queue is gone.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import List, Optional

from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, IngestFrontEnd
from rtsm.io.websocket import handle_lens_text_message, parse_lens_message

logger = logging.getLogger(__name__)


class ReplayReceiver:
    """Replays a recorded WebSocket session through the RTSM pipeline.

    Implements the same lifecycle contract as WebSocketReceiver:
    start() launches a daemon thread, stop() signals it to halt.
    Frames are fed at the original recording rate (real-time) so that
    TTL caches and throttles in the pipeline behave identically.
    """

    name = "replay"

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
        pose_sink: Optional[callable] = None,
        ledger_sink: Optional[callable] = None,
        on_frame_correction: Optional[callable] = None,
    ) -> None:
        self._recording_dir = os.path.abspath(recording_dir)
        self._ingest_q = ingest_queue

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

        self._latency_analytics = latency_analytics
        self._apply_camera_flip = apply_camera_flip
        self._on_pose_corrections = on_pose_corrections
        self._on_pose_corrections_batch = on_pose_corrections_batch
        self._on_frame_correction = on_frame_correction
        # The one ingest chain on the REAL queue: the admit-before-decode
        # check, the trace depth, the throttle (ingest.clock: "sensor" makes
        # the non-KF throttle compare recorded header timestamps, so the
        # admitted set is the same at any speed), the receive-time robot pose
        # and the P2 ledgers all see the queue the frames actually go to.
        self._fe = IngestFrontEnd(
            source="replay", policy=WEBSOCKET_POLICY, ingest_queue=ingest_queue,
            throttle_clock=throttle_clock, keyframe_every_n=keyframe_every_n,
            nonkf_min_interval_s=nonkf_min_interval_s, require_tracking_normal=require_tracking_normal,
            confidence_threshold=confidence_threshold, pose_sink=pose_sink, event_sink=event_sink,
            ledger_sink=ledger_sink, latency_analytics=latency_analytics,
            on_camera_frame=on_camera_frame, on_keyframe=on_keyframe,
        )

        self._replay_speed = max(0.1, replay_speed)  # <1 = slower, >1 = faster

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._done = threading.Event()

    @property
    def frontend(self) -> IngestFrontEnd:
        return self._fe

    def start(self) -> None:
        """Launch the replay loop in a daemon thread."""
        self._thread = threading.Thread(
            target=self._replay_loop, name="replay-receiver", daemon=True
        )
        self._thread.start()
        logger.info(f"[replay] Started from {self._recording_dir}")

    def stop(self) -> None:
        """Signal the replay thread to stop. Under ingest.policy=lossless this
        also CLOSES the ingest queue -- terminally: the replay thread may be
        blocked in put() and only close() wakes it; get() keeps draining but
        no later put is admitted. Other policies never block, so their queue
        is left open."""
        self._stop_event.set()
        if getattr(self._ingest_q, "policy", None) == "lossless":
            close = getattr(self._ingest_q, "close", None)
            if callable(close):
                close()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until replay finishes. Returns True if completed."""
        return self._done.wait(timeout=timeout)

    def liveness(self) -> dict:
        t = self._thread
        return {"alive": bool(t is not None and t.is_alive()), **self._fe.liveness()}

    # ── Core replay loop ──

    def _replay_loop(self) -> None:
        # _done is set on EVERY exit (also an unexpected exception): run.py /
        # demo.py block on wait(), and a hung replay is worse than a short one.
        try:
            self._replay_loop_impl()
        finally:
            self._done.set()

    def _replay_loop_impl(self) -> None:
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
                    self._fe.last_rx_mono = time.monotonic()

                    try:
                        pkt = parse_lens_message(self._fe, raw, apply_camera_flip=self._apply_camera_flip,
                                                 latency_analytics=self._latency_analytics)
                    except Exception as e:
                        # Same as the live stream loop: one bad frame (the
                        # parse_error trace line is emitted inside the
                        # front-end, with the header ids) must not end the
                        # replay.
                        logger.error(f"[replay] frame parse error: {e}")
                        continue
                    if pkt is not None and self._fe.enqueue(pkt):
                        frames_enqueued += 1

                elif kind == "text":
                    try:
                        handle_lens_text_message(
                            entry["payload"], apply_camera_flip=self._apply_camera_flip,
                            on_pose_corrections=self._on_pose_corrections,
                            on_pose_corrections_batch=self._on_pose_corrections_batch,
                            on_frame_correction=self._on_frame_correction, source="replay",
                        )
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
