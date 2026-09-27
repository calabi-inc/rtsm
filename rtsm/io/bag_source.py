"""
The ``bag`` ingest source (Gate 4.5 plan, P3 task 1): a transport adapter on
the one ingest front-end that reads a ROS 1 bag, a rosbag2 directory or a
bare MCAP through ``rtsm.io.bag_reader`` and offers every paired RGB-D frame
with its composed pose. Same policy chain as the websocket / replay path
(minted keyframes every N, non-keyframe throttle on the ingest clock,
admission before decode, decode on admit), so a bag is evaluated exactly the
way a live stream is ingested.

Pacing: ``speed=None`` (the eval default) offers frames as fast as the ingest
queue accepts them -- under the lossless lane the producer waits, nothing is
dropped; ``speed=1.0`` sleeps by header-stamp deltas like the replayer.

Poses in bags are already in the OpenCV camera convention (ROS optical frames),
so ``SourceContext.apply_camera_flip`` is ignored here (logged once).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

from rtsm.io.bag_reader import BagRefusal, BagStats, iter_bag_frames
from rtsm.io.contracts import SourceContext
from rtsm.io.ingest_frontend import WEBSOCKET_POLICY, IngestFrontEnd

logger = logging.getLogger(__name__)


class BagSource:
    name = "bag"

    def __init__(self, cfg: dict, ctx: SourceContext, *, path: str | os.PathLike, speed: Optional[float] = None,
                 topics: Optional[Dict[str, str]] = None, pair_tolerance_s: float = 0.02, tf_extrapolation_s: float = 0.05,
                 world_frame: Optional[str] = None, camera_frame: Optional[str] = None, assume_aligned: bool = False,
                 typestore: str = "humble", max_frames: Optional[int] = None, source_name: str = "bag") -> None:
        if not path:
            raise ValueError("bag source needs a path (io.bag.path or --bag)")
        self._path = str(path)
        self._reader_kw = dict(topics=topics, pair_tolerance_s=float(pair_tolerance_s), tf_extrapolation_s=float(tf_extrapolation_s),
                               world_frame=world_frame, camera_frame=camera_frame, assume_aligned=bool(assume_aligned),
                               typestore=str(typestore), max_frames=max_frames)
        self._speed = (float(speed) if speed else None)
        self._ctx = ctx
        self._require_tracking_cfg = bool(ctx.require_tracking_normal)
        if ctx.apply_camera_flip:
            logger.info("[bag] visualization.apply_camera_flip is ignored for bags: ROS optical frames are already the OpenCV convention")
        self._fe = IngestFrontEnd(
            source=source_name, policy=WEBSOCKET_POLICY, ingest_queue=ctx.ingest_queue,
            throttle_clock=ctx.clock_mode, keyframe_every_n=ctx.keyframe_every_n, keyframe_interval_s=ctx.keyframe_interval_s,
            nonkf_min_interval_s=ctx.nonkf_min_interval_s, require_tracking_normal=False,   # decided per bag (tracking topic present?)
            confidence_threshold=ctx.confidence_threshold, pose_sink=ctx.pose_sink, clearance_sink=ctx.clearance_sink,
            event_sink=ctx.event_sink, ledger_sink=ctx.ledger_sink, latency_analytics=ctx.latency_analytics,
            on_camera_frame=ctx.on_camera_frame, on_keyframe=ctx.on_keyframe,
        )
        self._stats = BagStats()
        self._enqueued = 0
        self._admit_errors = 0
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._done = threading.Event()

    # ── Source protocol ──

    @property
    def frontend(self) -> IngestFrontEnd:
        return self._fe

    @property
    def path(self) -> str:
        return self._path

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="bag-source", daemon=True)
        self._thread.start()
        logger.info("[bag] started: %s (speed %s)", self._path, self._speed or "as fast as accepted")

    def stop(self) -> None:
        """Signal the thread; under the lossless lane also close the queue so a
        blocked put() wakes up (the replayer's rule)."""
        self._stop_event.set()
        q = self._ctx.ingest_queue
        if getattr(q, "policy", None) == "lossless":
            close = getattr(q, "close", None)
            if callable(close):
                close()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout=timeout)

    def liveness(self) -> dict:
        t = self._thread
        return {"alive": bool(t is not None and t.is_alive()), **self._fe.liveness()}

    def stats(self) -> Dict[str, Any]:
        d = self._stats.as_dict()
        d.update({"path": self._path, "enqueued": self._enqueued, "admit_errors": self._admit_errors,
                  "done": self._done.is_set(), "error": (str(self._error) if self._error else None),
                  "refusal_codes": (list(self._error.codes) if isinstance(self._error, BagRefusal) else None)})
        return d

    @property
    def error(self) -> Optional[BaseException]:
        return self._error

    # ── the loop ──

    def _loop(self) -> None:
        try:
            self._loop_impl()
        except BagRefusal as e:
            self._error = e
            logger.error("[bag] refused %s: %s", self._path, e)
        except Exception as e:  # noqa: BLE001 -- surfaced through stats()/error, never silently lost
            self._error = e
            logger.exception("[bag] reader failed on %s", self._path)
        finally:
            self._done.set()

    def _loop_impl(self) -> None:
        first = True
        prev_stamp: Optional[int] = None
        for raw in iter_bag_frames(self._path, stats=self._stats, source=self._fe.source, **self._reader_kw):
            if self._stop_event.is_set():
                break
            if first:
                first = False
                # the tracking filter only means something when the bag carries a tracking state
                has_tracking = bool((self._stats.topics or {}).get("tracking"))
                self._fe.require_tracking_normal = self._require_tracking_cfg and has_tracking
                self._fe.new_session(raw.header.session_id or os.path.basename(os.path.normpath(self._path)))
                logger.info("[bag] topics %s | pose %s %s -> %s | tracking filter %s",
                            {k: v for k, v in (self._stats.topics or {}).items() if k != "rules" and v}, self._stats.pose_kind,
                            self._stats.world_frame, self._stats.camera_frame, "on" if self._fe.require_tracking_normal else "off (no tracking topic)")
            if self._speed and prev_stamp is not None and raw.header.t_sensor_ns:
                delay = (raw.header.t_sensor_ns - prev_stamp) / 1e9 / self._speed
                deadline = time.monotonic() + max(0.0, delay)
                while time.monotonic() < deadline and not self._stop_event.is_set():
                    time.sleep(min(deadline - time.monotonic(), 0.1))
            prev_stamp = raw.header.t_sensor_ns
            self._fe.last_rx_mono = time.monotonic()
            try:
                pkt = self._fe.admit(raw)
            except Exception as e:  # noqa: BLE001 -- parse_error line written inside admit(); one bad frame never ends the bag
                self._admit_errors += 1
                logger.warning("[bag] frame seq %s skipped: %s", raw.header.seq, e)
                continue
            if pkt is not None and self._fe.enqueue(pkt):
                self._enqueued += 1
        logger.info("[bag] complete: %s frames yielded, %s enqueued, %s unpaired rgb, %s without pose",
                    self._stats.yielded, self._enqueued, self._stats.unpaired_rgb, self._stats.pose_missing)
