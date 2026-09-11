from __future__ import annotations

import queue
from typing import Optional

from rtsm.core.datamodel import FramePacket


class IngestQueue:
    """
    Thread-safe queue for delivering FramePacket objects from IO/subscribers
    to the core pipeline.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=maxsize)

    def put(self, pkt: FramePacket, block: bool = False, timeout: Optional[float] = None) -> bool:
        try:
            self._q.put(pkt, block=block, timeout=0.0 if timeout is None else timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        try:
            return self._q.get(timeout=0.0 if timeout is None else timeout)
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._q.qsize()

    def full(self) -> bool:
        """True when a non-blocking put would fail right now. Receivers consult
        this BEFORE decoding a frame (admit-before-decode) so a congested
        pipeline does not make the receiver decode frames it will drop."""
        return self._q.full()

    @property
    def maxsize(self) -> int:
        return int(self._q.maxsize)

    # ── surface shared with rtsm/io/ingest_lanes.IngestLanes (P1 task 3) ──
    # put()/get() above are deliberately untouched: this class IS the
    # `ingest.policy: legacy` rollback and must behave exactly as before.

    policy = "legacy"

    def refusal(self, is_keyframe: bool, keyframe_origin: Optional[str] = None) -> Optional[str]:
        """Admit-before-decode check: the legacy queue refuses any frame
        while full."""
        return "queue_full" if self._q.full() else None

    def depth(self) -> dict:
        return {"legacy": self._q.qsize()}

    def backlog_signal(self) -> dict:
        n = self._q.qsize()
        return {"lane_full": n >= self._q.maxsize, "age_dropped": 0, "depth": {"legacy": n}}

    def stats(self) -> dict:
        n = self._q.qsize()
        return {"policy": self.policy, "maxsize": self.maxsize, "depth": {"legacy": n},
                "lane_full": n >= self._q.maxsize}

    def set_on_drop(self, cb) -> None:
        """No lane-side drops exist in the legacy queue (tail-drop refusals
        are reported by put() -> False and traced by the receiver)."""
        return None

    def close(self) -> None:
        return None


