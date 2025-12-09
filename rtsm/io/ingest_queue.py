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


