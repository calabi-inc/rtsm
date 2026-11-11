"""Waiting for a monotonic instant in a source thread (the replay and the bag source pace their input with it)."""
from __future__ import annotations

import threading
import time
from typing import Optional

__all__ = ["sleep_until"]


def sleep_until(deadline: float, stop_event: Optional[threading.Event] = None, *, slice_s: float = 0.1) -> bool:
    """Sleep until ``time.monotonic()`` reaches ``deadline``, in slices of at most ``slice_s`` so that a stop
    request is seen within one slice. True when the instant was reached, False when the stop came first.

    The clock is read ONCE per slice and the slice is that reading's distance to the deadline. Reading it twice,
    once to decide whether to wait and once for the length, lets the deadline pass between the two readings: the
    thread can lose the processor there, and a clock that advances in ticks (Windows before Python 3.13: 15.6 ms)
    can tick there. The length is then negative, ``time.sleep`` raises ValueError, and the calling thread ends
    with its input half delivered.
    """
    while True:
        if stop_event is not None and stop_event.is_set():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return True
        time.sleep(min(remaining, slice_s))
