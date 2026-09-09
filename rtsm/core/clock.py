"""
Ingest clocks (Gate 4.5 plan, P1 task 1).

Every memory-timing decision (non-keyframe throttle, ingest-gate grace / TTL /
parallax ages, proto expiry, LTM upsert scheduling) reads *one* injected clock
instead of `time.monotonic()` directly:

  WallClock    process-monotonic time. The live default: decisions follow
               real elapsed time, as before this change.
  SensorClock  the frames' own timestamps. The dispatcher (pipeline) calls
               `advance()` for every dequeued frame; everything downstream then
               sees the frame's sensor time. Replays become speed-independent
               and repeatable, which is what the A/A determinism gate needs.

Design points of SensorClock:
  * Values stay in `time.monotonic()` magnitude: the first frame anchors
    `offset = wall_mono - t_sensor`, so every TTL/period expressed in seconds
    keeps working and mixed stamps never differ by a sensor epoch.
  * Within one `frame_epoch` the clock never runs backwards (out-of-order
    frames clamp to the last value).
  * On a `frame_epoch` change (ARKit session restart: sensor time jumps) the
    offset is re-based so `now_mono()` stays continuous. Clamping instead
    would freeze the sweep gate until sensor time caught up with the old
    epoch; jumping would expire every proto object at once.
  * Before the first frame `now_mono()` falls back to the wall clock, so
    objects created before any frame (none today) would still get a stamp.

Determinism only depends on differences between readings, so the wall-clock
anchor of the first frame does not make two runs differ.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now_mono(self) -> float: ...
    def advance(self, t_sensor_ns: Optional[int], frame_epoch: Optional[int] = None) -> float: ...


class WallClock:
    """Process-monotonic clock; `advance()` is a no-op."""

    name = "wall"

    def now_mono(self) -> float:
        return time.monotonic()

    def advance(self, t_sensor_ns: Optional[int], frame_epoch: Optional[int] = None) -> float:
        return time.monotonic()


class SensorClock:
    """Clock driven by the dequeued frames' sensor timestamps (see module doc)."""

    name = "sensor"

    def __init__(self, fallback: Callable[[], float] = time.monotonic,
                 rebase_after_s: float = 5.0) -> None:
        self._fallback = fallback
        self._lock = threading.Lock()
        self._now: Optional[float] = None
        self._offset: Optional[float] = None
        self._epoch: Optional[int] = None
        # A backwards sensor jump larger than this WITHIN an epoch is treated
        # as a restarted source (re-base, keep `now` continuous) rather than
        # out-of-order jitter (clamp). Without it a re-replayed recording or a
        # restart that did not bump frame_epoch would freeze the clock at its
        # old maximum and every timing gate would stall.
        self.rebase_after_s = float(rebase_after_s)
        self.rebases: int = 0          # epoch changes / large-jump re-bases seen
        self.clamped: int = 0          # backwards sensor stamps clamped

    def now_mono(self) -> float:
        now = self._now
        return now if now is not None else self._fallback()

    def reset(self) -> None:
        """Forget the anchor and epoch (POST /reset, new replay in-process).
        The next frame re-anchors; `now_mono()` falls back to wall until then."""
        with self._lock:
            self._now = None
            self._offset = None
            self._epoch = None

    def advance(self, t_sensor_ns: Optional[int], frame_epoch: Optional[int] = None) -> float:
        """Move the clock to a frame's sensor time. Returns the new `now_mono()`.

        A missing / non-positive sensor stamp leaves the clock where it is
        (the frame will be timed at the previous frame's sensor time).
        """
        if t_sensor_ns is None or int(t_sensor_ns) <= 0:
            return self.now_mono()
        ts = int(t_sensor_ns) * 1e-9
        with self._lock:
            if self._offset is None:
                # First frame: anchor sensor time to the current wall time.
                self._offset = self._fallback() - ts
                self._epoch = frame_epoch
                self._now = ts + self._offset
                return self._now
            assert self._now is not None
            if self._epoch is None and frame_epoch is not None:
                # First epoch information after an epoch-less start: adopt it,
                # this is not a change of epoch.
                self._epoch = frame_epoch
            elif frame_epoch is not None and frame_epoch != self._epoch:
                # New sensor epoch: keep `now` continuous, re-base the offset.
                self._offset = self._now - ts
                self._epoch = frame_epoch
                self.rebases += 1
                return self._now
            cand = ts + self._offset
            if cand < self._now:
                if (self._now - cand) > self.rebase_after_s:
                    # Large backwards jump inside one epoch: a restarted
                    # source or a re-replay. Re-base instead of freezing.
                    self._offset = self._now - ts
                    self.rebases += 1
                    return self._now
                self.clamped += 1
                return self._now
            self._now = cand
            return cand

    @property
    def epoch(self) -> Optional[int]:
        return self._epoch


def make_clock(mode: str) -> "WallClock | SensorClock":
    """`wall` -> WallClock, `sensor` -> SensorClock (callers resolve `auto` first)."""
    mode = str(mode or "wall").lower()
    if mode == "sensor":
        return SensorClock()
    if mode == "wall":
        return WallClock()
    raise ValueError(f"ingest.clock must be auto|wall|sensor, got {mode!r}")


def resolve_clock_mode(configured: Optional[str], *, replay: bool) -> str:
    """Resolve `ingest.clock` (`auto|wall|sensor`) to `wall` or `sensor`.

    auto = sensor when replaying a recording (results must not depend on replay
    speed), wall for live receivers (unchanged until measured live).
    """
    mode = str(configured or "auto").lower()
    if mode == "auto":
        return "sensor" if replay else "wall"
    if mode in ("wall", "sensor"):
        return mode
    raise ValueError(f"ingest.clock must be auto|wall|sensor, got {configured!r}")
