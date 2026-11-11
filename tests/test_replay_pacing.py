"""A source thread waits for the next message's instant in slices, so that stop() is seen within one slice.

The wait has to read the clock once per slice. With two readings (one to decide whether to wait, one for the
length of the slice) the deadline can pass between them; the length is then negative, time.sleep raises
ValueError and the thread ends with the recording half replayed. The run used to finish normally all the same.

These tests drive the wait and both sources with a clock that advances on every reading, so that a deadline
falls between two consecutive readings, and check that a source that does stop on an error says so. CPU-only.
"""
from __future__ import annotations

import json
import threading
import time as real_time

import pytest

from rtsm.io.ingest_queue import IngestQueue


class SteppingTime:
    """Stands in for the ``time`` module inside a source: every reading of the monotonic clock advances it by
    ``step`` and a sleep advances it by its length. A negative length raises, as ``time.sleep`` does."""

    def __init__(self, step: float, start: float = 1000.0) -> None:
        self.step, self.now = float(step), float(start)
        self.slept: list = []
        self.readings = 0

    def monotonic(self) -> float:
        self.readings += 1
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        self.slept.append(float(seconds))
        self.now += float(seconds)

    def __getattr__(self, name):                       # time.time, perf_counter, ...: the real ones
        return getattr(real_time, name)


def _use_clock(monkeypatch, clock: SteppingTime, *modules) -> None:
    """Give the source modules (and the shared wait, where it exists) the stepping clock instead of ``time``."""
    import importlib
    for name in modules + ("rtsm.io.pacing",):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "time"):
            monkeypatch.setattr(mod, "time", clock)


# ───────────────────────────── the wait itself ─────────────────────────────

@pytest.mark.parametrize("wait_s", [0.0005, 0.0101, 0.25])
def test_sleep_until_never_sleeps_a_negative_length_whatever_the_clock_does(monkeypatch, wait_s):
    """Sweep the clock's step so that the deadline falls between two readings in many ways. A wait that reads the
    clock twice per slice hands a negative length to sleep for some of these steps; one reading per slice cannot."""
    from rtsm.io import pacing

    for k in range(1, 400):
        clock = SteppingTime(step=wait_s * k / 173.0)
        monkeypatch.setattr(pacing, "time", clock)
        deadline = clock.now + wait_s
        assert pacing.sleep_until(deadline) is True
        assert clock.now >= deadline                    # it returned at or after the instant, never before
        assert all(0.0 < s <= 0.1 for s in clock.slept)


def test_sleep_until_slices_a_long_wait_and_sees_a_stop_within_a_slice(monkeypatch):
    from rtsm.io import pacing

    clock = SteppingTime(step=1e-6)
    monkeypatch.setattr(pacing, "time", clock)
    assert pacing.sleep_until(clock.now + 0.35) is True
    assert len(clock.slept) == 4 and clock.slept[:3] == [0.1, 0.1, 0.1] and 0.04 < clock.slept[3] < 0.05

    stop = threading.Event()
    clock = SteppingTime(step=1e-6)
    real_sleep = clock.sleep

    def sleep_then_stop(seconds: float) -> None:
        real_sleep(seconds)
        stop.set()                                      # the stop arrives during the first slice

    clock.sleep = sleep_then_stop
    monkeypatch.setattr(pacing, "time", clock)
    assert pacing.sleep_until(clock.now + 10.0, stop) is False
    assert clock.slept == [0.1]                         # one slice, not the ten seconds

    stop = threading.Event(); stop.set()
    clock = SteppingTime(step=1e-6)
    monkeypatch.setattr(pacing, "time", clock)
    assert pacing.sleep_until(clock.now + 10.0, stop) is False and clock.slept == []
    assert pacing.sleep_until(clock.now - 1.0) is True and clock.slept == []      # an instant already past: no sleep


# ───────────────────────────── the replay source ─────────────────────────────

def _recording(tmp_path, n: int, gap_s: float):
    """A Lens recording of n frames whose index stamps are exactly gap_s apart."""
    from test_admit_before_decode import _ws_frame
    rec = tmp_path / "rec"
    rec.mkdir()
    offset = 0
    with open(rec / "messages.bin", "wb") as bin_f, open(rec / "index.jsonl", "w", encoding="utf-8") as idx_f:
        for i in range(n):
            msg = _ws_frame(i + 1, (i + 1) * 100_000_000)
            bin_f.write(msg)
            idx_f.write(json.dumps({"seq": i, "offset": offset, "length": len(msg), "t_mono_s": round(5.0 + i * gap_s, 6)}) + "\n")
            offset += len(msg)
    (rec / "meta.json").write_text(json.dumps({"session_id": "pacing"}), encoding="utf-8")
    return rec


def _replay(rec, speed: float, events: list):
    from rtsm.io.replayer import ReplayReceiver
    rr = ReplayReceiver(recording_dir=str(rec), ingest_queue=IngestQueue(maxsize=64), keyframe_every_n=30,
                        nonkf_min_interval_s=0.0, replay_speed=speed, event_sink=events.append)
    rr.start()
    assert rr.wait(20.0)
    return rr


# gap 10 ms. With a step between 5 and 10 ms the first reading after the deadline is set lies before it and the next
# one after it (step 6 ms: 4 ms before, 2 ms after); the other steps are there for the wait's ordinary paths.
@pytest.mark.parametrize("step_s", [0.006, 0.0099, 0.00501, 0.0034, 0.0000007])
def test_replay_survives_a_deadline_that_passes_between_two_clock_readings(tmp_path, monkeypatch, step_s):
    rec = _recording(tmp_path, 6, gap_s=0.010)
    clock = SteppingTime(step=step_s)
    _use_clock(monkeypatch, clock, "rtsm.io.replayer")
    events: list = []
    rr = _replay(rec, 1.0, events)
    assert [(e.decision, e.frame_seq) for e in events] == [("enqueued", i) for i in range(1, 7)]      # the whole recording
    assert all(s >= 0.0 for s in clock.slept)
    assert getattr(rr, "error", None) is None


def test_an_unpaced_replay_survives_a_clock_that_advances_in_ticks(tmp_path, monkeypatch):
    """What `rtsm eval` does with a recording when nothing is paced: a speed of 1e9, so every gap becomes a deadline
    a fraction of a nanosecond away. On a clock that stands still between ticks (Windows before Python 3.13) the wait
    then lasts until the next tick, and the tick may arrive between two readings."""
    rec = _recording(tmp_path, 8, gap_s=0.2)

    class TickingTime(SteppingTime):
        """Readings repeat a value until the tick advances; it advances on a schedule of readings, wherever they fall.
        A sleep shorter than a tick does not move the reading."""
        def __init__(self, advance_on: int) -> None:
            super().__init__(step=0.0)
            self.advance_on = advance_on

        def monotonic(self) -> float:
            self.readings += 1
            if self.readings % self.advance_on == 0:
                self.now += 0.015625
            return self.now

        def sleep(self, seconds: float) -> None:
            if seconds < 0:
                raise ValueError("sleep length must be non-negative")
            self.slept.append(float(seconds))

    for advance_on in (2, 3, 4, 5, 7):
        clock = TickingTime(advance_on)
        _use_clock(monkeypatch, clock, "rtsm.io.replayer")
        events: list = []
        rr = _replay(rec, 1e9, events)
        assert [(e.decision, e.frame_seq) for e in events] == [("enqueued", i) for i in range(1, 9)], advance_on
        assert getattr(rr, "error", None) is None


def test_a_replay_that_stops_on_an_error_says_so(tmp_path, monkeypatch, caplog):
    """wait() still returns (a hung replay is worse than a short one), but the short one is no longer taken for a
    whole one: the error is logged, kept on the source and reported in its statistics."""
    rec = _recording(tmp_path, 5, gap_s=0.010)

    class ThirdWaitFails(SteppingTime):
        def sleep(self, seconds: float) -> None:
            if len(self.slept) == 2:
                raise OSError("the wait failed")        # anything that is not caught per frame ends the thread
            super().sleep(seconds)

    clock = ThirdWaitFails(step=1e-6)
    _use_clock(monkeypatch, clock, "rtsm.io.replayer")
    monkeypatch.setattr(threading, "excepthook", lambda args: None)            # the pre-fix traceback stays out of the output
    events: list = []
    with caplog.at_level("ERROR", logger="rtsm.io.replayer"):
        rr = _replay(rec, 1.0, events)
    assert [(e.decision, e.frame_seq) for e in events] == [("enqueued", 1), ("enqueued", 2), ("enqueued", 3)]
    st = rr.stats()
    assert isinstance(rr.error, OSError) and st["error"] == "the wait failed" and st["done"] is True and st["alive"] is False
    assert st["messages"] == 5 and st["replayed"] == 3 and st["enqueued"] == 3
    assert any("NOT replayed whole" in r.getMessage() for r in caplog.records)

    # a whole replay: no error, every message replayed
    whole = tmp_path / "whole"
    whole.mkdir()
    clock = SteppingTime(step=1e-6)
    _use_clock(monkeypatch, clock, "rtsm.io.replayer")
    ok = _replay(_recording(whole, 4, gap_s=0.010), 1.0, [])
    st = ok.stats()
    assert ok.error is None and st["error"] is None and st["done"] is True
    assert st["messages"] == 4 and st["replayed"] == 4 and st["enqueued"] == 4 and st["path"].endswith("rec")


# ───────────────────────────── the bag source ─────────────────────────────

def test_a_paced_bag_survives_a_deadline_that_passes_between_two_clock_readings(tmp_path, monkeypatch):
    pytest.importorskip("rosbags", reason="the [eval] extra is not installed")
    from rtsm.io.contracts import SourceContext
    from rtsm.io.sources import make_source
    from test_bag_reader import simple_bag_messages, write_bag

    bag = write_bag(tmp_path / "b", simple_bag_messages(6))                       # stamps 100 ms apart
    # many steps and speeds: for a wait that reads the clock twice per slice, some of them put a deadline between the readings
    for step_s, speed in [(0.004, 4.5), (0.004, 3.0), (0.0031, 7.0), (0.0007, 20.0), (0.0123, 1.0), (0.0000009, 50.0)]:
        clock = SteppingTime(step=step_s)
        _use_clock(monkeypatch, clock, "rtsm.io.bag_source")
        q = IngestQueue(64)
        ctx = SourceContext(ingest_queue=q, clock_mode="sensor", keyframe_every_n=30, nonkf_min_interval_s=0.0,
                            require_tracking_normal=True, confidence_threshold=2)
        src = make_source("bag", {"io": {"bag": {"speed": speed}}}, ctx, path=str(bag))
        src.start()
        assert src.wait(30)
        st = src.stats()
        assert st["error"] is None and st["enqueued"] == 6 and q.qsize() == 6, (step_s, speed, st["error"])
        assert all(s >= 0.0 for s in clock.slept)
