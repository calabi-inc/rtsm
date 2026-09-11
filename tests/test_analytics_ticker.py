"""AnalyticsTicker — the ONE owner of the Tier-1 -> Tier-2 rollup (P1 task 5).

Before task 5 the rollup ran only inside the visualization push loop and only
while a browser client was attached, so a headless process never produced a
per-second bucket. These tests pin: buckets appear with no visualization
server at all; the WM snapshot and both buffers are driven by the ticker; the
count-based selection puts every appended frame in exactly one bucket
whatever clock stamped it; a late tick flags the bucket instead of discarding
its data; the ticker's health counters; and that the visualization server is a
pure consumer (it never rolls up and refuses buffers without a ticker).
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from rtsm.analytics import (
    AnalyticsBundle,
    AnalyticsTicker,
    FrameTimingStats,
    PipelineLatencyBuffer,
    SegAnalyticsBuffer,
    SegFrameStats,
    build_analytics,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeWM:
    def __init__(self, objects: int = 124, confirmed: int = 65):
        self.objects, self.confirmed, self.calls = objects, confirmed, 0

    def stats(self):
        self.calls += 1
        return {"objects": self.objects, "confirmed": self.confirmed}


class RaisingWM:
    def stats(self):
        raise RuntimeError("wm.stats() exploded")


def _frame(ts: float | None = None, **kw) -> FrameTimingStats:
    # The pipeline stamps with perf_counter, NOT the monotonic clock the cursor uses.
    return FrameTimingStats(timestamp=(time.perf_counter() if ts is None else ts), t_total=0.01, **kw)


def _seg(ts: float | None = None) -> SegFrameStats:
    return SegFrameStats(timestamp=(time.perf_counter() if ts is None else ts), n_total=2, n_dual=1)


def _buffers(max_frames: int = 300):
    return PipelineLatencyBuffer(max_frames=max_frames), SegAnalyticsBuffer(max_frames=max_frames)


def _sum(rows, key):
    return sum(int(b[key]) for b in rows)


def _wait_ticks(ticker: AnalyticsTicker, n: int, timeout: float = 5.0) -> None:
    """Wait until the free-running ticker has ticked n times (deadline-based,
    not a fixed sleep: Windows quantises Event.wait to ~15.6 ms)."""
    deadline = time.monotonic() + timeout
    while ticker.stats()["ticks"] < n:
        assert time.monotonic() < deadline, f"ticker reached only {ticker.stats()['ticks']} ticks in {timeout}s"
        time.sleep(0.01)


STATS_KEYS = {"interval_s", "ticks", "late_ticks", "stale_rollups", "ring_truncated", "last_tick_age_s", "stalled", "alive"}


# ── the headless thread ──────────────────────────────────────────────────────


def test_headless_thread_rolls_both_buffers_without_any_client():
    lat, seg = _buffers()
    wm = FakeWM()
    ticker = AnalyticsTicker(lat, seg, wm=wm, interval_s=0.02)
    assert ticker.daemon and ticker.name == "analytics-ticker"
    ticker.start()
    try:
        lat.append(_frame()); seg.append(_seg())              # so the receipts below cannot land in an empty bucket
        for _ in range(9):
            lat.record_frame_received()
        for _ in range(5):
            lat.append(_frame())
            seg.append(_seg())
            time.sleep(0.03)
        _wait_ticks(ticker, 8)
    finally:
        ticker.stop()
    assert not ticker.is_alive()

    lh, sh = lat.hourly_history(), seg.hourly_history()
    assert len(lh) >= 8 and len(sh) >= 8, (len(lh), len(sh))
    assert _sum(lh, "frames_in_bucket") == 6 and _sum(sh, "frames_in_bucket") == 6
    assert _sum(lh, "frames_received") == 9
    assert lat.aggregate()["input_hz"] > 0                     # was 0.0 forever in headless runs
    assert lh[-1]["wm_total"] == 124 and lh[-1]["wm_confirmed"] == 65 and lh[-1]["wm_proto"] == 59
    assert wm.calls >= 8
    st = ticker.stats()
    assert set(st) == STATS_KEYS
    assert st["ticks"] >= 8 and st["alive"] is False and st["interval_s"] == 0.02 and st["stalled"] in (False, True)
    assert len(lh) == st["ticks"] == len(sh)                   # one bucket per tick per buffer: no second owner


def test_wm_snapshot_reaches_the_bucket_and_none_wm_is_skipped():
    clk = FakeClock()
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg, wm=FakeWM(10, 4), now_fn=clk)
    t.arm(clk())
    rec = t.tick(now_mono=clk() + 1.0)
    assert (rec.latency.wm_total, rec.latency.wm_confirmed, rec.latency.wm_proto) == (10, 4, 6)
    assert rec.seg is not None and rec.tick == 1 and rec.late is False

    lat2, seg2 = _buffers()
    t2 = AnalyticsTicker(lat2, seg2, wm=None, now_fn=clk)
    t2.arm(clk())
    rec2 = t2.tick(now_mono=clk() + 1.0)
    assert rec2.latency.wm_total == 0 and rec2.tick == 1


def test_raising_wm_neither_stops_the_thread_nor_skips_the_rollup():
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg, wm=RaisingWM(), interval_s=0.02)
    t.start()
    try:
        lat.append(_frame())
        _wait_ticks(t, 3)
        assert t.is_alive()
        assert t.stats()["ticks"] >= 3
    finally:
        t.stop()
    assert _sum(lat.hourly_history(), "frames_in_bucket") == 1


# ── publication protocol ─────────────────────────────────────────────────────


def test_latest_since_ordering_and_history_bound():
    clk = FakeClock()
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg, now_fn=clk, history=4)
    t.arm(clk())
    assert t.latest() is None and t.since(None) == [] and t.since(0) == []
    for i in range(1, 7):
        t.tick(now_mono=clk() + i)
    assert t.latest().tick == 6
    assert [r.tick for r in t.since(2)] == [3, 4, 5, 6]        # bounded: 1 and 2 are gone
    assert [r.tick for r in t.since(None)] == [3, 4, 5, 6]
    assert [r.tick for r in t.since(5)] == [6]
    assert t.since(6) == []


# ── lateness, stale intervals, exactness ─────────────────────────────────────


def test_late_tick_flags_the_bucket_and_keeps_every_frame():
    clk = FakeClock()
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg, now_fn=clk)
    t.arm(clk())                                              # buffers re-anchored at 1000.0
    for _ in range(3):
        lat.append(_frame()); seg.append(_seg())
    r1 = t.tick(now_mono=1001.0)
    assert (r1.latency.frames_in_bucket, r1.latency.elapsed_s, r1.latency.stale_interval, r1.late) == (3, 1.0, False, False)
    assert (r1.seg.frames_in_bucket, r1.seg.elapsed_s, r1.seg.stale_interval) == (3, 1.0, False)   # seg re-anchored by arm() too
    for _ in range(2):
        lat.append(_frame()); seg.append(_seg())
    r2 = t.tick(now_mono=1003.5)                              # 2.5 s gap: late AND stale, nothing skipped
    assert r2.late is True
    assert (r2.latency.frames_in_bucket, r2.latency.elapsed_s, r2.latency.stale_interval) == (2, 2.5, True)
    assert (r2.seg.frames_in_bucket, r2.seg.stale_interval) == (2, True)
    st = t.stats(now_mono=1003.5)
    assert (st["ticks"], st["late_ticks"], st["stale_rollups"], st["ring_truncated"]) == (2, 1, 2, 0)   # both buffers flagged
    assert st["last_tick_age_s"] == 0.0
    assert _sum(lat.hourly_history(), "frames_in_bucket") == 5 == lat.aggregate()["counters"]["processed"]
    assert _sum(seg.hourly_history(), "frames_in_bucket") == 5


def test_arm_reanchors_time_only_so_every_count_sums_to_its_lifetime_counter():
    """Buffers are built before the model loads and the receiver runs before
    the ticker starts. start() must not make the first bucket span that
    startup (time cursor -> now), but every event recorded before start —
    frames appended, frames received, drops — must still land in the first
    bucket: for every counter, sum over buckets == the lifetime counter. The
    first gate run caught the alternative (drop cursors re-anchored): six
    throttle skips recorded before the ticker started were in no bucket."""
    clk = FakeClock()
    lat, seg = _buffers()
    lat._last_rollup_ts = seg._last_rollup_ts = 900.0         # built 100 s "ago"
    for _ in range(4):
        lat.record_frame_received()
    for _ in range(6):
        lat.record_throttle_skip()                            # recorded by the receiver before the owner started
    lat.record_gate_rejection()
    lat.append(_frame())                                      # appended before the owner started
    seg.append(_seg())
    t = AnalyticsTicker(lat, seg, now_fn=clk)
    t.arm(clk())
    r = t.tick(now_mono=1001.0)
    assert r.latency.stale_interval is False and r.latency.elapsed_s == 1.0
    assert (r.seg.stale_interval, r.seg.elapsed_s, r.seg.frames_in_bucket) == (False, 1.0, 1)
    assert (r.latency.frames_in_bucket, r.latency.frames_received, r.latency.throttle_skips, r.latency.gate_rejections) == (1, 4, 6, 1)
    assert r.latency.input_hz == 4.0                          # the first bucket absorbs the pre-start receipts
    c = lat.aggregate()["counters"]
    hl = lat.hourly_history()
    assert (_sum(hl, "frames_in_bucket"), _sum(hl, "frames_received"), _sum(hl, "throttle_skips"), _sum(hl, "gate_rejections")) == \
        (c["processed"], c["received"], c["throttle_skips"], c["gate_rejections"])


def test_count_based_selection_is_clock_domain_free():
    """The pipeline stamps FrameTimingStats.timestamp with perf_counter and the
    rollup cursor is monotonic; the old `timestamp > cursor` filter compared
    two clocks and lost a frame whose step straddled a tick. Stamps that are
    far behind AND far ahead of the cursor now land in exactly one bucket."""
    lat = PipelineLatencyBuffer()
    lat.reset_rollup_clock(now_mono=100.0)
    for _ in range(3):
        lat.append(_frame(ts=0.5))                            # a clock 100 s behind the cursor
    b1 = lat.roll_up_second(now_mono=101.0)
    assert b1.frames_in_bucket == 3 and b1.processing_hz == 3.0
    for _ in range(2):
        lat.append(_frame(ts=1e9))                            # a clock far ahead
    b2 = lat.roll_up_second(now_mono=102.0)
    b3 = lat.roll_up_second(now_mono=103.0)
    assert (b2.frames_in_bucket, b3.frames_in_bucket) == (2, 0)
    assert _sum(lat.hourly_history(), "frames_in_bucket") == 5

    seg = SegAnalyticsBuffer()
    seg.reset_rollup_clock(now_mono=100.0)
    seg.append(_seg(ts=0.5)); seg.append(_seg(ts=1e9))
    assert seg.roll_up_second(now_mono=101.0).frames_in_bucket == 2
    assert seg.roll_up_second(now_mono=102.0).frames_in_bucket == 0


def test_reset_then_rollup_counts_only_the_new_frames():
    for buf, mk in ((PipelineLatencyBuffer(), _frame), (SegAnalyticsBuffer(), _seg)):
        buf.reset_rollup_clock(now_mono=10.0)
        for _ in range(3):
            buf.append(mk())
        assert buf.roll_up_second(now_mono=11.0).frames_in_bucket == 3
        buf.clear()                                           # POST /reset
        for _ in range(2):
            buf.append(mk())
        assert buf.roll_up_second(now_mono=12.0).frames_in_bucket == 2, type(buf).__name__


def test_ring_truncation_keeps_the_count_exact_and_is_counted():
    lat = PipelineLatencyBuffer(max_frames=5)
    lat.reset_rollup_clock(now_mono=10.0)
    for _ in range(12):
        lat.append(_frame())
    b = lat.roll_up_second(now_mono=12.0)
    assert b.frames_in_bucket == 12 and b.processing_hz == 6.0
    assert b.t_total_ms == pytest.approx(10.0, abs=0.5)      # means over the ring's tail
    assert lat.rollup_stats(now_mono=12.0) == {"rollups": 1, "stale_rollups": 0, "ring_truncated": 7, "last_rollup_age_s": 0.0}


def test_rollup_stats_survive_clear_and_are_reported_by_the_ticker():
    lat, seg = _buffers()
    lat.reset_rollup_clock(now_mono=0.0)
    lat.roll_up_second(now_mono=5.0)                          # stale
    lat.clear()
    assert lat.rollup_stats(now_mono=5.0)["stale_rollups"] == 1 and lat.rollup_stats()["rollups"] == 1
    t = AnalyticsTicker(lat, seg, now_fn=FakeClock(5.0))
    assert t.stats()["stale_rollups"] == 1 and t.stats()["ticks"] == 0 and t.stats()["last_tick_age_s"] is None


def test_stale_threshold_follows_the_interval():
    lat, seg = _buffers()
    assert AnalyticsTicker(lat, seg).stale_after_s == 2.0
    assert AnalyticsTicker(lat, seg, interval_s=0.02).stale_after_s == 2.0      # never below 2 s
    assert AnalyticsTicker(lat, seg, interval_s=3.0).stale_after_s == 6.0


# ── construction / lifecycle ─────────────────────────────────────────────────


def test_ticker_requires_a_buffer_and_a_positive_interval():
    lat, seg = _buffers()
    with pytest.raises(ValueError):
        AnalyticsTicker(None, None)
    with pytest.raises(ValueError):
        AnalyticsTicker(lat, seg, interval_s=0)
    t = AnalyticsTicker(lat, None)
    t.stop()                                                  # never started: a no-op
    assert not t.is_alive() and not t.armed


def test_build_analytics_disabled_and_enabled():
    off = build_analytics({"analytics": {"enable": False}})
    assert isinstance(off, AnalyticsBundle)
    assert (off.seg, off.latency, off.ticker, off.enabled) == (None, None, None, False)
    off.start(); off.stop()                                   # no-ops

    wm = FakeWM()
    on = build_analytics({"analytics": {"retention_s": 10, "buffer_frames": 7}}, wm=wm)
    assert on.enabled and (on.retention_s, on.buffer_frames) == (10.0, 7)
    assert on.latency._buffer.maxlen == 7 and on.seg._buffer.maxlen == 7 and on.latency._retention_s == 10.0
    assert on.ticker.interval_s == 1.0 and not on.ticker.is_alive()
    on.start()
    try:
        assert on.ticker.is_alive() and on.ticker.armed
        on.start()                                            # idempotent: no RuntimeError("threads can only be started once")
    finally:
        on.stop()
    assert not on.ticker.is_alive()

    dflt = build_analytics({})
    assert dflt.enabled and (dflt.retention_s, dflt.buffer_frames) == (3600.0, 300)
    assert build_analytics({"analytics": "yes"}).enabled     # a scalar block falls back to the defaults


# ── the visualization server is a consumer ──────────────────────────────────


def _viz(lat, seg, ticker):
    from rtsm.visualization.server import VisualizationServer
    return VisualizationServer(cfg={}, working_memory=None, seg_analytics=seg, latency_analytics=lat,
                               analytics_ticker=ticker)


class _NoRollup(PipelineLatencyBuffer):
    def roll_up_second(self, *a, **k):
        raise AssertionError("the visualization server must never roll up")

    def snapshot_wm(self, *a, **k):
        raise AssertionError("the visualization server must never snapshot the WM")


class _NoSegRollup(SegAnalyticsBuffer):
    def roll_up_second(self, *a, **k):
        raise AssertionError("the visualization server must never roll up")


def test_viz_server_refuses_buffers_without_a_ticker():
    from rtsm.visualization.server import VisualizationServer
    lat, seg = _buffers()
    with pytest.raises(ValueError, match="analytics_ticker"):
        VisualizationServer(cfg={}, working_memory=None, latency_analytics=lat)
    with pytest.raises(ValueError, match="analytics_ticker"):
        VisualizationServer(cfg={}, working_memory=None, seg_analytics=seg)
    VisualizationServer(cfg={}, working_memory=None)          # no buffers: fine (the config-echo test's shape)


def _code_names(code) -> set:
    names = set(code.co_names)
    for c in code.co_consts:
        if hasattr(c, "co_names"):
            names |= _code_names(c)
    return names


def test_viz_server_never_rolls_up_static_pin():
    """No method or coroutine of the class — not only the two documented
    anchors — may name the rollup or the WM snapshot: with the headless gate
    running no viz server at all, this pin is the guard against a second
    owner in viz-on runs."""
    import inspect
    from rtsm.visualization.server import VisualizationServer
    seen = []
    for name, member in vars(VisualizationServer).items():
        fn = getattr(member, "__func__", member)
        if inspect.isfunction(fn):
            seen.append(name)
            names = _code_names(fn.__code__)
            assert "roll_up_second" not in names and "snapshot_wm" not in names, name
    assert {"_push_analytics_loop", "_analytics_messages", "start_tasks", "__init__"} <= set(seen)


def test_analytics_messages_builder_consumes_tick_records_only():
    lat, seg = _NoRollup(), _NoSegRollup()
    real_lat, real_seg = _buffers()
    clk = FakeClock()
    ticker = AnalyticsTicker(real_lat, real_seg, now_fn=clk)   # rolls the REAL buffers; the viz holds the raising ones
    ticker.arm(clk())
    recs = [ticker.tick(now_mono=clk() + i) for i in (1, 2)]
    vs = _viz(lat, seg, ticker)

    full = vs._analytics_messages(True, [])
    assert len(full) == 1 and full[0]["mode"] == "full" and full[0]["type"] == "runtime_analytics"
    assert "hourly" in full[0]["latency"] and "hourly" in full[0]["segmentation"] and "config" in full[0]

    app = vs._analytics_messages(False, recs)
    assert [m["mode"] for m in app] == ["append", "append"]
    assert all(isinstance(m["latency"]["bucket"], dict) and isinstance(m["segmentation"]["bucket"], dict) for m in app)
    lb, sb = app[0]["latency"]["bucket"], app[0]["segmentation"]["bucket"]
    assert lb["frames_in_bucket"] == 0 and {"frames_received", "elapsed_s", "stale_interval"} <= set(lb)
    assert {"elapsed_s", "stale_interval"} <= set(sb) and (lb["elapsed_s"], lb["stale_interval"]) == (1.0, False)
    assert app[1]["segmentation"]["backend"] == "fastsam"

    idle = vs._analytics_messages(False, [])
    assert len(idle) == 1 and idle[0]["mode"] == "append"
    assert idle[0]["latency"]["bucket"] is None and idle[0]["segmentation"]["bucket"] is None
    assert "aggregate" in idle[0]["latency"]


class _Broadcaster:
    def __init__(self):
        self.client_count = 0
        self.sent = []

    async def _broadcast_json(self, msg):
        self.sent.append(msg)


def test_push_loop_cursor_protocol_full_on_attach_then_one_append_per_bucket():
    """Nobody listening: the cursor follows the ticker (no backlog dump on
    attach). First push with a client: a full sync. Then exactly one append
    per new tick, aggregate-only appends in between, no duplicates."""
    clk = FakeClock()
    lat, seg = _buffers()
    ticker = AnalyticsTicker(lat, seg, now_fn=clk)
    ticker.arm(clk())
    vs = _viz(lat, seg, ticker)
    bc = _Broadcaster()
    vs.broadcaster = bc
    vs._analytics_push_s = 0.01
    vs._analytics_full_sync_interval = 1000

    async def drive():
        vs._running = True
        task = asyncio.create_task(vs._push_analytics_loop())
        for i in (1, 2, 3):
            ticker.tick(now_mono=clk() + i)                  # ticks while nobody listens
        await asyncio.sleep(0.12)
        assert bc.sent == []                                  # no client: nothing pushed
        bc.client_count = 1
        await asyncio.sleep(0.12)
        assert bc.sent and bc.sent[0]["mode"] == "full"       # first push after attach
        n_after_full = len(bc.sent)
        assert all(m["mode"] == "append" and m["latency"]["bucket"] is None for m in bc.sent[1:n_after_full])
        ticker.tick(now_mono=clk() + 4)
        ticker.tick(now_mono=clk() + 5)
        await asyncio.sleep(0.12)
        vs._running = False
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass    # the loop ends by cancellation; anything else it raised must surface

    # A private loop, not asyncio.run(): run() leaves the main thread with NO
    # current event loop afterwards, which breaks later tests in the same
    # session that still rely on asyncio.get_event_loop() (tests/test_mcp_server.py).
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(drive())
    finally:
        loop.close()
    modes = [m["mode"] for m in bc.sent]
    assert modes[0] == "full" and modes.count("full") == 1
    with_bucket = [m for m in bc.sent if m["mode"] == "append" and m["latency"]["bucket"] is not None]
    assert len(with_bucket) == 2, [m["latency"]["bucket"]["elapsed_s"] if m["latency"]["bucket"] else None for m in bc.sent]
    # ticks 1-3 were never pushed as appends (the full sync carried them in `hourly`)
    assert len(bc.sent[0]["latency"]["hourly"]) == 3


def _run_loop(vs, steps):
    """Drive _push_analytics_loop on a private loop; `steps` is an async callable."""
    async def drive():
        vs._running = True
        task = asyncio.create_task(vs._push_analytics_loop())
        try:
            await steps()
        finally:
            vs._running = False
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass    # the loop ends by cancellation; anything else it raised must surface
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(drive())
    finally:
        loop.close()


def _no_duplicate_buckets(sent):
    """Protocol invariant: an appended bucket never repeats one the preceding
    full sync already carried, and each full's history covers the previous."""
    covered: set = set()
    for m in sent:
        if m["mode"] == "full":
            hist = {b["wall_ts"] for b in m["latency"]["hourly"]}
            assert covered <= hist, "a full sync lost history"
            covered = hist
        elif m["latency"]["bucket"] is not None:
            ts = m["latency"]["bucket"]["wall_ts"]
            assert ts not in covered, "bucket appended twice"
            covered.add(ts)


def test_push_loop_full_on_every_client_increase_not_on_decrease():
    clk = FakeClock()
    lat, seg = _buffers()
    ticker = AnalyticsTicker(lat, seg, now_fn=clk)
    ticker.arm(clk())
    vs = _viz(lat, seg, ticker)
    bc = _Broadcaster()
    vs.broadcaster = bc
    vs._analytics_push_s = 0.01
    vs._analytics_full_sync_interval = 1000
    marks = []

    async def steps():
        ticker.tick(now_mono=clk() + 1)
        bc.client_count = 1
        await asyncio.sleep(0.08)
        marks.append(len(bc.sent))                            # [0]: after the first client
        bc.client_count = 2                                   # a second browser tab attaches
        await asyncio.sleep(0.08)
        marks.append(len(bc.sent))
        bc.client_count = 1                                   # one leaves: no full
        await asyncio.sleep(0.08)
        marks.append(len(bc.sent))
        bc.client_count = 2                                   # attaches again: full
        await asyncio.sleep(0.08)

    _run_loop(vs, steps)
    modes = [m["mode"] for m in bc.sent]
    assert modes[0] == "full"
    assert modes[marks[0]] == "full", modes                   # first push after 1 -> 2
    assert "full" not in modes[marks[1]:marks[2]], modes      # nothing on 2 -> 1
    assert modes[marks[2]] == "full", modes                   # first push after 1 -> 2 again
    assert modes.count("full") == 3
    _no_duplicate_buckets(bc.sent)


def test_push_loop_periodic_full_and_gap_resync():
    """A periodic full every N pushes, and a full (never a hole) when the
    ticker's retained history no longer covers the cursor."""
    clk = FakeClock()
    lat, seg = _buffers()
    ticker = AnalyticsTicker(lat, seg, now_fn=clk, history=4)
    ticker.arm(clk())
    vs = _viz(lat, seg, ticker)
    bc = _Broadcaster()
    vs.broadcaster = bc
    vs._analytics_push_s = 0.01
    vs._analytics_full_sync_interval = 3
    n = [0]

    def tick():
        n[0] += 1
        ticker.tick(now_mono=clk() + n[0])

    marks = []

    async def steps():
        bc.client_count = 1
        await asyncio.sleep(0.08)                             # full, then periodic fulls every 3 pushes
        marks.append(len(bc.sent))
        tick(); tick()
        await asyncio.sleep(0.08)                             # two appends with buckets (or covered by a periodic full)
        marks.append(len(bc.sent))
        for _ in range(6):                                    # more ticks than the history keeps -> gap
            tick()
        await asyncio.sleep(0.08)

    _run_loop(vs, steps)
    modes = [m["mode"] for m in bc.sent]
    assert modes[0] == "full" and modes.count("full") >= 3, modes
    after_gap = bc.sent[marks[1]:]
    assert after_gap and after_gap[0]["mode"] == "full", [m["mode"] for m in after_gap]   # gap -> full, not 4 appends
    assert len(after_gap[0]["latency"]["hourly"]) == 8
    _no_duplicate_buckets(bc.sent)


# ── code-review additions: stalled, cadence, lifecycle, bounds ──────────────


def test_stalled_is_the_read_time_view_of_a_ticker_that_stopped_ticking():
    """late_ticks only moves when a tick eventually happens; a ticker wedged in
    wm.stats() reads late_ticks 0 / alive True forever — `stalled` is what a
    gate or G1-C must read."""
    clk = FakeClock()
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg, now_fn=clk)
    assert t.stats(now_mono=clk() + 100)["stalled"] is False          # not armed: nothing to be late for
    t.arm(clk())
    assert t.stats(now_mono=clk() + 1.5)["stalled"] is False
    assert t.stats(now_mono=clk() + 3.0)["stalled"] is True           # never ticked, 3 s after arm
    r = t.tick(now_mono=clk() + 3.0)
    assert r.late is True
    st = t.stats(now_mono=clk() + 3.5)
    assert (st["stalled"], st["late_ticks"], st["last_tick_age_s"]) == (False, 1, 0.5)
    assert t.stats(now_mono=clk() + 5.5)["stalled"] is True

    gate = threading.Event()

    class BlockedWM:
        def stats(self):
            gate.wait()
            return {"objects": 0, "confirmed": 0}

    lat2, seg2 = _buffers()
    t2 = AnalyticsTicker(lat2, seg2, wm=BlockedWM(), interval_s=0.02)
    t2.start()
    try:
        time.sleep(0.15)
        st = t2.stats()
        assert (st["ticks"], st["late_ticks"], st["alive"], st["stalled"], st["last_tick_age_s"]) == (0, 0, True, True, None)
        gate.set()
        _wait_ticks(t2, 1)
    finally:
        gate.set()
        t2.stop()


def test_next_deadline_cadence_rule():
    nd = AnalyticsTicker.next_deadline
    assert nd(10.0, 10.0, 1.0, False) == 11.0                         # on time: the schedule advances, no drift
    assert nd(10.0, 10.3, 1.0, False) == 11.0                         # small jitter: still the schedule
    assert nd(10.0, 10.99, 1.0, False) == pytest.approx(11.49)        # 1x-2x wake delay: never a ~10 ms bucket
    assert nd(10.0, 12.5, 1.0, True) == 13.5                          # late: restart from now, no catch-up burst


def test_start_after_stop_raises_and_the_bundle_says_so(caplog):
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg)
    t.stop()
    with pytest.raises(RuntimeError, match="stop"):
        t.start()
    b = build_analytics({})
    b.start(); b.stop()
    assert not b.ticker.is_alive()
    import logging
    with caplog.at_level(logging.WARNING, logger="rtsm.analytics.ticker"):
        b.start()                                                     # cannot restart a thread: logged, not silent
    assert any("stopped" in rec.getMessage() for rec in caplog.records)
    assert not b.ticker.is_alive()


def test_default_history_bound_is_64():
    clk = FakeClock()
    lat, seg = _buffers()
    t = AnalyticsTicker(lat, seg, now_fn=clk)
    t.arm(clk())
    for i in range(1, 71):
        t.tick(now_mono=clk() + i)
    recs = t.since(None)
    assert len(recs) == 64 and recs[0].tick == 7 and recs[-1].tick == 70


def test_seg_ring_truncation_keeps_the_count_exact_and_is_counted():
    seg = SegAnalyticsBuffer(max_frames=5)
    seg.reset_rollup_clock(now_mono=10.0)
    for _ in range(12):
        seg.append(_seg())
    b = seg.roll_up_second(now_mono=12.0)
    assert (b.frames_in_bucket, b.mean_total, b.dual_rate) == (12, 2.0, 0.5)   # means over the ring's tail
    assert seg.rollup_stats(now_mono=12.0) == {"rollups": 1, "stale_rollups": 0, "ring_truncated": 7, "last_rollup_age_s": 0.0}


def test_effective_ratio_is_zero_without_input_this_interval():
    lat = PipelineLatencyBuffer()
    lat.reset_rollup_clock(now_mono=10.0)
    lat.append(_frame()); lat.append(_frame())                        # the lossless drain: frames, no receipts
    b = lat.roll_up_second(now_mono=11.0)
    assert (b.frames_in_bucket, b.frames_received, b.processing_hz, b.input_hz, b.effective_ratio) == (2, 0, 2.0, 0.0, 0.0)
    lat.record_frame_received(); lat.append(_frame())
    assert lat.roll_up_second(now_mono=12.0).effective_ratio == 1.0


def test_clear_keeps_the_time_cursor_on_the_tickers_clock():
    """POST /reset resets the counters and count cursors; the time cursor
    stays tick-to-tick, so the bucket after a reset covers a real interval."""
    for buf, mk in ((PipelineLatencyBuffer(), _frame), (SegAnalyticsBuffer(), _seg)):
        buf.reset_rollup_clock(now_mono=10.0)
        buf.append(mk())
        assert buf.roll_up_second(now_mono=11.0).elapsed_s == 1.0
        buf.clear()                                                   # at some wall time unrelated to the ticker's clock
        buf.append(mk())
        b = buf.roll_up_second(now_mono=12.0)
        assert (b.elapsed_s, b.frames_in_bucket, b.stale_interval) == (1.0, 1, False), type(buf).__name__
