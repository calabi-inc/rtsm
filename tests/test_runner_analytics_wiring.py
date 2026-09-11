"""Both runner entry points wire the analytics ticker (P1 task 5) — bytecode pins.

Task 4's lesson: an entry point regressed for two PRs because no CPU test
executed its startup block. The full startup of `rtsm.run.main` needs models
(task 6 adds that test); until then these pins read the compiled function
bodies of `rtsm.demo.run_demo` and `rtsm.run.main` and fail if either stops
building the analytics through `build_analytics`, stops passing the ticker to
BOTH `create_app` and `VisualizationServer`, starts it before the working
memory exists, or starts it after `pipe.run_forever()`.
"""
from __future__ import annotations

import dis
import importlib

import pytest


def _codes(code):
    yield code
    for c in code.co_consts:
        if hasattr(c, "co_code"):
            yield from _codes(c)


def _all_names(code):
    out = set()
    for c in _codes(code):
        out |= set(c.co_names)
    return out


def _kw_tuples(code):
    return [c for cc in _codes(code) for c in cc.co_consts
            if isinstance(c, tuple) and c and all(isinstance(x, str) for x in c)]


def _attr_calls_on(code, var: str, attr: str):
    """Offsets of `<var>.<attr>` loads in the top-level function body
    (EXTENDED_ARG prefixes, present once a body has > 256 names, are skipped)."""
    ins = [x for x in dis.get_instructions(code) if x.opname != "EXTENDED_ARG"]
    out = []
    for prev, cur in zip(ins, ins[1:]):
        if cur.opname == "LOAD_ATTR" and cur.argval == attr and prev.opname in ("LOAD_FAST", "LOAD_DEREF", "LOAD_FAST_CHECK") \
                and prev.argval == var:
            out.append(cur.offset)
    return out


def _offsets(code, opname_prefix: str, argval: str):
    return [i.offset for i in dis.get_instructions(code)
            if i.opname.startswith(opname_prefix) and i.argval == argval]


def _first(code, opname_prefix: str, argval: str):
    offs = _offsets(code, opname_prefix, argval)
    return offs[0] if offs else None


@pytest.mark.parametrize("modname,fn", [("rtsm.demo", "run_demo"), ("rtsm.run", "main")])
def test_runner_builds_wires_starts_and_stops_the_ticker(modname, fn):
    code = getattr(importlib.import_module(modname), fn).__code__
    names = _all_names(code)
    assert "build_analytics" in names, f"{modname}.{fn} must build analytics through build_analytics"
    for old in ("SegAnalyticsBuffer", "PipelineLatencyBuffer"):
        assert old not in names, f"{modname}.{fn} constructs {old} directly: a second rollup owner or none"

    kw = _kw_tuples(code)
    assert sum(1 for t in kw if "analytics_ticker" in t) >= 2, \
        f"{modname}.{fn} must pass analytics_ticker to both create_app and VisualizationServer: {kw}"
    assert any("ingest_provider" in t for t in kw), f"{modname}.{fn} must wire /healthz.ingest (ingest_provider)"

    starts = _attr_calls_on(code, "analytics", "start")
    stops = _attr_calls_on(code, "analytics", "stop")
    assert starts and stops, (starts, stops)

    # Order: wm exists before build_analytics (the ticker snapshots wm.stats()),
    # and the ticker starts before the PIPELINE loop, i.e. after every load.
    # (run.py also loads `sub.run_forever` as the ZeroMQ subscriber's thread
    # target, earlier in the body — hence the pin on `pipe.run_forever`.)
    wm_store = _first(code, "STORE_", "wm")
    build = _first(code, "LOAD_", "build_analytics")
    assert wm_store is not None and build is not None
    assert wm_store < build, "build_analytics must run after WorkingMemory is constructed"
    pipe_loops = _attr_calls_on(code, "pipe", "run_forever")
    assert pipe_loops, "no pipe.run_forever() found"
    assert all(starts[0] < o for o in pipe_loops), "analytics.start() must precede pipe.run_forever()"
    # ...and AFTER the API server is up (i.e. after every model / index load),
    # so late_ticks / stale_rollups describe the run, not startup GIL holds.
    srv = _first(code, "LOAD_", "start_server")
    assert srv is not None and starts[0] > srv, "analytics.start() must follow start_server()"


# ---------------- P1 task 6: receiver timing comes from lane_cfg, never from io.websocket ----------------


@pytest.mark.parametrize("modname,fn,sites,zmq", [("rtsm.demo", "run_demo", 1, 0), ("rtsm.run", "main", 3, 1)])
def test_runner_receivers_read_timing_from_lane_cfg_not_ws_cfg(modname, fn, sites, zmq):
    """The silent-fallback trap: a leftover ws_cfg.get("keyframe_every_n", 30) would
    keep every default-valued run green. The receivers must take the validated
    LaneConfig values (run.py: record-only, replay, websocket; + ZeroMQ for the
    throttle), and the two key names must not be read through `.get()` anywhere
    in the runner body."""
    code = getattr(importlib.import_module(modname), fn).__code__
    # Exact counts: one read per receiver site (an extra read of the old block cannot hide behind a >=).
    assert len(_attr_calls_on(code, "lane_cfg", "keyframe_every_n")) == sites
    assert len(_attr_calls_on(code, "lane_cfg", "nonkf_min_interval_s")) == sites + zmq
    if zmq:
        assert _attr_calls_on(code, "lane_cfg", "pair_window_s") and _attr_calls_on(code, "lane_cfg", "pair_window_frames")
    # The key names may legitimately remain as constants: a receiver call with
    # many kwargs is compiled through CALL_FUNCTION_EX with a dict of names.
    # What must be gone is reading them through a mapping: `.get("<key>", ...)`.
    keys = ("keyframe_every_n", "nonkf_min_interval_s")
    for cc in _codes(code):
        ins = [x for x in dis.get_instructions(cc) if x.opname != "EXTENDED_ARG"]
        for i, x in enumerate(ins):
            if x.opname == "LOAD_CONST" and x.argval in keys:
                window = [y.argval for y in ins[max(0, i - 3):i] if y.opname.startswith("LOAD_ATTR")]
                assert "get" not in window, f"{modname}.{fn} still reads {x.argval!r} via .get() (offset {x.offset})"


def test_viz_echo_reads_the_ingest_block():
    from rtsm.visualization.server import VisualizationServer
    consts = [c for c in VisualizationServer._extract_analytics_config.__code__.co_consts if isinstance(c, str)]
    assert "ingest" in consts
