"""IngestGate non-keyframe grace (P1 task 6): off by default (0), guarded so 0 is
truly off even when the ingest clock steps back (the post-/reset shape), and
still a working opt-in above 0."""
from __future__ import annotations

import numpy as np

from rtsm.core.ingest_gate import IngestGate
from rtsm.stores.sweep_cache import SweepCache


def _gate(grace=None):
    cfg = {"ingest": {"dup_window_ns": 0}}            # dup window off: isolate the grace
    if grace is not None:
        cfg["ingest"]["non_kf_grace_s"] = grace
    return IngestGate(cfg)


def _decide(g, *, is_keyframe, ts_ns, now):
    return g.should_accept(is_keyframe=is_keyframe, ts_ns=ts_ns, sweep_cache=SweepCache(), cell=(0, 0, 0),
                           vbin=(0, 0), cam_pos=np.zeros(3), fwd_unit=np.array([0.0, 0.0, 1.0]), Z=None,
                           look_cell=None, now_mono=now)


def test_grace_is_off_by_default_even_when_the_clock_steps_back():
    g = _gate()
    assert g.non_kf_grace_s == 0.0
    assert _decide(g, is_keyframe=True, ts_ns=10_000_000_000, now=10.0).reason == "keyframe"
    # now < the keyframe's arrival: a bare `dt < 0.0` would have fired here
    assert _decide(g, is_keyframe=False, ts_ns=11_000_000_000, now=9.0).reason != "non_kf_grace"
    assert _decide(g, is_keyframe=False, ts_ns=11_000_000_000, now=10.001).reason != "non_kf_grace"


def test_grace_above_zero_still_defers_a_non_keyframe():
    g = _gate(0.5)
    _decide(g, is_keyframe=True, ts_ns=10_000_000_000, now=10.0)
    assert _decide(g, is_keyframe=False, ts_ns=11_000_000_000, now=10.2).reason == "non_kf_grace"
    assert _decide(g, is_keyframe=False, ts_ns=12_000_000_000, now=10.6).reason != "non_kf_grace"
