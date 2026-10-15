"""TfBuffer on a live stream (0.3.0 review, 2026-10-14): out-of-order samples go in place, and a horizon bounds
each hop's history. Without a horizon (the bag reader) nothing is dropped."""
from __future__ import annotations

from rtsm.io.tf_buffer import TfBuffer

Q = (0.0, 0.0, 0.0, 1.0)


def _s(t: float) -> int:
    return int(t * 1e9)


def test_out_of_order_sample_is_inserted_in_place():
    buf = TfBuffer(extrapolation_s=0.05)
    for t in (3.0, 1.0, 2.0):
        buf.add("odom", "base", _s(t), (t, 0.0, 0.0), Q)
    hop = buf._hops[("odom", "base")]
    assert hop.stamps == [_s(1.0), _s(2.0), _s(3.0)]
    assert [float(v[0]) for v in hop.ts] == [1.0, 2.0, 3.0]
    assert buf.n_timed == 3 and buf.n_dropped == 0


def test_horizon_bounds_a_live_hop_and_a_bag_keeps_everything():
    live = TfBuffer(extrapolation_s=0.05, horizon_s=1.0)
    for t in (0.0, 0.5, 0.9, 1.0, 2.5):
        live.add("odom", "base", _s(t), (t, 0.0, 0.0), Q)
    hop = live._hops[("odom", "base")]
    assert hop.stamps == [_s(2.5)] and live.n_dropped == 4           # cutoff 2.5 - 1.0 = 1.5 s: everything older went
    live.add("odom", "base", _s(2.0), (2.0, 0.0, 0.0), Q)           # a late sample inside the horizon is kept, in order
    assert hop.stamps == [_s(2.0), _s(2.5)]
    bag = TfBuffer(extrapolation_s=0.05)
    for t in (0.0, 0.5, 0.9, 1.0, 2.5):
        bag.add("odom", "base", _s(t), (t, 0.0, 0.0), Q)
    assert len(bag._hops[("odom", "base")].stamps) == 5 and bag.n_dropped == 0
