"""/healthz.ingest and /stats/analytics.rollup (P1 task 5).

`/healthz` gains an `ingest` block sourced from the ingest lane object itself
(IngestLanes.stats(), the same snapshot as /stats.ingest_lanes), present in
every runner and mode — live, replay, eval, `rtsm demo` — with no flag, and
it never folds into `status`. `/stats/analytics` gains `rollup`, the analytics
ticker's own health, `null` when no ticker is wired.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from rtsm.analytics import build_analytics
from rtsm.api.server import create_app
from rtsm.core.datamodel import FramePacket, IngestMeta, TimeBundle
from rtsm.io.ingest_lanes import IngestLanes
from rtsm.io.ingest_queue import IngestQueue


class _StubWM:
    def stats(self):
        return {"objects": 0, "confirmed": 0, "upserts_total": 0}


def _client(**kw) -> TestClient:
    return TestClient(create_app(working_memory=_StubWM(), registry=CollectorRegistry(), **kw))


def _pkt(seq: int, kf: bool = False) -> FramePacket:
    return FramePacket(
        time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=seq * 1000, seq=seq),
        rgb=np.zeros((2, 2, 3), dtype=np.uint8), depth_m=None, pose=None, intr=None,
        is_keyframe=kf, ingest=IngestMeta(rx_seq=seq),
    )


# ── /healthz.ingest ──────────────────────────────────────────────────────────


def test_healthz_without_provider_is_exactly_status_ok():
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_healthz_ingest_is_the_lossless_lane_snapshot_and_lane_full_at_capacity():
    lanes = IngestLanes("lossless", lossless_depth=2)
    c = _client(ingest_provider=lanes.stats)
    body = c.get("/healthz").json()
    ing = body["ingest"]
    assert (ing["policy"], ing["maxsize"], ing["depth"], ing["lane_full"], ing["closed"]) == ("lossless", 2, {"fifo": 0}, False, False)
    for k in ("admitted_kf", "admitted_nonkf", "nonkf_superseded", "kf_dropped", "kf_lane_full",
              "age_dropped", "blocked_puts", "closed_puts", "max_depth_seen", "blocked_s", "lossless_depth"):
        assert k in ing, k
    assert lanes.put(_pkt(1, kf=True)) and lanes.put(_pkt(2))
    body = c.get("/healthz").json()
    ing = body["ingest"]
    assert ing["lane_full"] is True and ing["depth"] == {"fifo": 2}
    assert ing["admitted_kf"] + ing["admitted_nonkf"] == 2
    assert ing == lanes.stats()                                  # verbatim: the same dict /stats.ingest_lanes serves
    # never folds into status, even at capacity
    assert body["status"] == "ok" and "reasons" not in body and "frame_flow" not in body


def test_healthz_ingest_latest_policy_lane_full_means_the_keyframe_lane():
    lanes = IngestLanes("latest", keyframe_lane_depth=2)
    c = _client(ingest_provider=lanes.stats)
    ing = c.get("/healthz").json()["ingest"]
    assert ing["policy"] == "latest" and ing["lane_full"] is False and ing["depth"] == {"keyframe": 0, "latest": 0}
    assert ing["keyframe_lane_depth"] == 2 and "max_frame_age_s" in ing
    lanes.put(_pkt(1))                                           # the slot alone never makes the lane full
    assert c.get("/healthz").json()["ingest"]["lane_full"] is False
    lanes.put(_pkt(2, kf=True)); lanes.put(_pkt(3, kf=True))
    body = c.get("/healthz").json()
    assert body["ingest"]["lane_full"] is True and body["ingest"]["depth"] == {"keyframe": 2, "latest": 1}
    assert body["status"] == "ok"


def test_healthz_ingest_legacy_queue_carries_lane_full_too():
    q = IngestQueue(maxsize=1)
    c = _client(ingest_provider=q.stats)
    ing = c.get("/healthz").json()["ingest"]
    assert (ing["policy"], ing["lane_full"], ing["depth"]) == ("legacy", False, {"legacy": 0})
    assert q.put(_pkt(1))
    assert c.get("/healthz").json()["ingest"]["lane_full"] is True


def test_healthz_ingest_error_shape_when_the_provider_raises():
    def boom():
        raise RuntimeError("lane gone")
    body = _client(ingest_provider=boom).get("/healthz").json()
    assert body["ingest"] == {"error": "unavailable"} and body["status"] == "ok" and "reasons" not in body


def test_healthz_ingest_sits_next_to_frame_flow_not_inside_it():
    lanes = IngestLanes("lossless", lossless_depth=4)
    ff = {"state": "backlogged", "degraded": True, "reasons": ["ingest lane full for 3 consecutive polls"],
          "backlog": {"lane_full": True, "age_dropped": 0, "depth": {"fifo": 4}}}
    body = _client(frame_flow_provider=lambda: ff, ingest_provider=lanes.stats).get("/healthz").json()
    assert body["status"] == "degraded" and body["reasons"] == ff["reasons"]     # the watchdog folds, as before
    assert body["frame_flow"] == ff and body["ingest"]["lane_full"] is False      # two sources, two read times


# ── /stats.ingest_lanes and /stats/analytics.rollup ─────────────────────────


def test_stats_ingest_lanes_is_the_same_snapshot():
    lanes = IngestLanes("lossless", lossless_depth=3)
    lanes.put(_pkt(1))
    c = _client(extra_stats_provider=lambda: {"ingest_lanes": lanes.stats()}, ingest_provider=lanes.stats)
    stats, health = c.get("/stats").json(), c.get("/healthz").json()
    assert stats["ingest_lanes"] == health["ingest"] and stats["ingest_lanes"]["lane_full"] is False


def test_stats_analytics_rollup_is_the_ticker_health_or_null():
    b = build_analytics({})
    c = _client(seg_analytics=b.seg, latency_analytics=b.latency, analytics_ticker=b.ticker)
    body = c.get("/stats/analytics").json()
    assert body["latency"]["hourly"] == [] and body["segmentation"]["hourly"] == []
    roll = body["rollup"]
    assert roll["ticks"] == 0 and roll["late_ticks"] == 0 and roll["stale_rollups"] == 0 and roll["alive"] is False
    assert roll["interval_s"] == 1.0 and roll["last_tick_age_s"] is None and roll["ring_truncated"] == 0

    b.ticker.arm(100.0)
    b.ticker.tick(now_mono=101.0)                                # one headless rollup, no viz anywhere
    body = c.get("/stats/analytics").json()
    assert len(body["latency"]["hourly"]) == 1 and len(body["segmentation"]["hourly"]) == 1
    assert body["latency"]["hourly"][0]["elapsed_s"] == 1.0 and body["latency"]["hourly"][0]["stale_interval"] is False
    assert body["rollup"]["ticks"] == 1

    no_ticker = _client(seg_analytics=b.seg, latency_analytics=b.latency)
    assert no_ticker.get("/stats/analytics").json()["rollup"] is None


def test_stats_analytics_still_503_without_buffers():
    assert _client().get("/stats/analytics").status_code == 503
