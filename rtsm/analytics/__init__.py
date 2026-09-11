"""
RTSM Runtime Analytics — two-tier buffer system for real-time insight.

Tier 1: Per-frame raw data in ring buffers (short window, ~75s)
Tier 2: Per-second aggregated buckets (1-hour retention, wall-clock eviction),
        rolled up by ONE owner per process — AnalyticsTicker (rtsm/analytics/ticker.py),
        a 1 Hz daemon thread that runs whenever ``analytics.enable`` is true,
        with or without a visualization client. The visualization server only
        consumes what the ticker publishes.

Usage (runners):
    from rtsm.analytics import build_analytics
    analytics = build_analytics(cfg, wm=wm)          # buffers + ticker (None when disabled)
    ...pass analytics.seg / analytics.latency to the receivers, the pipeline, the API,
       and analytics.ticker to the API + VisualizationServer...
    analytics.start()                                # right before pipe.run_forever()
    analytics.stop()                                 # in the runner's finally
"""
from rtsm.analytics.seg_analytics import SegAnalyticsBuffer, SegFrameStats, SegSecondBucket
from rtsm.analytics.latency_analytics import PipelineLatencyBuffer, FrameTimingStats, LatencySecondBucket
from rtsm.analytics.ticker import AnalyticsBundle, AnalyticsTicker, TickRecord, build_analytics

__all__ = [
    "SegAnalyticsBuffer",
    "SegFrameStats",
    "SegSecondBucket",
    "PipelineLatencyBuffer",
    "FrameTimingStats",
    "LatencySecondBucket",
    "AnalyticsTicker",
    "AnalyticsBundle",
    "TickRecord",
    "build_analytics",
]
