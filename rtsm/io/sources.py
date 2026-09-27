"""
Ingest source registry (Gate 4.5 plan, P3 task 0.5).

A *source* is a transport adapter on the one ingest front-end
(``rtsm/io/ingest_frontend.py``): it turns bytes from somewhere (a Lens
websocket, a ZeroMQ bridge, a recording, a bag file) into ``RawFrame`` /
``PoseSample`` events and satisfies the ``Source`` protocol
(``rtsm/io/contracts.py``: ``name``, ``start()``, ``stop()``, ``liveness()``).

The runners build one ``SourceContext`` (queue, ingest settings, receive-time
sinks, callbacks) and ask this module for a source by name. Built-ins:
``websocket``, ``zeromq``, ``replay``. Third-party sources register under the
``rtsm.sources`` entry-point group::

    [project.entry-points."rtsm.sources"]
    mybag = "my_package.rtsm_source:make_source"

with ``make_source(cfg: dict, ctx: SourceContext, **options) -> Source``. A
plug-in name that collides with a built-in is ignored with a warning (the
built-ins are the anchored code paths).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from rtsm.io.contracts import Source, SourceContext

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "rtsm.sources"

SourceFactory = Callable[..., Any]     # (cfg: dict, ctx: SourceContext, **options) -> Source


class UnknownSourceError(ValueError):
    """``io.receiver`` (or --replay) named a source nobody registered."""


# ───────────────────────────── built-in factories ─────────────────────────────

def websocket_source(cfg: dict, ctx: SourceContext, **options: Any):
    """The Lens websocket receiver (``rtsm/io/websocket.py``)."""
    from rtsm.io.websocket import WebSocketReceiver
    ws_cfg = (cfg.get("io") or {}).get("websocket") or {}
    return WebSocketReceiver(
        ingest_queue=ctx.ingest_queue,
        host=str(ws_cfg.get("host", "0.0.0.0")),
        port=int(ws_cfg.get("port", 8765)),
        require_tracking_normal=ctx.require_tracking_normal,
        keyframe_every_n=ctx.keyframe_every_n,
        nonkf_min_interval_s=ctx.nonkf_min_interval_s,
        confidence_threshold=ctx.confidence_threshold,
        apply_camera_flip=ctx.apply_camera_flip,
        on_keyframe=ctx.on_keyframe,
        on_camera_frame=ctx.on_camera_frame,
        on_pose_corrections=ctx.on_pose_corrections,
        on_pose_corrections_batch=ctx.on_pose_corrections_batch,
        on_raw_message=ctx.on_raw_message,
        on_handshake_done=ctx.on_handshake_done,
        on_frame_correction=ctx.on_frame_correction,
        pose_sink=ctx.pose_sink,
        clearance_sink=ctx.clearance_sink,
        event_sink=ctx.event_sink,
        ledger_sink=ctx.ledger_sink,
        throttle_clock=ctx.clock_mode,
        latency_analytics=ctx.latency_analytics,
    )


def zeromq_source(cfg: dict, ctx: SourceContext, **options: Any):
    """The RTAB-Map bridge subscriber (``rtsm/io/zeromq.py``). ``pair_window_s``
    / ``pair_window_frames`` (LaneConfig) arrive as options from the runner."""
    from rtsm.io.zeromq import ZeroMQSubscriber
    io_cfg = cfg.get("io") or {}
    units_cfg = cfg.get("units") or {}
    kwargs: Dict[str, Any] = dict(
        camera_endpoint=io_cfg.get("camera_endpoint", "tcp://127.0.0.1:5555"),
        rtabmap_endpoint=io_cfg.get("rtabmap_endpoint", "tcp://127.0.0.1:6000"),
        ingest_queue=ctx.ingest_queue,
        depth_m_per_unit=float(units_cfg.get("depth_m_per_unit", 0.001)),
        pose_m_per_unit=float(units_cfg.get("pose_m_per_unit", 1.0)),
        on_kf_packet=ctx.on_kf_packet,
        on_kf_pose_update=ctx.on_kf_pose_update,
        pose_sink=ctx.pose_sink,
        event_sink=ctx.event_sink,
        ledger_sink=ctx.ledger_sink,
        throttle_clock=ctx.clock_mode,
        nonkf_min_interval_s=ctx.nonkf_min_interval_s,
        latency_analytics=ctx.latency_analytics,
    )
    if options.get("pair_window_s") is not None:
        kwargs["frame_window_ttl_s"] = float(options["pair_window_s"])
    if options.get("pair_window_frames") is not None:
        kwargs["frame_window_max_items"] = int(options["pair_window_frames"])
    return ZeroMQSubscriber(**kwargs)


def replay_source(cfg: dict, ctx: SourceContext, **options: Any):
    """The recording replayer (``rtsm/io/replayer.py``). Options:
    ``recording_dir`` (required), ``replay_speed`` (default 1.0). The replayer
    takes no clearance sink (the guard is a live-receiver feature)."""
    recording_dir = options.get("recording_dir")
    if not recording_dir:
        raise ValueError("replay source needs recording_dir")
    from rtsm.io.replayer import ReplayReceiver
    return ReplayReceiver(
        recording_dir=str(recording_dir),
        ingest_queue=ctx.ingest_queue,
        require_tracking_normal=ctx.require_tracking_normal,
        keyframe_every_n=ctx.keyframe_every_n,
        nonkf_min_interval_s=ctx.nonkf_min_interval_s,
        confidence_threshold=ctx.confidence_threshold,
        apply_camera_flip=ctx.apply_camera_flip,
        on_keyframe=ctx.on_keyframe,
        on_camera_frame=ctx.on_camera_frame,
        on_pose_corrections=ctx.on_pose_corrections,
        on_pose_corrections_batch=ctx.on_pose_corrections_batch,
        on_frame_correction=ctx.on_frame_correction,
        latency_analytics=ctx.latency_analytics,
        replay_speed=float(options.get("replay_speed", 1.0) or 1.0),
        event_sink=ctx.event_sink,
        ledger_sink=ctx.ledger_sink,
        throttle_clock=ctx.clock_mode,
        pose_sink=ctx.pose_sink,
    )


_BUILTIN: Dict[str, SourceFactory] = {
    "websocket": websocket_source,
    "zeromq": zeromq_source,
    "replay": replay_source,
}
_REGISTERED: Dict[str, SourceFactory] = {}


# ───────────────────────────── registry ─────────────────────────────

def register_source(name: str, factory: SourceFactory, *, replace: bool = False) -> None:
    """Register a source factory in-process (tests, embedded use). A built-in
    name cannot be replaced."""
    key = str(name).lower()
    if key in _BUILTIN:
        raise ValueError(f"{key!r} is a built-in source and cannot be replaced")
    if key in _REGISTERED and not replace:
        raise ValueError(f"source {key!r} is already registered (replace=True to override)")
    _REGISTERED[key] = factory


def unregister_source(name: str) -> None:
    _REGISTERED.pop(str(name).lower(), None)


def _entry_point_sources() -> Dict[str, SourceFactory]:
    """Factories advertised under the ``rtsm.sources`` entry-point group. A
    plug-in that fails to load is skipped with a warning (never breaks the
    built-ins)."""
    found: Dict[str, SourceFactory] = {}
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as e:  # noqa: BLE001 -- metadata problems must not break startup
        logger.debug("[sources] entry point scan failed: %s", e)
        return found
    for ep in eps:
        key = str(ep.name).lower()
        if key in _BUILTIN:
            logger.warning("[sources] entry point %r collides with a built-in source; ignored", ep.name)
            continue
        try:
            found[key] = ep.load()
        except Exception as e:  # noqa: BLE001
            logger.warning("[sources] entry point %r failed to load: %s", ep.name, e)
    return found


def available_sources() -> Dict[str, SourceFactory]:
    """Built-ins, then in-process registrations, then entry points (earlier wins)."""
    out: Dict[str, SourceFactory] = dict(_BUILTIN)
    for key, f in _REGISTERED.items():
        out.setdefault(key, f)
    for key, f in _entry_point_sources().items():
        out.setdefault(key, f)
    return out


def make_source(name: str, cfg: dict, ctx: SourceContext, **options: Any):
    """Build the source ``name`` from the runner's context. Raises
    UnknownSourceError (a ValueError) listing the names that exist."""
    key = str(name).lower()
    sources = available_sources()
    factory = sources.get(key)
    if factory is None:
        raise UnknownSourceError(
            f"Unknown ingest source {name!r} (io.receiver). Available: {', '.join(sorted(sources))}."
        )
    src = factory(cfg, ctx, **options)
    if not isinstance(src, Source):
        raise TypeError(f"source factory {key!r} returned {type(src).__name__}, which lacks name/start/stop/liveness")
    return src
