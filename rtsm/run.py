from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*allow_in_graph is deprecated.*", category=FutureWarning)

# ── GPU-dependent imports (require rtsm[gpu]) ──
_GPU_AVAILABLE = True
_GPU_IMPORT_ERROR = None
try:
    from rtsm.core.pipeline import Pipeline
    from rtsm.models.clip.adapter import CLIPAdapter
    from rtsm.models.clip.vocab_classifier import ClipVocabClassifier
except ImportError as e:
    _GPU_AVAILABLE = False
    _GPU_IMPORT_ERROR = str(e)
    Pipeline = None  # type: ignore[assignment,misc]
    CLIPAdapter = None  # type: ignore[assignment,misc]
    ClipVocabClassifier = None  # type: ignore[assignment,misc]

# ── Core imports (always available) ──
from rtsm.models.segmentation import get_segmenter  # noqa: F401 -- kept: tests patch run.get_segmenter as the model-load guard
from rtsm.engine import build_runtime, load_models
from rtsm.stores.working_memory import WorkingMemory, resolve_pose_stale_after_s
from rtsm.stores.proximity_index import ProximityIndex, GridSpec
from rtsm.core.association import Associator
from rtsm.core.ingest_gate import IngestGate
from rtsm.stores.sweep_cache import SweepCache
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.ingest_lanes import LaneConfig, lane_drop_handler, make_ingest_queue
from rtsm.io.websocket import WebSocketReceiver
from rtsm.io.contracts import SourceContext
from rtsm.io.sources import UnknownSourceError, available_sources, make_source
from rtsm.utils.net import print_server_addresses, get_local_ipv4_addresses
from rtsm.utils.static_dir import find_static_dir
from rtsm.stores.sweep_policy import SweepPolicy
from rtsm.api.server import create_app, start_server, ResetComponents
from rtsm.cfg import ConfigError, cfg_path, config_fingerprint
from rtsm.cfg.cli import add_config_arguments, config_from_args
from rtsm.cfg.tuning import validate_tuning
from rtsm.evaluation.event_log import EventLogWriter, resolve_ledger_config
from rtsm.core.clock import make_clock, resolve_clock_mode

import argparse
import sys
import threading
import logging
from pathlib import Path

# Configure logging at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Reduce verbosity of noisy subsystems
logging.getLogger("rtsm.stores.proximity_index").setLevel(logging.WARNING)
logging.getLogger("rtsm.core.association").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def main(argv: "list[str] | None" = None):
    # argv: the console entries (rtsm.cli) call main() bare -> sys.argv; tests
    # pass a list, so the startup block below is executable on CPU (P1 task 6).
    argv = sys.argv[1:] if argv is None else list(argv)
    # Dispatch 'demo' subcommand before argparse (preserves backward compat)
    if argv and argv[0] == "demo":
        from rtsm.demo import run_demo
        run_demo(argv[1:])
        return

    parser = argparse.ArgumentParser(description="RTSM - Real-Time Spatio-Semantic Memory")
    parser.add_argument("--replay", type=str, default=None, metavar="DIR",
                        help="Replay a recorded session from DIR at original rate")
    parser.add_argument("--record", type=str, default=None, metavar="DIR",
                        help="Record raw WebSocket session to DIR")
    parser.add_argument("--bag", type=str, default=None, metavar="PATH",
                        help="Read a ROS 1 .bag, a rosbag2 directory (sqlite3 | mcap) or a bare .mcap through the bag source "
                             "(sets io.receiver=bag; the ingest clock and policy resolve as under --replay)")
    parser.add_argument("--bag-speed", type=float, default=None, metavar="X",
                        help="Pace bag frames by their header stamps at X times real time (default: as fast as the ingest queue accepts)")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Replay speed multiplier (<1 = slower, e.g. 0.5 = half speed)")
    parser.add_argument("--record-only", action="store_true",
                        help="Record without running pipeline (no GPU needed)")
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument("--viz", action="store_true",
                           help="Start the visualization server (3D dashboard + browser auto-open); "
                                "off by default since P1 task 7. Same as --set visualization.enable=true")
    viz_group.add_argument("--no-viz", action="store_true",
                           help="Force the visualization server off (headless replay / eval / CI); "
                                "same as --set visualization.enable=false")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))
    if args.viz:
        cfg.setdefault("visualization", {})["enable"] = True
    if args.no_viz:
        cfg.setdefault("visualization", {})["enable"] = False

    print("=" * 60)
    print("  RTSM - Real-Time Spatio-Semantic Memory")
    print("=" * 60)

    logger.info("Configuration loaded: %s (SHA-256 %s)",
                cfg_path(args.config if args.config is not None else "rtsm.yaml"),
                config_fingerprint(cfg))
    for advisory in validate_tuning(cfg):
        logger.warning("Configuration: %s", advisory)

    # Ingest clock (ingest.clock: auto|wall|sensor). Drives frame admission and
    # memory timing: the receiver non-KF throttle, ingest-gate grace / TTL /
    # parallax ages, proto expiry and LTM scheduling. auto = sensor under
    # --replay (speed-independent, repeatable), wall for live receivers.
    # Resolved here, before any model or index is built, so a bad value exits
    # through the config-error path like every other config mistake.
    try:
        clock_mode = resolve_clock_mode((cfg.get("ingest") or {}).get("clock", "auto"),
                                        replay=bool(args.replay or args.bag))
    except ValueError as exc:
        parser.error(str(exc))
    clock = make_clock(clock_mode)
    logger.info("Ingest clock: %s (ingest.clock=%s)", clock_mode,
                (cfg.get("ingest") or {}).get("clock", "auto"))
    # Ingest policy (ingest.policy: auto|latest|lossless|legacy) -- validated
    # here too, so a typo fails before any model loads; lossless is refused
    # for live receivers (its producer-paced put would block the receive loop).
    try:
        lane_cfg = LaneConfig.from_cfg(cfg, replay=bool(args.replay or args.bag))
        resolve_pose_stale_after_s(cfg)     # robot_pose.stale_after_s: finite, > 0
        ledger_cfg = resolve_ledger_config(cfg)      # diagnostics.ledgers / ledger_format (P2)
    except ValueError as exc:
        parser.error(str(exc))
    logger.info("Ingest policy: %s (ingest.policy=%s)", lane_cfg.policy, lane_cfg.configured_policy)
    # The ingest source name (io.receiver) is validated here too (P3 task
    # 0.5): a name nobody registered (built-in or `rtsm.sources` entry point)
    # exits with the known names before the GPU check, long before a model
    # loads. --replay always uses the replay source.
    if args.bag:
        cfg.setdefault("io", {})["receiver"] = "bag"
        cfg["io"].setdefault("bag", {})["path"] = args.bag
    if not args.replay:
        _rt = str((cfg.get("io") or {}).get("receiver", "zeromq")).lower()
        _known = available_sources()
        if _rt not in _known:
            parser.error(f"Unknown ingest source {_rt!r} (io.receiver). Available: {', '.join(sorted(_known))}.")
        if _rt == "bag":
            # The bag is probed here (topics, pose source, registration): a refusal costs seconds, not a model load.
            _bag_cfg = dict((cfg.get("io") or {}).get("bag") or {})
            if not _bag_cfg.get("path"):
                parser.error("io.receiver=bag needs io.bag.path (or --bag PATH)")
            try:
                from rtsm.io.bag_reader import probe_bag
                _probe = probe_bag(_bag_cfg["path"], topics=dict(_bag_cfg.get("topics") or {}),
                                   tf_extrapolation_s=float(_bag_cfg.get("tf_extrapolation_s", 0.05)),
                                   world_frame=_bag_cfg.get("world_frame") or None, camera_frame=_bag_cfg.get("camera_frame") or None,
                                   assume_aligned=bool(_bag_cfg.get("assume_aligned", False)), typestore=str(_bag_cfg.get("typestore") or "humble"))
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                parser.error(f"--bag: {exc}")
            if _probe.refusal:
                parser.error("bag refused: " + "; ".join(f"{c}: {d}" for c, d in _probe.refusal))
            logger.info("Bag %s: %s | topics %s | pose %s %s -> %s | %s", _bag_cfg["path"], _probe.bag,
                        {k: v for k, v in _probe.topics.items() if k != "rules" and v}, _probe.pose_kind, _probe.world_frame,
                        _probe.camera_frame, _probe.registration)

    # After the ingest validation on purpose: a bad config exits through
    # parser.error on any machine (CPU test), before this check can return.
    # ── Check GPU dependencies unless in record-only mode ──
    if not (args.record and args.record_only) and not _GPU_AVAILABLE:
        print(f"\nERROR: GPU dependencies not installed: {_GPU_IMPORT_ERROR}")
        print("Install with:  pip install \"rtsm[gpu]\"  or  pip install \"rtsm[all]\"")
        print("For CUDA support, add:  --extra-index-url https://download.pytorch.org/whl/cu128")
        return

    # ── Record-only mode: skip all heavy init, just record raw WebSocket ──
    if args.record and args.record_only:
        from rtsm.io.recorder import SessionRecorder
        recorder = SessionRecorder(output_dir=args.record, config_snapshot=cfg)

        io_cfg = cfg.get("io", {})
        ws_cfg = io_cfg.get("websocket", {})
        vis_cfg = cfg.get("visualization", {})
        # Record-only has no consumer: one slot, then every frame is refused
        # BEFORE its RGB decode (the recorder reads raw bytes upstream of the
        # parser). Deliberately not the lane object -- a latest slot would keep
        # admitting and decoding frames nobody reads.
        ingest_q = IngestQueue(maxsize=1)

        ws_receiver = WebSocketReceiver(
            ingest_queue=ingest_q,
            host=str(ws_cfg.get("host", "0.0.0.0")),
            port=int(ws_cfg.get("port", 8765)),
            require_tracking_normal=bool(ws_cfg.get("require_tracking_normal", True)),
            keyframe_every_n=lane_cfg.keyframe_every_n,
            nonkf_min_interval_s=lane_cfg.nonkf_min_interval_s,
            confidence_threshold=int(ws_cfg.get("confidence_threshold", 1)),
            apply_camera_flip=bool(vis_cfg.get("apply_camera_flip", False)),
            on_raw_message=recorder.on_message,
            on_handshake_done=recorder.on_handshake,
        )
        ws_receiver.start()
        ws_port = int(ws_cfg.get("port", 8765))
        local_ips = get_local_ipv4_addresses()
        display_host = local_ips[0] if local_ips else "0.0.0.0"
        print_server_addresses(ws_port)
        logger.info(f"Record-only mode: ws://{display_host}:{ws_port}/stream -> {args.record}")
        print(f"  Recording to: {args.record}")
        print("  Press Ctrl+C to stop recording")

        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            recorder.close()
        return

    # Create segmenter from config
    # ── The engine (rtsm/engine.py, P3 task 2): models once, then the runtime ──
    # Same construction as before, moved into the factory the eval runner
    # shares; the G1-B anchor through this runner is the proof it did not move.
    models = load_models(cfg)
    segmenter, clip, vocab_clf = models.segmenter, models.clip, models.vocab_clf
    # Determine world-frame up axis from receiver type (ARKit=Y-up, D435i/ROS=Z-up)
    io_cfg = cfg.get("io", {})
    receiver_type = str(io_cfg.get("receiver", "zeromq")).lower()
    up_axis_default = "y" if receiver_type in ("websocket", "bag") or args.replay else "z"
    event_log = EventLogWriter(
        enabled=ledger_cfg.enabled,
        configured_path=ledger_cfg.event_log_path,
        extra_meta={"ingest_clock": clock_mode, "ingest_policy": lane_cfg.policy},
        # P2 ledgers (diagnostics.ledgers): the pose ledger rides in the same file.
        ledgers=ledger_cfg.ledgers,
        ledger_format=ledger_cfg.ledger_format,
    )
    event_sink = event_log.sink()
    ledger_sink = event_log.ledger_sink()      # None unless diagnostics.ledgers is on
    rt = build_runtime(cfg, models, clock=clock, lane_cfg=lane_cfg, event_log=event_log, up_axis_default=up_axis_default)
    proximity_index, wm, analytics = rt.proximity_index, rt.wm, rt.analytics
    seg_analytics, latency_analytics = analytics.seg, analytics.latency
    assoc, ingest_gate, vectors, ingest_q, sweep_cache, up_axis = (rt.associator, rt.ingest_gate, rt.vectors, rt.ingest_q,
                                                                    rt.sweep_cache, rt.up_axis)

    # ---------------- Visualization Server (optional) ----------------
    vis_cfg = cfg.get("visualization", {})
    vis_server = None
    if vis_cfg.get("enable", True):
        from rtsm.visualization.server import VisualizationServer
        vis_server = VisualizationServer(
            cfg=cfg,
            working_memory=wm,
            host=vis_cfg.get("host", "0.0.0.0"),
            port=int(vis_cfg.get("port", 8081)),
            seg_analytics=seg_analytics,
            latency_analytics=latency_analytics,
            ingest_queue=ingest_q,
            analytics_ticker=analytics.ticker,
        )
        logger.info("Visualization server initialized")

    # Resolve display IP (use real network IP instead of 0.0.0.0)
    local_ips = get_local_ipv4_addresses()
    display_host = local_ips[0] if local_ips else "0.0.0.0"

    # ---------------- Receiver (Replay, WebSocket, or ZMQ) ----------------
    units_cfg = cfg.get("units", {})
    ws_cfg = io_cfg.get("websocket", {})
    recorder = None
    # Receive-time forward clearance (io.clearance.enable, default false).
    # WorkingMemory owns the flag: it decides whether /stats carries the
    # forward_clearance key, and the websocket receiver only gets a sink
    # when it does. Flag off => the receiver never computes depth statistics
    # and /stats is identical to a build without the feature.
    clearance_enabled = bool(getattr(wm, "clearance_enabled", False))
    if clearance_enabled and (args.replay or args.bag or receiver_type != "websocket"):
        logger.warning(
            "io.clearance.enable=true but no receive-time clearance source: "
            "only the live websocket receiver computes it (replay=%s, receiver=%s); "
            "/stats.forward_clearance will stay null", bool(args.replay), receiver_type)

    # Frame-flow trace (diagnostics.*): one JSONL writer shared by the receiver
    # thread and the pipeline thread. Disabled (the default) => every hook is a
    # no-op and event_sink is None, so receivers skip building events entirely.

    # ── The ingest source (P3 task 0.5) ──
    # Every source is a transport adapter on the ONE ingest front-end
    # (rtsm/io/ingest_frontend.py); the registry (rtsm/io/sources.py) builds
    # it from one context. `io.receiver` names a built-in (websocket |
    # zeromq) or a plug-in registered under the `rtsm.sources` entry-point
    # group; --replay selects the replay source regardless.
    if args.record and not args.replay and receiver_type == "websocket":
        from rtsm.io.recorder import SessionRecorder
        recorder = SessionRecorder(output_dir=args.record, config_snapshot=cfg)
    source_ctx = SourceContext(
        ingest_queue=ingest_q,
        clock_mode=clock_mode,
        keyframe_every_n=lane_cfg.keyframe_every_n,
        nonkf_min_interval_s=lane_cfg.nonkf_min_interval_s,
        require_tracking_normal=bool(ws_cfg.get("require_tracking_normal", True)),
        confidence_threshold=int(ws_cfg.get("confidence_threshold", 1)),
        apply_camera_flip=bool(vis_cfg.get("apply_camera_flip", False)),
        # Receive-time robot pose passthrough: agents polling robot_pose get
        # input-rate freshness (~5 Hz) instead of the pipeline's sweep-gated
        # processing rate (~1 Hz). Under replay too, so replay-based
        # pose-freshness checks mean something.
        pose_sink=wm.update_robot_pose,
        # Receive-time depth clearance (wall guard for blind agent motion),
        # same freshness rationale. Opt-in via io.clearance.enable (see
        # clearance_enabled above); None makes the receiver skip the depth
        # statistic entirely. Live websocket only (the replayer takes none).
        clearance_sink=wm.set_forward_clearance if clearance_enabled else None,
        event_sink=event_sink,
        ledger_sink=ledger_sink,
        latency_analytics=latency_analytics,
        on_camera_frame=vis_server.broadcast_camera_frame if vis_server else None,
        on_keyframe=vis_server.handle_frame_packet if vis_server else None,
        on_pose_corrections=vis_server.handle_kf_pose_update if vis_server else None,
        on_pose_corrections_batch=vis_server.handle_pose_corrections_batch if vis_server else None,
        on_kf_packet=vis_server.handle_kf_packet if vis_server else None,
        on_kf_pose_update=vis_server.handle_kf_pose_update if vis_server else None,
        on_raw_message=recorder.on_message if recorder else None,
        on_handshake_done=recorder.on_handshake if recorder else None,
    )
    source_name = "replay" if args.replay else receiver_type
    try:
        source = make_source(
            source_name, cfg, source_ctx,
            recording_dir=args.replay, replay_speed=args.replay_speed,
            path=(args.bag or None), speed=args.bag_speed,
            pair_window_s=lane_cfg.pair_window_s,             # ingest.* (P1 task 6)
            pair_window_frames=lane_cfg.pair_window_frames,
        )
    except UnknownSourceError as exc:
        raise ValueError(str(exc)) from exc
    source.start()
    replay_like = bool(args.replay or source.name == "bag")
    replay_receiver = source if replay_like else None
    # FrameWindow (ZeroMQ) or None: the /reset handler clears the pairing window
    frame_window_for_reset = getattr(source, "fw", None)
    if args.replay:
        logger.info(f"Replay receiver started from {args.replay} (speed={args.replay_speed}x)")
    elif source.name == "bag":
        logger.info(f"Bag source started: {getattr(source, 'path', '?')} (speed={args.bag_speed or 'as fast as accepted'})")
    elif source.name == "websocket":
        ws_port = int(ws_cfg.get("port", 8765))
        print_server_addresses(ws_port)
        logger.info(f"WebSocket receiver started on ws://{display_host}:{ws_port}/stream")
        if recorder:
            logger.info(f"Recording to {args.record}")
    elif source.name == "zeromq":
        logger.info("ZeroMQ dual-socket subscriber started (camera + RTABMap)")
    else:
        logger.info(f"Ingest source {source.name!r} started")

    # Visualization tasks start via API server lifespan (start_tasks())
    if vis_server:
        logger.info("Visualization server initialized (tasks start with API server)")

    pipe = rt.pipeline

    # ---------------- Frame-flow watchdog (live receivers only) ----------------
    # Distinguishes "starved" (input stopped), "hung" (frames waiting, loop
    # silent), and "receiver_dead" — surfaced on /healthz. Replay mode skips it
    # (a finished replay looks exactly like starvation).
    watchdog = None
    wd_cfg = cfg.get("health", {}).get("watchdog", {})
    if bool(wd_cfg.get("enable", True)) and not replay_like:
        receiver_liveness = getattr(source, "liveness", None)
        if receiver_liveness is not None:
            from rtsm.core.watchdog import Watchdog
            watchdog = Watchdog(
                heartbeat=pipe.heartbeat,
                queue_size=ingest_q.qsize,
                backlog=ingest_q.backlog_signal,
                receiver_liveness=receiver_liveness,
                starved_after_s=float(wd_cfg.get("starved_after_s", 5.0)),
                hung_after_s=float(wd_cfg.get("hung_after_s", 10.0)),
                poll_interval_s=float(wd_cfg.get("poll_interval_s", 1.0)),
            )
            watchdog.start()
            logger.info("Frame-flow watchdog started")

    # ---------------- Start FastAPI control-plane ----------------
    api_cfg = cfg.get("api", {})
    host = str(api_cfg.get("host", "0.0.0.0"))
    port = int(api_cfg.get("port", 8000))

    # Components that can be reset without restarting RTSM
    reset_components = ResetComponents(
        sweep_cache=sweep_cache,
        frame_window=frame_window_for_reset,  # FrameWindow (ZMQ) or None (WebSocket)
        vis_server=vis_server,
        clock=clock,  # SensorClock re-anchors on /reset (WallClock: no-op)
    )

    mcp_cfg = cfg.get("mcp", {})
    mcp_enabled = bool(mcp_cfg.get("enable", False))

    # Resolve frontend static dir and viz broadcaster for single-port serving.
    # Headless (visualization.enable false): no dashboard page either -- the API
    # root would otherwise serve a frontend whose /ws never connects.
    static_dir = find_static_dir() if vis_server else None
    vis_broadcaster = vis_server.broadcaster if vis_server else None
    vis_server_registry = vis_server.registry if vis_server else None

    app = create_app(
        working_memory=wm,
        clip_adapter=clip,
        vectors=vectors,
        extra_stats_provider=lambda: {
            "ingest_q": ingest_q.qsize(),
            "ingest_lanes": ingest_q.stats(),
            "pose_conversion_failures": pipe.pose_conversion_failures,
        },
        reset_components=reset_components,
        seg_analytics=seg_analytics,
        latency_analytics=latency_analytics,
        mcp_enabled=mcp_enabled,
        vis_server=vis_server,
        vis_broadcaster=vis_broadcaster,
        vis_registry=vis_server_registry,
        static_dir=static_dir,
        frame_flow_provider=watchdog.status if watchdog else None,
        ingest_provider=ingest_q.stats,
        analytics_ticker=analytics.ticker,
    )
    start_server(app, host=host, port=port)
    logger.info(f"FastAPI server started on http://{display_host}:{port}")
    if mcp_enabled:
        logger.info(f"MCP server (SSE) available at http://{display_host}:{port}/mcp/sse")

    print("=" * 60)
    print("  RTSM is running! Waiting for data...")
    if args.replay:
        print(f"  Receiver: Replay ({args.replay})")
    elif source.name == "bag":
        print(f"  Receiver: Bag ({getattr(source, 'path', '?')})")
    elif receiver_type == "websocket":
        ws_port = int(io_cfg.get("websocket", {}).get("port", 8765))
        print(f"  Receiver: WebSocket (ws://{display_host}:{ws_port}/stream)")
        if recorder:
            print(f"  Recording: {args.record}")
    elif receiver_type == "zeromq":
        print(f"  Receiver: ZeroMQ")
        print(f"  Camera:  {io_cfg.get('camera_endpoint', 'tcp://127.0.0.1:5555')}")
        print(f"  RTABMap: {io_cfg.get('rtabmap_endpoint', 'tcp://127.0.0.1:6000')}")
    else:
        print(f"  Receiver: {source.name}")
    print(f"  API:     http://{display_host}:{port}")
    if static_dir:
        print(f"  Web UI:  http://{display_host}:{port}")
    if vis_broadcaster:
        print(f"  Viz WS:  ws://{display_host}:{port}/ws")
    if mcp_enabled:
        print(f"  MCP:     http://{display_host}:{port}/mcp/sse")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    # Auto-open browser to the web UI
    if vis_server:
        from rtsm.utils.browser import open_browser
        url = f"http://localhost:{port}"
        threading.Timer(1.5, open_browser, args=[url]).start()

    # Force-flush all confirmed objects to FAISS after replay completes.
    # Without this, most confirmed objects aren't upserted because the
    # flush timer can't keep up with the replay speed.
    if replay_like and vectors is not None:
        def _flush_after_replay():
            replay_receiver.wait()
            import time as _time
            # Wait for pipeline to drain queue + finish processing last frames
            for _ in range(120):
                if ingest_q.qsize() == 0:
                    _time.sleep(5.0)  # wait 5s after queue empty for pipeline to finish
                    break
                _time.sleep(0.5)
            ready = wm.collect_ready_for_upsert(force_all=True)
            if ready:
                try:
                    vectors.upsert_batch(ready)
                    logger.info(f"[run] Force-flushed {len(ready)} objects to vector store after replay")
                except Exception as e:
                    logger.warning(f"[run] Force flush failed: {e}")

        flush_thread = threading.Thread(target=_flush_after_replay, daemon=True, name="replay-flush")
        flush_thread.start()

    # The Tier-2 rollup starts with the consumer: every load, the receiver and
    # the API server are up, so the ticker's counters describe the run only.
    analytics.start()
    try:
        pipe.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
        event_log.close()   # no-op if the pipeline already closed it
        analytics.stop()

if __name__ == "__main__":
    main()
