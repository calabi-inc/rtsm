"""
RTSM Demo Runner

Zero-friction demo: runs the full RTSM pipeline on a pre-recorded clip
with 3D visualization, camera feed overlay, and REST API on a single port.

Usage:
    rtsm demo              # from pip install or repo checkout
    rtsm demo --no-viz     # skip visualization (terminal + API only)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*allow_in_graph is deprecated.*", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("rtsm.stores.proximity_index").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _find_demo_data() -> str:
    """Locate the demo clip recording directory.

    Search order:
      1. recordings/demo_clip/ (repo checkout)
      2. <package_dir>/../recordings/demo_clip/ (installed alongside package)

    Returns:
        Absolute path to the demo clip directory.

    Raises:
        FileNotFoundError: If demo data cannot be found.
    """
    # 1. Package data (pip install — ships inside rtsm/data/demo_clip/)
    from importlib import resources
    pkg_data = resources.files("rtsm.data").joinpath("demo_clip")
    pkg_data_path = Path(str(pkg_data))
    if pkg_data_path.is_dir() and (pkg_data_path / "messages.bin").is_file():
        return str(pkg_data_path.resolve())

    # 2. CWD-relative (repo checkout, uses symlink)
    cwd_path = Path("recordings/demo_clip")
    if cwd_path.is_dir() and (cwd_path / "messages.bin").is_file():
        return str(cwd_path.resolve())

    raise FileNotFoundError(
        "Demo data not found.\n"
        "If you installed via pip, try upgrading:  pip install -U rtsm[gpu]\n"
        "If from source:  git pull && rtsm demo"
    )


def _find_static_dir() -> str | None:
    """Locate the built frontend static directory.

    Search order (dev build wins so you never need to copy):
      1. demo/dist/ (dev, after npm run build)
      2. rtsm/static/ (packaged release)
    """
    dev_dist = Path("demo/dist")
    if dev_dist.is_dir() and (dev_dist / "index.html").is_file():
        return str(dev_dist.resolve())
    pkg_static = Path(__file__).parent / "static"
    if pkg_static.is_dir() and (pkg_static / "index.html").is_file():
        return str(pkg_static)
    return None


def run_demo(argv: list[str] | None = None) -> None:
    """Run the RTSM demo."""
    parser = argparse.ArgumentParser(
        prog="rtsm demo",
        description="Run RTSM demo with pre-recorded data",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization server")
    parser.add_argument("--port", type=int, default=8002, help="API/UI port (default: 8002)")
    args = parser.parse_args(argv or [])

    print("=" * 60)
    print("  RTSM Demo - Real-Time Spatio-Semantic Memory")
    print("=" * 60)

    # ── Check GPU dependencies ──
    try:
        from rtsm.core.pipeline import Pipeline
        from rtsm.models.clip.adapter import CLIPAdapter
        from rtsm.models.clip.vocab_classifier import ClipVocabClassifier
    except ImportError as e:
        print(f"\n  ERROR: GPU dependencies not installed: {e}")
        print("  Install with:  pip install \"rtsm[gpu]\"")
        print("  For CUDA:      --extra-index-url https://download.pytorch.org/whl/cu128")
        return

    # ── Locate demo data ──
    print("\n  [1/5] Checking demo data...")
    try:
        demo_dir = _find_demo_data()
        print(f"        Found: {demo_dir}")
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        return

    # ── Load config ──
    print("  [2/5] Loading configuration...")
    from rtsm.cfg import load_config, cfg_path
    cfg = load_config("demo_config.yaml")
    print(f"        Config: {cfg_path('demo_config.yaml')}")
    print(f"        Backend: {cfg['segmentation']['backend']}")

    # ── Load models ──
    print("  [3/5] Loading segmentation model (grounded_sam2)...")
    print("        (Downloads ~1GB from HuggingFace on first run)")
    from rtsm.models.segmentation import get_segmenter
    segmenter = get_segmenter(cfg)
    segmenter.warmup()
    print(f"        Segmentation ready: {segmenter.name}")

    clip_cfg = cfg.get("clip", {})
    clip_model = clip_cfg.get("model", "ViT-B-16-SigLIP")
    clip_pretrained = clip_cfg.get("pretrained", "webli")
    clip_local = clip_cfg.get("local_dir", "model_store/clip")
    print(f"  [4/5] Loading CLIP model ({clip_model})...")
    print("        (Downloads ~350MB on first run)")
    clip = CLIPAdapter(clip_model, clip_pretrained, clip_local, device=cfg.get("device", "cuda"))
    vocab_clf = ClipVocabClassifier(
        clip.artifacts.model, clip.artifacts.tokenizer, clip.artifacts.preprocess,
        str(cfg_path("clip/vocab.yaml")), device=cfg.get("device", "cuda"),
    )
    print("        CLIP ready")

    # ── Initialize spatial memory ──
    print("  [5/5] Initializing spatial memory + pipeline...")
    from rtsm.stores.working_memory import WorkingMemory
    from rtsm.stores.proximity_index import ProximityIndex, GridSpec
    from rtsm.core.association import Associator
    from rtsm.core.ingest_gate import IngestGate
    from rtsm.stores.sweep_cache import SweepCache
    from rtsm.io.ingest_queue import IngestQueue
    from rtsm.io.replayer import ReplayReceiver
    from rtsm.api.server import create_app, start_server, ResetComponents
    from rtsm.utils.net import get_local_ipv4_addresses
    from rtsm.analytics import SegAnalyticsBuffer, PipelineLatencyBuffer

    io_cfg = cfg.get("io", {})
    ws_cfg = io_cfg.get("websocket", {})
    vis_cfg = cfg.get("visualization", {})
    up_axis = "y"  # ARKit is Y-up

    scfg = cfg.get("sweep_cache", {})
    pi_grid = GridSpec(
        cell_m=float(scfg.get("grid_size_m", 0.25)),
        use_3d=not bool(scfg.get("two_d", True)),
        up_axis=up_axis,
    )
    proximity_index = ProximityIndex(pi_grid)
    wm = WorkingMemory(cfg, index=proximity_index)
    assoc = Associator(cfg)
    ingest_gate = IngestGate(cfg)

    # In-memory FAISS (no persistent index needed for demo)
    vec_cfg = cfg.get("vectors", {})
    vectors = None
    if bool(vec_cfg.get("enable", True)):
        from rtsm.stores.vectors.faiss_client import FaissClient
        vectors = FaissClient(cfg)

    ingest_q = IngestQueue(maxsize=512)
    sweep_cache = SweepCache(
        grid_size_m=float(scfg.get("grid_size_m", 0.25)),
        per_cell_cap=int(scfg.get("per_cell_cap", 64)),
        neighbors_max=int(scfg.get("neighbors_max", 128)),
        two_d=bool(scfg.get("two_d", True)),
        yaw_bins=int(scfg.get("yaw_bins", 12)),
        pitch_bins=int(scfg.get("pitch_bins", 5)),
        pitch_deg=float(scfg.get("pitch_deg", 60.0)),
        look_lru_keep=int(scfg.get("look_lru_keep", 8)),
        up_axis=up_axis,
    )

    # Analytics
    analytics_cfg = cfg.get("analytics", {})
    seg_analytics = None
    latency_analytics = None
    if analytics_cfg.get("enable", True):
        seg_analytics = SegAnalyticsBuffer()
        latency_analytics = PipelineLatencyBuffer()

    # ── Visualization server (processing only, WebSocket merged into API) ──
    vis_server = None
    vis_broadcaster = None
    vis_server_registry = None

    if not args.no_viz and vis_cfg.get("enable", True):
        from rtsm.visualization.server import VisualizationServer
        vis_server = VisualizationServer(
            cfg=cfg,
            working_memory=wm,
            host=vis_cfg.get("host", "0.0.0.0"),
            port=int(vis_cfg.get("port", 8083)),
            seg_analytics=seg_analytics,
            latency_analytics=latency_analytics,
        )
        vis_broadcaster = vis_server.broadcaster
        vis_server_registry = vis_server.registry

    # ── Start replay ──
    replay = ReplayReceiver(
        recording_dir=demo_dir,
        ingest_queue=ingest_q,
        require_tracking_normal=bool(ws_cfg.get("require_tracking_normal", True)),
        keyframe_every_n=int(ws_cfg.get("keyframe_every_n", 5)),
        nonkf_min_interval_s=float(ws_cfg.get("nonkf_min_interval_s", 0.3)),
        confidence_threshold=int(ws_cfg.get("confidence_threshold", 1)),
        apply_camera_flip=bool(vis_cfg.get("apply_camera_flip", False)),
        on_keyframe=vis_server.handle_frame_packet if vis_server else None,
        on_camera_frame=vis_server.broadcast_camera_frame if vis_server else None,
        on_pose_corrections=vis_server.handle_kf_pose_update if vis_server else None,
        on_pose_corrections_batch=vis_server.handle_pose_corrections_batch if vis_server else None,
        latency_analytics=latency_analytics,
    )

    pipe = Pipeline(
        cfg=cfg,
        segmenter=segmenter,
        clip=clip,
        working_mem=wm,
        proximity_index=proximity_index,
        associator=assoc,
        ingest_gate=ingest_gate,
        vocab_clf=vocab_clf,
        vectors=vectors,
        ingest_q=ingest_q,
        sweep_cache=sweep_cache,
        seg_analytics=seg_analytics,
        latency_analytics=latency_analytics,
    )

    # ── Start API + viz WebSocket + static frontend on single port ──
    static_dir = _find_static_dir()
    local_ips = get_local_ipv4_addresses()
    display_host = local_ips[0] if local_ips else "localhost"
    port = args.port

    mcp_cfg = cfg.get("mcp", {})
    app = create_app(
        working_memory=wm,
        clip_adapter=clip,
        vectors=vectors,
        extra_stats_provider=lambda: {"ingest_q": ingest_q.qsize()},
        reset_components=ResetComponents(sweep_cache=sweep_cache, vis_server=vis_server),
        seg_analytics=seg_analytics,
        latency_analytics=latency_analytics,
        mcp_enabled=bool(mcp_cfg.get("enable", False)),
        vis_server=vis_server,
        vis_broadcaster=vis_broadcaster,
        vis_registry=vis_server_registry,
        static_dir=static_dir,
    )
    start_server(app, host="0.0.0.0", port=port)

    # Start replay
    replay.start()

    print()
    print("=" * 60)
    print("  RTSM Demo is running!")
    print(f"  API:     http://{display_host}:{port}")
    if static_dir:
        print(f"  Web UI:  http://{display_host}:{port}")
    else:
        print("  Web UI:  Not available (build frontend with: cd demo && npm run build)")
    if vis_server:
        print(f"  Viz WS:  ws://{display_host}:{port}/ws")
    print(f"  Objects: http://{display_host}:{port}/objects")
    print(f"  Search:  http://{display_host}:{port}/search/semantic?query=cup")
    print()
    print("  Processing demo clip... (50 frames, ~25s)")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    # Auto-open browser to the web UI
    if static_dir and not args.no_viz:
        from rtsm.utils.browser import open_browser
        url = f"http://localhost:{port}"
        threading.Timer(1.5, open_browser, args=[url]).start()

    try:
        pipe.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Print summary
        all_objs = list(wm.iter_objects())
        confirmed = sum(1 for o in all_objs if o.confirmed)
        proto = sum(1 for o in all_objs if not o.confirmed)
        print(f"\n  Demo complete: {confirmed} confirmed objects, {proto} proto")
        print(f"  Query the API: curl http://{display_host}:{port}/objects")
