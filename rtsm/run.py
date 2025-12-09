from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from rtsm.core.pipeline import Pipeline
from rtsm.models.fastsam.adapter import FastSAMAdapter
from rtsm.models.clip.adapter import CLIPAdapter
from rtsm.stores.working_memory import WorkingMemory
from rtsm.stores.proximity_index import ProximityIndex, GridSpec
from rtsm.core.association import Associator
from rtsm.stores.vectors.faiss_client import FaissClient
from rtsm.core.ingest_gate import IngestGate
from rtsm.models.clip.vocab_classifier import ClipVocabClassifier
from rtsm.stores.sweep_cache import SweepCache
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.zeromq import ZeroMQSubscriber
from rtsm.stores.sweep_policy import SweepPolicy
from rtsm.api.server import create_app, start_server, ResetComponents

import yaml
import threading
import logging

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


def main():
    print("=" * 60)
    print("  RTSM - Real-Time Spatio-Semantic Memory")
    print("=" * 60)

    cfg = yaml.safe_load(open("config/rtsm.yaml", "r"))
    logger.info("Configuration loaded from config/rtsm.yaml")
    fastsam = FastSAMAdapter("model_store/fastsam/FastSAM-x.pt", cfg.get("device","cuda"))
    logger.info(f"FastSAM model successfully loaded from model_store/fastsam/FastSAM-x.pt")
    clip = CLIPAdapter("ViT-B-32", "openai", "model_store/clip", device=cfg.get("device","cuda"))
    logger.info(f"CLIP model successfully loaded from model_store/clip")
    # Proximity index config
    scfg = cfg.get("sweep_cache", {})
    two_d = bool(scfg.get("two_d", True))
    cell_m = float(scfg.get("grid_size_m", 0.25))
    per_cell_cap = int(scfg.get("per_cell_cap", 64))
    neighbors_max = int(scfg.get("neighbors_max", 128))
    pi_grid = GridSpec(cell_m=cell_m, use_3d=not two_d)
    proximity_index = ProximityIndex(pi_grid, per_cell_cap=per_cell_cap, neighbors_max=neighbors_max)
    logger.info(f"Proximity index successfully initialized")
    wm = WorkingMemory(cfg, index=proximity_index)
    logger.info(f"Working memory successfully initialized")
    assoc = Associator(cfg)
    ingest_gate = IngestGate(cfg)
    logger.info(f"Ingest gate successfully initialized")
    vocab_clf = ClipVocabClassifier(clip.artifacts.model, clip.artifacts.tokenizer, clip.artifacts.preprocess, "config/clip/vocab.yaml", device=cfg.get("device","cuda"))
    logger.info(f"CLIP vocabulary classifier successfully initialized")
    vec_cfg = cfg.get("vectors", {})
    backend = str(vec_cfg.get("backend", "faiss")).lower()
    vectors = None
    if bool(vec_cfg.get("enable", True)):
        if backend == "milvus":
            from rtsm.stores.vectors.milvus_client import MilvusClient  # lazy import
            vectors = MilvusClient(cfg)
        else:
            vectors = FaissClient(cfg)
            logger.info(f"Faiss vectors successfully initialized")

    # Prepare ingest plumbing
    # Note: Intrinsics are now dynamic per-frame from camera.rgbd topic
    ingest_q = IngestQueue(maxsize=512)
    sweep_cache = SweepCache(
        grid_size_m=float(cfg.get("sweep_cache", {}).get("grid_size_m", 0.25)),
        per_cell_cap=int(cfg.get("sweep_cache", {}).get("per_cell_cap", 64)),
        neighbors_max=int(cfg.get("sweep_cache", {}).get("neighbors_max", 128)),
        two_d=bool(cfg.get("sweep_cache", {}).get("two_d", True)),
        yaw_bins=int(cfg.get("sweep_cache", {}).get("yaw_bins", 12)),
        pitch_bins=int(cfg.get("sweep_cache", {}).get("pitch_bins", 5)),
        pitch_deg=float(cfg.get("sweep_cache", {}).get("pitch_deg", 60.0)),
        look_lru_keep=int(cfg.get("sweep_cache", {}).get("look_lru_keep", 8)),
    )
    logger.info(f"Sweep cache successfully initialized")

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
        )
        logger.info("Visualization server initialized")

    # Start ZMQ subscriber in background (dual-socket: camera + RTABMap)
    # Unit scales (incoming → meters). Defaults provided match typical RTABMap setup.
    io_cfg = cfg.get("io", {})
    units_cfg = cfg.get("units", {})

    sub = ZeroMQSubscriber(
        camera_endpoint=io_cfg.get("camera_endpoint", "tcp://127.0.0.1:5555"),
        rtabmap_endpoint=io_cfg.get("rtabmap_endpoint", "tcp://127.0.0.1:6000"),
        ingest_queue=ingest_q,
        depth_m_per_unit=float(units_cfg.get("depth_m_per_unit", 0.001)),
        pose_m_per_unit=float(units_cfg.get("pose_m_per_unit", 1.0)),
        # Visualization callbacks (None if disabled)
        on_kf_packet=vis_server.handle_kf_packet if vis_server else None,
        on_kf_pose_update=vis_server.handle_kf_pose_update if vis_server else None,
    )
    t = threading.Thread(target=sub.run_forever, daemon=True)
    t.start()
    logger.info(f"ZeroMQ dual-socket subscriber started (camera + RTABMap)")

    # Start visualization server if enabled
    if vis_server:
        vis_server.start()
        logger.info(f"Visualization WebSocket server started on port {vis_cfg.get('port', 8081)}")

    pipe = Pipeline(
        cfg=cfg, 
        fastsam=fastsam, 
        clip=clip, 
        working_mem=wm, 
        proximity_index=proximity_index, 
        associator=assoc, 
        ingest_gate=ingest_gate, 
        vocab_clf=vocab_clf, 
        vectors=vectors, 
        ingest_q=ingest_q, 
        sweep_cache=sweep_cache )

    # ---------------- Start FastAPI control-plane ----------------
    api_cfg = cfg.get("api", {})
    host = str(api_cfg.get("host", "0.0.0.0"))
    port = int(api_cfg.get("port", 8000))

    # Components that can be reset without restarting RTSM
    reset_components = ResetComponents(
        sweep_cache=sweep_cache,
        frame_window=sub.fw,  # FrameWindow created inside ZeroMQSubscriber
        vis_server=vis_server,
    )

    app = create_app(
        working_memory=wm,
        clip_adapter=clip,
        vectors=vectors,
        extra_stats_provider=lambda: {
            "ingest_q": ingest_q.qsize(),
        },
        reset_components=reset_components,
    )
    start_server(app, host=host, port=port)
    logger.info(f"FastAPI server started on http://{host}:{port}")

    print("=" * 60)
    print("  RTSM is running! Waiting for data...")
    print(f"  Camera:  {io_cfg.get('camera_endpoint', 'tcp://127.0.0.1:5555')}")
    print(f"  RTABMap: {io_cfg.get('rtabmap_endpoint', 'tcp://127.0.0.1:6000')}")
    print(f"  API:     http://{host}:{port}")
    if vis_server:
        vis_port = vis_cfg.get("port", 8081)
        print(f"  Vis WS:  ws://0.0.0.0:{vis_port}/ws")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    pipe.run_forever()