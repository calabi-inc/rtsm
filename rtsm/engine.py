"""
The engine factory (Gate 4.5 plan, P3 task 2): the one place that turns a
config into loaded models and a runtime (memory, index, gate, queue,
pipeline). ``rtsm/run.py`` and the eval runner (``rtsm/evaluation/runner.py``)
both build through it, so an offline evaluation runs the same objects the
live runner does. The construction below is ``run.py``'s, moved verbatim
(2026-09-27); the G1-B anchor through ``run.py`` is the proof it did not
change.

Two steps because the eval runner repeats a run N times on ONE set of loaded
models: ``load_models`` (segmenter + CLIP + vocabulary classifier, seconds
to tens of seconds) and ``build_runtime`` (everything else, milliseconds).

``rtsm/demo.py`` still constructs its own objects (it builds the proximity
index without the per-cell caps and the sweep cache from the demo base):
switching it is a separate, gated change.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from rtsm.cfg import cfg_path
from rtsm.core.association import Associator
from rtsm.core.clock import Clock
from rtsm.core.ingest_gate import IngestGate
from rtsm.evaluation.event_log import EventLogWriter
from rtsm.io.ingest_lanes import LaneConfig, lane_drop_handler, make_ingest_queue
from rtsm.models.segmentation import get_segmenter
from rtsm.stores.proximity_index import GridSpec, ProximityIndex
from rtsm.stores.sweep_cache import SweepCache
from rtsm.stores.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


@dataclass
class Models:
    segmenter: Any
    clip: Any
    vocab_clf: Any


@dataclass
class Runtime:
    proximity_index: ProximityIndex
    wm: WorkingMemory
    analytics: Any                     # rtsm.analytics.AnalyticsBundle (.seg, .latency, .ticker, .start/.stop)
    associator: Associator
    ingest_gate: IngestGate
    vectors: Any                       # FaissClient | MilvusClient | None
    ingest_q: Any                      # IngestLanes | IngestQueue (make_ingest_queue)
    sweep_cache: SweepCache
    event_log: EventLogWriter
    clock: Clock
    pipeline: Any                      # rtsm.core.pipeline.Pipeline
    up_axis: str
    sessions: Any = None               # rtsm.core.sessions.SessionRegistry | None


def _warmup_frame(width: int, height: int):
    """A synthetic RGB frame, flat rectangles on grey: something a segmenter returns masks on."""
    import cv2
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    frame = np.full((height, width, 3), 96, np.uint8)
    for _ in range(24):
        x0, y0 = int(rng.integers(0, max(1, width - 40))), int(rng.integers(0, max(1, height - 40)))
        x1, y1 = x0 + int(rng.integers(30, max(31, width // 3))), y0 + int(rng.integers(30, max(31, height // 3)))
        cv2.rectangle(frame, (x0, y0), (x1, y1), tuple(int(c) for c in rng.integers(0, 256, 3)), -1)
    return Image.fromarray(frame)


def warm_segmenter(segmenter: Any, cfg: dict) -> Optional[dict]:
    """The segmenter's first inference, on a synthetic frame, before a source starts.

    A backend's ``warmup()`` loads weights. Its first real call still builds the library's inference backend,
    starts CUDA and cuDNN and, for an exported model, loads the engine: seconds to tens of seconds that
    otherwise fall on the first frame of a session, while the source keeps delivering. Two calls here (the
    second is the steady state) move that to start-up. Nothing of it reaches the memory or a ledger.
    A failure is logged and left to the first frame; ``segmentation.warmup_inference: false`` skips the call.
    Returns {"ms": .., "masks": ..}, or None when skipped or failed.
    """
    if getattr(segmenter, "consumes_detections", False):     # the external backend runs no model of its own
        return None
    import time
    import torch
    from rtsm.models.segmentation import segmentation_vocab

    cam = cfg.get("camera", {}) or {}
    try:
        frame = _warmup_frame(int(cam.get("width") or 640), int(cam.get("height") or 480))
        vocab = segmentation_vocab(cfg)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(2):
                out = segmenter.segment(frame, vocab=vocab)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        info = {"ms": round((time.perf_counter() - t0) * 1e3), "masks": int(getattr(out, "count", 0) or 0)}
        logger.info(f"Segmenter warmed up on a synthetic frame: {info['ms']} ms, {info['masks']} masks")
        return info
    except Exception:  # noqa: BLE001 -- the first frame then pays the start-up, as it did before
        logger.exception("Segmenter warm-up inference failed; the first frame will carry the model start-up")
        return None


def warm_embedder(clip: Any, vocab_clf: Any, cfg: dict) -> Optional[dict]:
    """The image tower's and the vocabulary classifier's first calls, on noise crops, before a source starts:
    one batch of the size the pipeline embeds per frame (``staging.topk_preclip``) and one of half the batch
    size. Same contract as warm_segmenter."""
    import time
    import numpy as np
    import torch

    try:
        batch = int(cfg.get("clip", {}).get("batch_size", 16))
        size = int(cfg.get("staging", {}).get("clip_input", 224))
        topk = int(cfg.get("staging", {}).get("topk_preclip", 15) or 15)
        rng = np.random.default_rng(0)
        crops = [rng.integers(0, 256, (size, size, 3), dtype=np.uint8) for _ in range(max(1, batch))]
        t0 = time.perf_counter()
        embs = None
        with torch.no_grad():
            for n in (max(1, min(topk, batch)), max(1, batch // 2)):
                embs = clip.encode_images_batch(crops[:n], batch_size=batch)
            if vocab_clf is not None and embs is not None and embs.numel() > 0:
                vocab_clf.ensure_text_on_device()
                vocab_clf.classify_feats(embs, return_topk=int(cfg.get("clip", {}).get("vocab_topk", 5)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        info = {"ms": round((time.perf_counter() - t0) * 1e3)}
        logger.info(f"Image tower and vocabulary classifier warmed up on noise crops: {info['ms']} ms")
        return info
    except Exception:  # noqa: BLE001
        logger.exception("Image tower warm-up inference failed; the first frame will carry the model start-up")
        return None


def load_models(cfg: dict) -> Models:
    """Segmenter (warmed up), CLIP adapter, CLIP vocabulary classifier -- the
    GPU-heavy objects, loaded once. Each model also runs its first inference
    here, on synthetic input (warm_segmenter / warm_embedder), so that the
    first frame of a session is not the first call of every model; the
    segmenter goes first, while the process holds the least memory."""
    from rtsm.models.clip.adapter import CLIPAdapter
    from rtsm.models.clip.vocab_classifier import ClipVocabClassifier

    warm = bool(cfg.get("segmentation", {}).get("warmup_inference", True))
    segmenter = get_segmenter(cfg)
    logger.info(f"Segmentation backend created: {segmenter.name}")
    segmenter.warmup()
    if warm:
        warm_segmenter(segmenter, cfg)
    logger.info(f"Segmentation models loaded and ready: {segmenter.name}")

    clip_cfg = cfg.get("clip", {})
    clip_model = clip_cfg.get("model", "ViT-B-32")
    clip_pretrained = clip_cfg.get("pretrained", "openai")
    clip_local = clip_cfg.get("local_dir", "model_store/clip")
    clip = CLIPAdapter(clip_model, clip_pretrained, clip_local, device=cfg.get("device", "cuda"))
    logger.info(f"CLIP model loaded: {clip_model} ({clip_pretrained})")
    vocab_clf = ClipVocabClassifier(clip.artifacts.model, clip.artifacts.tokenizer, clip.artifacts.preprocess,
                                    str(cfg_path("clip/vocab.yaml")), device=cfg.get("device", "cuda"))
    logger.info(f"CLIP vocabulary classifier successfully initialized")
    if warm:
        warm_embedder(clip, vocab_clf, cfg)
    return Models(segmenter=segmenter, clip=clip, vocab_clf=vocab_clf)


def build_runtime(cfg: dict, models: Models, *, clock: Clock, lane_cfg: LaneConfig, event_log: EventLogWriter,
                  up_axis_default: str = "z", sessions: Any = None) -> Runtime:
    """Proximity index, working memory, analytics bundle, associator, ingest
    gate, vector store, ingest queue (with the lane drop handler wired to the
    event log and the analytics), sweep cache, and the pipeline over them.
    ``up_axis_default``: "y" for ARKit sources (websocket / replay / Lens
    bags), "z" for D435i / ROS; ``sweep_cache.up_axis`` overrides."""
    from rtsm.analytics import build_analytics
    from rtsm.core.pipeline import Pipeline

    # Proximity index config
    scfg = cfg.get("sweep_cache", {})
    two_d = bool(scfg.get("two_d", True))
    cell_m = float(scfg.get("grid_size_m", 0.25))
    per_cell_cap = int(scfg.get("per_cell_cap", 64))
    neighbors_max = int(scfg.get("neighbors_max", 128))
    up_axis = str(scfg.get("up_axis", up_axis_default))
    pi_grid = GridSpec(cell_m=cell_m, use_3d=not two_d, up_axis=up_axis)
    proximity_index = ProximityIndex(pi_grid, per_cell_cap=per_cell_cap, neighbors_max=neighbors_max)
    logger.info(f"Proximity index successfully initialized")
    wm = WorkingMemory(cfg, index=proximity_index, clock=clock)
    logger.info(f"Working memory successfully initialized")
    # Runtime analytics: the two buffers + the AnalyticsTicker (the ONE owner of
    # the 1 Hz rollup); the caller starts it right before the pipeline loop so
    # its late_ticks / stale_rollups measure the run, not the model loads.
    # Below WorkingMemory because the ticker snapshots wm.stats() each tick.
    analytics = build_analytics(cfg, wm=wm)
    seg_analytics = analytics.seg
    latency_analytics = analytics.latency
    if analytics.enabled:
        logger.info(f"Runtime analytics initialized (retention={analytics.retention_s}s, buffer={analytics.buffer_frames} frames; rollup owner = analytics ticker)")
    assoc = Associator(cfg)
    ingest_gate = IngestGate(cfg)
    logger.info(f"Ingest gate successfully initialized")
    vec_cfg = cfg.get("vectors", {})
    backend = str(vec_cfg.get("backend", "faiss")).lower()
    vectors = None
    if bool(vec_cfg.get("enable", True)):
        if backend == "milvus":
            from rtsm.stores.vectors.milvus_client import MilvusClient
            vectors = MilvusClient(cfg)
        else:
            from rtsm.stores.vectors.faiss_client import FaissClient
            vectors = FaissClient(cfg)
            vs = vectors.stats()
            logger.info(
                f"Faiss vectors initialized (dim={vs['dim']}, "
                f"loaded={vs['count']}, persist={vs['persist_path']})"
            )

    # Ingest plumbing (intrinsics are per frame, from the source)
    ingest_q = make_ingest_queue(lane_cfg)
    sweep_cache = SweepCache(
        grid_size_m=float(cfg.get("sweep_cache", {}).get("grid_size_m", 0.25)),
        per_cell_cap=int(cfg.get("sweep_cache", {}).get("per_cell_cap", 64)),
        neighbors_max=int(cfg.get("sweep_cache", {}).get("neighbors_max", 128)),
        two_d=bool(cfg.get("sweep_cache", {}).get("two_d", True)),
        yaw_bins=int(cfg.get("sweep_cache", {}).get("yaw_bins", 12)),
        pitch_bins=int(cfg.get("sweep_cache", {}).get("pitch_bins", 5)),
        pitch_deg=float(cfg.get("sweep_cache", {}).get("pitch_deg", 60.0)),
        look_lru_keep=int(cfg.get("sweep_cache", {}).get("look_lru_keep", 8)),
        up_axis=up_axis,
    )
    logger.info(f"Sweep cache successfully initialized")
    # Lane-side drops (superseded / kf_dropped / age under policy latest) ->
    # trace lines with source "lanes" + the analytics counters.
    ingest_q.set_on_drop(lane_drop_handler(event_log.sink(), latency_analytics, ingest_q))

    pipeline = Pipeline(
        cfg=cfg,
        segmenter=models.segmenter,
        clip=models.clip,
        working_mem=wm,
        proximity_index=proximity_index,
        associator=assoc,
        ingest_gate=ingest_gate,
        vocab_clf=models.vocab_clf,
        vectors=vectors,
        ingest_q=ingest_q,
        sweep_cache=sweep_cache,
        event_log=event_log,
        clock=clock,
        seg_analytics=seg_analytics,
        latency_analytics=latency_analytics,
        sessions=sessions,
    )
    return Runtime(proximity_index=proximity_index, wm=wm, analytics=analytics, associator=assoc, ingest_gate=ingest_gate,
                   vectors=vectors, ingest_q=ingest_q, sweep_cache=sweep_cache, event_log=event_log, clock=clock,
                   pipeline=pipeline, up_axis=up_axis, sessions=sessions)
