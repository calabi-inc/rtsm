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


def load_models(cfg: dict) -> Models:
    """Segmenter (warmed up), CLIP adapter, CLIP vocabulary classifier -- the
    GPU-heavy objects, loaded once."""
    from rtsm.models.clip.adapter import CLIPAdapter
    from rtsm.models.clip.vocab_classifier import ClipVocabClassifier

    segmenter = get_segmenter(cfg)
    logger.info(f"Segmentation backend created: {segmenter.name}")
    segmenter.warmup()
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
    return Models(segmenter=segmenter, clip=clip, vocab_clf=vocab_clf)


def build_runtime(cfg: dict, models: Models, *, clock: Clock, lane_cfg: LaneConfig, event_log: EventLogWriter,
                  up_axis_default: str = "z") -> Runtime:
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
    )
    return Runtime(proximity_index=proximity_index, wm=wm, analytics=analytics, associator=assoc, ingest_gate=ingest_gate,
                   vectors=vectors, ingest_q=ingest_q, sweep_cache=sweep_cache, event_log=event_log, clock=clock,
                   pipeline=pipeline, up_axis=up_axis)
