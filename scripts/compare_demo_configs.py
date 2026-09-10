#!/usr/bin/env python3
"""
Compare two demo configs by running the RTSM pipeline on the demo clip.

Collects: confirmed objects, proto objects, mean frame latency, total time.

Usage:
    python scripts/compare_demo_configs.py
"""

from __future__ import annotations

import os
import sys
import time
import yaml
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Suppress noisy loggers during comparison
for name in [
    "rtsm", "rtsm.core", "rtsm.models", "rtsm.stores", "rtsm.io",
    "rtsm.visualization", "rtsm.analytics", "rtsm.utils",
]:
    logging.getLogger(name).setLevel(logging.WARNING)


def run_config(config_name: str, replay_dir: str) -> dict:
    """Run pipeline with a config and return metrics."""
    from rtsm.cfg import load_config, cfg_path
    from rtsm.models.segmentation import get_segmenter
    from rtsm.models.clip.adapter import CLIPAdapter
    from rtsm.models.clip.vocab_classifier import ClipVocabClassifier
    from rtsm.stores.working_memory import WorkingMemory
    from rtsm.stores.proximity_index import ProximityIndex, GridSpec
    from rtsm.core.association import Associator
    from rtsm.core.ingest_gate import IngestGate
    from rtsm.stores.sweep_cache import SweepCache
    from rtsm.io.ingest_lanes import LaneConfig, make_ingest_queue
    from rtsm.io.replayer import ReplayReceiver
    from rtsm.core.pipeline import Pipeline

    cfg = load_config(config_name)

    # Init components
    segmenter = get_segmenter(cfg)
    segmenter.warmup()

    clip = CLIPAdapter("ViT-B-32", "openai", "model_store/clip", device=cfg.get("device", "cuda"))
    vocab_clf = ClipVocabClassifier(
        clip.artifacts.model, clip.artifacts.tokenizer, clip.artifacts.preprocess,
        str(cfg_path("clip/vocab.yaml")), device=cfg.get("device", "cuda")
    )

    io_cfg = cfg.get("io", {})
    receiver_type = str(io_cfg.get("receiver", "websocket")).lower()
    up_axis = "y" if receiver_type == "websocket" else "z"

    scfg = cfg.get("sweep_cache", {})
    pi_grid = GridSpec(
        cell_m=float(scfg.get("grid_size_m", 0.25)),
        use_3d=not bool(scfg.get("two_d", True)),
        up_axis=up_axis,
    )
    proximity_index = ProximityIndex(pi_grid)
    # Same ingest clock as `python -m rtsm --replay` (ingest.clock auto -> sensor
    # under replay), so A/B comparisons here use the runner's admission policy.
    from rtsm.core.clock import make_clock, resolve_clock_mode
    clock_mode = resolve_clock_mode((cfg.get("ingest") or {}).get("clock", "auto"), replay=True)
    clock = make_clock(clock_mode)
    wm = WorkingMemory(cfg, index=proximity_index, clock=clock)
    assoc = Associator(cfg)
    ingest_gate = IngestGate(cfg)

    vec_cfg = cfg.get("vectors", {})
    vectors = None
    if bool(vec_cfg.get("enable", True)):
        from rtsm.stores.vectors.faiss_client import FaissClient
        vectors = FaissClient(cfg)

    ingest_q = make_ingest_queue(LaneConfig.from_cfg(cfg, replay=True))   # auto -> lossless under replay
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
        clock=clock,
    )

    # Start replay
    ws_cfg = io_cfg.get("websocket", {})
    vis_cfg = cfg.get("visualization", {})
    replay = ReplayReceiver(
        recording_dir=replay_dir,
        ingest_queue=ingest_q,
        require_tracking_normal=bool(ws_cfg.get("require_tracking_normal", True)),
        keyframe_every_n=int(ws_cfg.get("keyframe_every_n", 30)),
        nonkf_min_interval_s=float(ws_cfg.get("nonkf_min_interval_s", 0.5)),
        pose_sink=wm.update_robot_pose,   # receive-time pose (the pipeline no longer writes it)
        confidence_threshold=int(ws_cfg.get("confidence_threshold", 1)),
        apply_camera_flip=bool(vis_cfg.get("apply_camera_flip", False)),
        throttle_clock=clock_mode,
    )

    replay.start()

    # Process all frames: run_one_step blocks on queue.get() internally
    import threading
    t_start = time.monotonic()
    frames_processed = 0
    frame_times = []

    def run_pipeline():
        nonlocal frames_processed
        while True:
            if replay._done.is_set() and ingest_q.qsize() == 0:
                break
            t0 = time.monotonic()
            try:
                pipe.run_one_step()
                frame_times.append(time.monotonic() - t0)
                frames_processed += 1
            except Exception:
                break

    pipe_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipe_thread.start()

    # Wait for replay to complete, then give pipeline time to drain. Under
    # ingest.policy=lossless the replayer BLOCKS when the FIFO is full, so a
    # pipeline thread that died would leave replay.wait() hanging forever:
    # poll with a bound, and close the queue (wakes the producer) if the
    # consumer is gone or on any exit.
    try:
        while not replay.wait(timeout=1.0):
            if not pipe_thread.is_alive():
                ingest_q.close()
                raise RuntimeError("pipeline thread exited before the replay finished (see log above)")
        time.sleep(1.0)  # let pipeline drain remaining frames
    finally:
        pipe.stop()
        ingest_q.close()
    pipe_thread.join(timeout=5.0)

    total_time = time.monotonic() - t_start

    # Collect WM stats
    all_objs = list(wm.iter_objects())
    confirmed = sum(1 for o in all_objs if o.confirmed)
    proto = sum(1 for o in all_objs if not o.confirmed)
    total_objects = len(all_objs)

    mean_latency = sum(frame_times) / len(frame_times) if frame_times else 0

    return {
        "config": config_name,
        "frames_processed": frames_processed,
        "confirmed": confirmed,
        "proto": proto,
        "total_objects": total_objects,
        "mean_latency_ms": round(mean_latency * 1000, 1),
        "total_time_s": round(total_time, 1),
    }


def main():
    replay_dir = "recordings/demo_clip"
    if not os.path.isdir(replay_dir):
        print(f"ERROR: Demo clip not found at {replay_dir}")
        sys.exit(1)

    configs = ["demo_config_a.yaml", "demo_config_b.yaml"]
    results = []

    for cfg_name in configs:
        print(f"\n{'='*60}")
        print(f"  Running: {cfg_name}")
        print(f"{'='*60}")
        r = run_config(cfg_name, replay_dir)
        results.append(r)
        print(f"  Confirmed: {r['confirmed']}, Proto: {r['proto']}, Total: {r['total_objects']}")
        print(f"  Frames: {r['frames_processed']}, Mean latency: {r['mean_latency_ms']}ms")
        print(f"  Total time: {r['total_time_s']}s")

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'Config A':<15} {'Config B':<15}")
    print(f"  {'-'*55}")
    for key in ["confirmed", "proto", "total_objects", "frames_processed", "mean_latency_ms", "total_time_s"]:
        a = results[0][key]
        b = results[1][key]
        winner = ""
        if key in ("confirmed", "frames_processed"):
            winner = " <-" if a >= b else ""
        elif key in ("mean_latency_ms", "total_time_s", "proto"):
            winner = " <-" if a <= b else ""
        print(f"  {key:<25} {str(a):<15} {str(b):<15}{winner}")


if __name__ == "__main__":
    main()
