from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import List, Optional, Dict, Any, Tuple
import time
import numpy as np
import torch
import cv2
from PIL import Image

from rtsm.utils.mask_staging import run_heuristics, MaskStats, FilterDiagnostics
from rtsm.utils.prepare_ann import prepare_ann
from rtsm.utils.periodic_logger import PeriodicLogger
from rtsm.models.segmentation import SegmentationAdapter, SegmentationResult
from rtsm.models.clip.adapter import CLIPAdapter
from rtsm.models.clip.vocab_classifier import ClipVocabClassifier
from rtsm.stores.working_memory import WorkingMemory
from rtsm.stores.proximity_index import ProximityIndex
from rtsm.core.association import Associator
from rtsm.core.ingest_gate import IngestGate
from rtsm.stores.sweep_cache import SweepCache
from rtsm.io.ingest_queue import IngestQueue
from rtsm.core.datamodel import FramePacket
from rtsm.evaluation.event_log import EventLogWriter, FrameEvent, summarize_sources

logger = logging.getLogger(__name__)


@dataclass
class Snapshot:
    rgb: np.ndarray               # HxWx3 uint8
    depth_m: Optional[np.ndarray] # HxW float32 (NaN for invalid)
    intrinsics: Dict[str,float]   # {fx,fy,cx,cy}
    pose_cam_T_world: Optional[np.ndarray]  # 4x4 float32

@dataclass
class Candidate:
    idx: int
    mask: torch.Tensor            # [H,W] bool
    stats: MaskStats
    priority: float               # computed priority score
    crop: Optional[np.ndarray] = None    # 224x224x3 uint8 after pre-clip
    emb_vis: Optional[np.ndarray] = None # CLIP visual embedding (L2-normalized)
    # Judgment crop (2026-08-30): native-resolution, UNMASKED, generously
    # padded cut of the same detection — for the stored snapshot gallery
    # that VLM selection and operator eyeballing consume. The 224 masked
    # `crop` above is an EMBEDDING input (background suppressed so CLIP
    # focuses on the object) and was never fit for judgment: the L1 pilot
    # showed the arbiter rubber-stamping wrong objects it could not
    # actually identify at that size, with the mean-fill background
    # destroying all context.
    crop_hires: Optional[np.ndarray] = None


def cut_judgment_crop(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                      pad_frac: float, max_px: int) -> Optional[np.ndarray]:
    """Unmasked judgment crop (2026-08-30): pure function, unit-tested.

    Cuts the bbox (already in RGB pixel space) from the full frame with
    context padding of pad_frac × the box's own size on every side,
    clamped to the frame; downscales with INTER_AREA only when the long
    side exceeds max_px. Returns None for degenerate boxes. The result
    is what the snapshot gallery stores — the crop VLM selection and
    operator eyeballing consume (the masked 224 embedding crop was never
    fit for judgment)."""
    H, W = rgb.shape[:2]
    bw, bh = x1 - x0, y1 - y0
    if bw <= 0 or bh <= 0:
        return None
    jx0 = max(0, int(x0 - bw * pad_frac))
    jy0 = max(0, int(y0 - bh * pad_frac))
    jx1 = min(W, int(x1 + bw * pad_frac))
    jy1 = min(H, int(y1 + bh * pad_frac))
    if jx1 <= jx0 or jy1 <= jy0:
        return None
    hires = rgb[jy0:jy1, jx0:jx1].copy()
    long_side = max(hires.shape[0], hires.shape[1])
    if long_side > max_px:
        scale = max_px / float(long_side)
        hires = cv2.resize(
            hires,
            (max(1, int(hires.shape[1] * scale)),
             max(1, int(hires.shape[0] * scale))),
            interpolation=cv2.INTER_AREA)
    return hires


@dataclass
class ScoringTraceEntry:
    """Per-candidate scoring trace row. Captured before dedup/topK so the full
    distribution is visible offline (sweep tuning needs to see what almost
    survived, not just what did)."""
    mask_idx: int
    priority: float
    coverage: float
    border_fraction: float
    depth_valid: float
    depth_spread: float
    bbox_area_norm: float
    confirmation_source: Optional[str]   # "dual" | "yoloe_only" | "fastsam_only" | None
    selected_topk: bool                  # True for candidates that survived the top-K cut


@dataclass
class ScoringTrace:
    n_candidates: int                    # total scored (pre-dedup)
    n_selected: int                      # actually returned by _score_and_select (post-dedup, capped at topK)
    entries: List[ScoringTraceEntry]

class Pipeline:
    def __init__(
        self,
        cfg: Dict[str,Any],
        segmenter: SegmentationAdapter,
        clip: CLIPAdapter,
        working_mem: WorkingMemory,
        proximity_index: ProximityIndex,
        associator: Associator,
        ingest_gate: IngestGate,
        vocab_clf: Optional[ClipVocabClassifier] = None,
        vectors: Optional[Any] = None,
        ingest_q: Optional[IngestQueue] = None,
        sweep_cache: Optional[SweepCache] = None,
        seg_analytics: Optional[Any] = None,
        latency_analytics: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.segmenter = segmenter
        self.clip = clip
        self.working_mem = working_mem
        self.proximity_index = proximity_index
        self.associator = associator
        self.ingest_gate = ingest_gate
        self.vocab_clf = vocab_clf
        self.vectors = vectors  # generic vector store client (FAISS or Milvus)
        self._running = False
        self._last_flush_ts = 0.0
        self._vector_flush_failures = 0
        self.ingest_q = ingest_q
        self.sweep_cache = sweep_cache or SweepCache()
        self._seg_analytics = seg_analytics
        self._latency_analytics = latency_analytics

        # Periodic summary logger
        log_cfg = cfg.get("logging", {})
        self._periodic_logger = PeriodicLogger(
            interval_s=float(log_cfg.get("summary_interval_s", 5.0)),
            enabled=bool(log_cfg.get("periodic_summary", True)),
        )

        # Phase 0 diagnostics: per-frame JSONL event log (off by default).
        # See rtsm/evaluation/event_log.py for path resolution rules.
        diag_cfg = cfg.get("diagnostics", {})
        self._event_log = EventLogWriter(
            enabled=bool(diag_cfg.get("enabled", False)),
            configured_path=diag_cfg.get("event_log_path"),
        )
        # Always-present attrs (populated each frame when set; default to None)
        self._last_filter_diagnostics: Optional[FilterDiagnostics] = None
        self._last_scoring_trace: Optional[ScoringTrace] = None

    # -------- public entrypoints --------
    def run_forever(self):
        logger.info(f"--------------------------------")
        logger.info(f"Pipeline started")
        logger.info(f"--------------------------------")
        self._running = True
        try:
            while self._running:
                self.run_one_step()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def stop(self):
        self._running = False
        
    # -------- single step (one snapshot) --------
    @torch.no_grad()
    def run_one_step(self):
        """
        Main processing loop for a single pipeline step:
        1. Grab the next available snapshot (RGB, depth, intrinsics, pose) from sensors.
        2. Run FastSAM segmentation on the RGB image to get instance masks.
        3. Convert segmentation output to boolean mask tensor [N,H,W].
        4. Apply mask heuristics (area, border, depth, planarity, etc.) to filter and annotate masks.
        5. Score and select the top-K mask candidates for further processing.
        6. For each candidate, crop the corresponding region from the RGB image for CLIP.
        7. Batch-encode crops with CLIP to obtain visual embeddings.
        8. If associator, working memory, and index are present, update them with the new candidates.
        9. Periodically flush new vectors to Milvus (if configured).
        This function is called repeatedly in the main loop to process incoming frames.
        """
        snap, pkt = self._get_snapshot_via_queue()
        if snap is None:
            time.sleep(0.01)
            return

        # Ingest gate: accept keyframes unconditionally, gate non-KFs with policy
        accept = True
        try:
            if pkt is not None and pkt.pose is not None:
                # compute cell/vbin from pose
                twc = pkt.pose.t_wc.astype(np.float32)
                q = pkt.pose.q_wc_xyzw.astype(np.float32)
                cell, vbin, fwd = self.sweep_cache.cell_and_vbin_from_pose(twc_xyz=twc, q_wc_xyzw=q)
                H, W = snap.rgb.shape[:2]
                Z = None
                if snap.depth_m is not None:
                    dH, dW = snap.depth_m.shape[:2]
                    zc = float(snap.depth_m[dH//2, dW//2])
                    if np.isfinite(zc) and zc > 0:
                        Z = zc
                ts_ns = int(pkt.time.t_sensor_ns or 0)
                if pkt.is_keyframe:
                    # Update gate's keyframe arrival bookkeeping (grace window) but always accept
                    _ = self.ingest_gate.should_accept(
                        is_keyframe=True,
                        ts_ns=ts_ns,
                        sweep_cache=self.sweep_cache,
                        cell=cell,
                        vbin=vbin,
                        cam_pos=twc,
                        fwd_unit=fwd,
                        Z=Z,
                        look_cell=None,
                        now_mono=time.monotonic(),
                    )
                    accept = True
                else:
                    dec = self.ingest_gate.should_accept(
                        is_keyframe=False,
                        ts_ns=ts_ns,
                        sweep_cache=self.sweep_cache,
                        cell=cell,
                        vbin=vbin,
                        cam_pos=twc,
                        fwd_unit=fwd,
                        Z=Z,
                        look_cell=None,
                        now_mono=time.monotonic(),
                    )
                    accept = bool(getattr(dec, 'accept', True))
            else:
                accept = True
        except Exception as e:
            logger.warning(f"ingest gate error; proceeding: {e}")
            accept = True
        if not accept:
            if self._latency_analytics:
                try:
                    self._latency_analytics.record_gate_rejection()
                except Exception:
                    pass
            return

        t_step_start = time.perf_counter()

        # 1) segmentation -> masks
        # Ingest frames are BGR by contract (io/websocket.decode_rgb); PIL
        # convention is RGB, and every PIL-consuming backend (GDINO, YOLOE,
        # SAM2 preprocessors) assumes it. Flip ONCE at this boundary —
        # feeding BGR-as-RGB silently degraded detection for months
        # (found 2026-08-15: a red Coca-Cola can stored as blue).
        pil_img = (Image.fromarray(snap.rgb[..., ::-1])
                   if isinstance(snap.rgb, np.ndarray) else snap.rgb)

        # Get segmentation vocab from config (for open-vocab models)
        seg_cfg = self.cfg.get("segmentation", {})
        backend = seg_cfg.get("backend", "fastsam")
        if backend == "dual":
            vocab = seg_cfg.get("yoloe", {}).get("vocab", None)
        elif backend == "yoloe":
            vocab = seg_cfg.get("yoloe", {}).get("vocab", None)
        elif backend == "grounded_sam2":
            vocab = seg_cfg.get("grounded_sam2", {}).get("vocab", None)
        else:
            vocab = None

        # Run segmentation
        t_seg_start = time.perf_counter()
        seg_result = self.segmenter.segment(pil_img, vocab=vocab)
        t_seg_end = time.perf_counter()
        # Store for downstream priority boost and label propagation
        self._last_seg_result = seg_result
        # Detector-label monitor (2026-08-28): tally what the PROMPTED
        # detector claims to see this frame, before any association/
        # confirmation gate can hide it — served in /stats so operators
        # can separate "detector never saw it" from "memory never
        # surfaced it".
        if self.working_mem is not None and seg_result.labels:
            self.working_mem.note_label_detections(seg_result.labels,
                                                   seg_result.scores)

        # Extract masks - use prepare_ann for consistency with existing pipeline
        if seg_result.has_masks:
            ann_bool = prepare_ann(seg_result.masks)  # [N,H,W] bool CPU
        else:
            # Fallback for detection-only models (no masks)
            ann_bool = torch.empty(0, pil_img.height, pil_img.width, dtype=torch.bool)

        n_masks = int(ann_bool.shape[0]) if hasattr(ann_bool, 'shape') else 0

        # 2) heuristics -> keep + stats
        #    Overlay per-frame intrinsics so mask staging uses the actual camera,
        #    not the static YAML defaults (critical for WebSocket/ARKit path).
        heur_cfg = self.cfg
        if snap.intrinsics:
            heur_cfg = {**self.cfg, "camera": {**self.cfg.get("camera", {}), **snap.intrinsics}}
            # Also propagate width/height from the actual frame
            h_frame, w_frame = snap.rgb.shape[:2]
            heur_cfg["camera"]["width"] = w_frame
            heur_cfg["camera"]["height"] = h_frame
        # Safety: if mask resolution differs from RGB (e.g. retina_masks=False
        # or a different segmentation backend), scale intrinsics to mask space
        # so _centroid_cam back-projects with consistent coordinates.
        mask_hw = None
        if ann_bool.shape[0] > 0:
            mask_h, mask_w = ann_bool.shape[1], ann_bool.shape[2]
            rgb_h_frame, rgb_w_frame = snap.rgb.shape[:2]
            if mask_h != rgb_h_frame or mask_w != rgb_w_frame:
                mask_hw = (mask_h, mask_w)
                sx = mask_w / rgb_w_frame
                sy = mask_h / rgb_h_frame
                cam = heur_cfg.get("camera", {})
                heur_cfg = {**heur_cfg, "camera": {
                    **cam,
                    "fx": float(cam.get("fx", 0)) * sx,
                    "fy": float(cam.get("fy", 0)) * sy,
                    "cx": float(cam.get("cx", 0)) * sx,
                    "cy": float(cam.get("cy", 0)) * sy,
                    "width": mask_w,
                    "height": mask_h,
                }}
                logger.debug(
                    f"mask/RGB resolution mismatch ({mask_w}x{mask_h} vs "
                    f"{rgb_w_frame}x{rgb_h_frame}), scaled intrinsics to mask space"
                )

        # Compute mask→RGB scale factors for downstream coordinate conversion.
        # MaskStats (bbox, centroid_px) are in mask space; crop extraction and
        # association reprojection need RGB-space coordinates.
        mask_to_rgb_sx = 1.0
        mask_to_rgb_sy = 1.0
        if mask_hw is not None:
            rgb_h_frame, rgb_w_frame = snap.rgb.shape[:2]
            mask_to_rgb_sx = rgb_w_frame / mask_hw[1]  # width
            mask_to_rgb_sy = rgb_h_frame / mask_hw[0]  # height

        t_heur_start = time.perf_counter()
        kept_masks, stats, filter_diag = run_heuristics(
            ann_bool,
            snap.depth_m,
            cfg=heur_cfg,
        )
        t_heur_end = time.perf_counter()
        self._last_filter_diagnostics = filter_diag
        logger.debug(f"staging: kept {len(kept_masks)}/{n_masks} masks after heuristics")

        # Scale centroid_px from mask space to RGB space for downstream consumers
        # (association reprojection gate, dedup centroid distance).
        # MaskStats are ephemeral per-frame, safe to mutate.
        if mask_to_rgb_sx != 1.0 or mask_to_rgb_sy != 1.0:
            for s in stats:
                cpx = getattr(s, 'centroid_px', None)
                if cpx is not None:
                    s.centroid_px = (cpx[0] * mask_to_rgb_sx, cpx[1] * mask_to_rgb_sy)

        # 3) compute priorities (soft-features) and pick top-K
        #    Use mask resolution for frame_hw when masks differ from RGB
        select_hw = mask_hw if mask_hw else snap.rgb.shape[:2]
        t_score_start = time.perf_counter()
        cands = self._score_and_select(kept_masks, stats, frame_hw=select_hw)
        t_score_end = time.perf_counter()

        # 4) pre-CLIP crop only top-K, then batch encode
        t_clip_start = time.perf_counter()
        self._make_crops_inplace(cands, snap.rgb, mask_to_rgb_sx, mask_to_rgb_sy)
        self._clip_batch_inplace(cands)
        t_clip_end = time.perf_counter()

        # 5) associate and update memory/index (if components provided)
        t_assoc_start = time.perf_counter()
        m, c = 0, 0
        is_kf = bool(pkt.is_keyframe) if pkt is not None else False
        # Build frame_id for WM pose correction tracking (matches vis server mesh_id)
        frame_id = None
        if pkt is not None and pkt.time.seq is not None:
            frame_id = f"ws_{pkt.time.seq}"
        if self.associator is not None and self.working_mem is not None and self.proximity_index is not None:
            stats_assoc = self.associator.update_with_candidates(cands, snap, self.working_mem, self.proximity_index, is_keyframe=is_kf, frame_id=frame_id)
            try:
                if isinstance(stats_assoc, dict):
                    m = int(stats_assoc.get("matched", 0))
                    c = int(stats_assoc.get("created", 0))
                    logger.debug(f"assoc: matched={m} created={c}")
            except Exception:
                pass

        t_assoc_end = time.perf_counter()

        # ---- Analytics collection (never kills pipeline) ----
        try:
            if self._latency_analytics:
                from rtsm.analytics.latency_analytics import FrameTimingStats
                self._latency_analytics.append(FrameTimingStats(
                    timestamp=t_step_start,
                    frame_seq=int(pkt.time.seq or 0) if pkt else 0,
                    is_keyframe=is_kf,
                    t_segmentation=t_seg_end - t_seg_start,
                    t_heuristics=t_heur_end - t_heur_start,
                    t_scoring=t_score_end - t_score_start,
                    t_clip=t_clip_end - t_clip_start,
                    t_association=t_assoc_end - t_assoc_start,
                    t_total=t_assoc_end - t_step_start,
                    queue_depth=self.ingest_q.qsize() if self.ingest_q else 0,
                    n_masks_in=n_masks,
                    n_masks_staged=len(kept_masks),
                    n_candidates=len(cands),
                    assoc_matched=m,
                    assoc_created=c,
                ))
            if self._seg_analytics:
                from rtsm.analytics.seg_analytics import SegFrameStats
                seg_cfg = self.cfg.get("segmentation", {})
                backend = seg_cfg.get("backend", "fastsam")
                cs = seg_result.confirmation_source
                if cs is not None:
                    seg_entry = SegFrameStats(
                        timestamp=t_step_start,
                        frame_seq=int(pkt.time.seq or 0) if pkt else 0,
                        backend=backend,
                        n_dual=sum(1 for s in cs if s == "dual"),
                        n_fastsam_only=sum(1 for s in cs if s == "fastsam_only"),
                        n_yoloe_only=sum(1 for s in cs if s == "yoloe_only"),
                        n_total=len(cs),
                        n_fastsam_raw=seg_result.fastsam_raw_count or 0,
                        n_yoloe_raw=seg_result.yoloe_raw_count or 0,
                    )
                else:
                    n = seg_result.count
                    seg_entry = SegFrameStats(
                        timestamp=t_step_start,
                        frame_seq=int(pkt.time.seq or 0) if pkt else 0,
                        backend=backend,
                        n_fastsam_only=n if backend in ("fastsam", "sam2") else 0,
                        n_yoloe_only=n if backend in ("yoloe", "grounded_sam2") else 0,
                        n_total=n,
                    )
                # Fill staged/selected counts by confirmation source
                if cs is not None:
                    for s in stats:
                        if s.idx < len(cs):
                            src = cs[s.idx]
                            if src == "dual": seg_entry.staged_dual += 1
                            elif src == "fastsam_only": seg_entry.staged_fastsam_only += 1
                            elif src == "yoloe_only": seg_entry.staged_yoloe_only += 1
                    for cd in cands:
                        if cd.stats.idx < len(cs):
                            src = cs[cd.stats.idx]
                            if src == "dual": seg_entry.selected_dual += 1
                            elif src == "fastsam_only": seg_entry.selected_fastsam_only += 1
                            elif src == "yoloe_only": seg_entry.selected_yoloe_only += 1
                self._seg_analytics.append(seg_entry)
        except Exception:
            logger.debug("analytics collection error", exc_info=True)

        # Update periodic logger with this frame's stats
        self._periodic_logger.tick(matched=m, created=c, masks=len(kept_masks))

        # 6) expire stale proto-objects past their TTL
        if self.working_mem is not None:
            expired = self.working_mem.expire_timeouts()
            if expired:
                logger.debug(f"expired {len(expired)} proto-objects: {expired}")

        # 7) periodic flush/upsert to vector store (if configured)
        self._maybe_flush_vectors()

        # 8) periodic summary log
        if self.working_mem is not None:
            queue_size = self.ingest_q.qsize() if self.ingest_q else 0
            self._periodic_logger.maybe_log(
                wm_stats=self.working_mem.stats(),
                queue_size=queue_size,
            )

        # Ingest bookkeeping after heavy processing
        try:
            if pkt is not None and pkt.pose is not None:
                twc = pkt.pose.t_wc.astype(np.float32)
                q = pkt.pose.q_wc_xyzw.astype(np.float32)
                cell, vbin, fwd = self.sweep_cache.cell_and_vbin_from_pose(twc_xyz=twc, q_wc_xyzw=q)
                ts_ns = int(pkt.time.t_sensor_ns or 0)
                self.ingest_gate.record_processed(
                    is_keyframe=bool(pkt.is_keyframe),
                    ts_ns=ts_ns,
                    sweep_cache=self.sweep_cache,
                    cell=cell,
                    vbin=vbin,
                    cam_pos=twc,
                    look_cell=None,
                    now_mono=time.monotonic(),
                )
                # Store latest robot pose for API queries
                timestamp = float(pkt.time.t_wall_utc_s or pkt.time.t_mono_s or 0.0)
                self.working_mem.update_robot_pose(twc, q, timestamp)
        except Exception:
            pass

        # ---- Phase 0 diagnostic event log (no-op when disabled) ----
        if self._event_log.enabled:
            try:
                fd = self._last_filter_diagnostics
                st = self._last_scoring_trace
                filter_dict: Dict[str, Any] = {}
                if fd is not None:
                    filter_dict = {
                        "total_input": fd.total_input,
                        "dropped_area": fd.dropped_area,
                        "dropped_depth_valid": fd.dropped_depth_valid,
                        "dropped_depth_spread": fd.dropped_depth_spread,
                        "passed": fd.passed,
                    }
                    if fd.drop_details:
                        filter_dict["drop_details"] = fd.drop_details
                scoring_dict: Dict[str, Any] = {}
                if st is not None:
                    scoring_dict = {
                        "n_candidates": st.n_candidates,
                        "n_selected": st.n_selected,
                        "top_k_priorities": [e.priority for e in st.entries[:5]],
                        "source_breakdown": summarize_sources(st.entries),
                    }
                n_conf = 0
                if self.working_mem is not None:
                    try:
                        n_conf = int(self.working_mem.stats().get("confirmed", 0))
                    except Exception:
                        n_conf = 0
                self._event_log.write(FrameEvent(
                    timestamp=t_step_start,
                    frame_seq=int(pkt.time.seq or 0) if pkt is not None else 0,
                    is_keyframe=is_kf,
                    n_masks_raw=n_masks,
                    filter=filter_dict,
                    scoring=scoring_dict,
                    n_matched=m,
                    n_created=c,
                    n_objects_confirmed=n_conf,
                    timing_ms={
                        "segmentation": (t_seg_end - t_seg_start) * 1000.0,
                        "heuristics": (t_heur_end - t_heur_start) * 1000.0,
                        "scoring": (t_score_end - t_score_start) * 1000.0,
                        "clip": (t_clip_end - t_clip_start) * 1000.0,
                        "association": (t_assoc_end - t_assoc_start) * 1000.0,
                        "total": (t_assoc_end - t_step_start) * 1000.0,
                    },
                ))
            except Exception:
                # Never let diagnostics kill the pipeline
                logger.debug("event log write failed", exc_info=True)

    # -------- internals --------
    def _get_snapshot_via_queue(self) -> Tuple[Optional[Snapshot], Optional[FramePacket]]:
        if self.ingest_q is None:
            return None, None
        pkt = self.ingest_q.get(timeout=0.0)
        if pkt is None:
            return None, None
        rgb = pkt.rgb
        depth_m = pkt.depth_m
        intr = None
        if pkt.intr is not None:
            intr = {"fx": float(pkt.intr.fx), "fy": float(pkt.intr.fy), "cx": float(pkt.intr.cx), "cy": float(pkt.intr.cy)}
        # Convert pose to T_cam_world if available
        T = None
        try:
            if pkt.pose is not None:
                T_wc = pkt.pose.T_wc()
                # Note: ARKit→OpenCV camera flip is applied at ingestion
                # (WebSocketReceiver) so T_wc is already in OpenCV convention.
                # Debug: log T_wc (periodically)
                if not hasattr(self, '_pipe_pose_log_count'):
                    self._pipe_pose_log_count = 0
                self._pipe_pose_log_count += 1
                if self._pipe_pose_log_count % 30 == 1:
                    t_raw = pkt.pose.t_wc
                    t_from_matrix = T_wc[:3, 3]
                    logger.debug(f"[pipeline] pkt.pose.t_wc={t_raw}, T_wc[:3,3]={t_from_matrix}")
                # inverse to get T_cam_world (maps world → OpenCV camera frame)
                R = T_wc[:3, :3]
                t = T_wc[:3, 3]
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R.T
                T[:3, 3] = (-R.T @ t)
        except Exception:
            T = None
        return Snapshot(rgb=rgb, depth_m=depth_m, intrinsics=intr, pose_cam_T_world=T), pkt

    def _score_and_select(
        self, masks: List[torch.Tensor], stats: List[MaskStats],
        frame_hw: Optional[Tuple[int, int]] = None,
    ) -> List[Candidate]:
        """
        This function takes a list of candidate masks and their computed statistics,
        and assigns a "priority" score to each candidate based on a weighted sum of features.
        The features include:
          - coverage: fraction of image covered by the mask (higher is better, up to a soft cap)
          - border_fraction: fraction of mask pixels on the image border (lower is better)
          - depth_valid: fraction of mask pixels with valid depth (higher is better)
          - depth_spread: spread (p90-p10) of depth values within the mask (lower is better)
          - bbox_area: area of the mask's bounding box, normalized by image area (moderate is better)
          - planarity/structure: geometric planarity features (optional, if available)
        The weights and thresholds for these features are configurable via self.cfg.
        Oversized masks (too large coverage or bbox) are penalized.
        The function returns the top-K candidates (by priority), where K is configurable.

        Compute priority for each candidate and store as .priority
        (The actual calculation is performed in the loop below.)

        After all candidates are scored, select the top-K by priority.
        If there are fewer than K candidates, return all.
        The returned list is sorted in descending order of priority.
        """
        
        if frame_hw is not None:
            H, W = frame_hw
        else:
            H, W = self.cfg.get("camera",{}).get("height"), self.cfg.get("camera",{}).get("width")
        # weights from cfg with safe defaults
        w_cov   = float(self.cfg.get("staging",{}).get("w_coverage", 1.0))
        w_edge  = float(self.cfg.get("staging",{}).get("w_border_fraction", -2.0))
        w_valid = float(self.cfg.get("staging",{}).get("w_depth_valid", 1.0))
        w_spread= float(self.cfg.get("staging",{}).get("w_depth_spread", -1.0))
        bias_big_bbox = float(self.cfg.get("staging",{}).get("w_bbox_area", 0.5))
        w_struct = float(self.cfg.get("staging",{}).get("w_struct", 0.4))  # penalty
        # soft-cap thresholds and oversize penalties
        cov_cap = float(self.cfg.get("staging",{}).get("coverage_soft_cap", 0.30))
        bbox_cap = float(self.cfg.get("staging",{}).get("bbox_soft_cap", 0.30))
        w_cov_over = float(self.cfg.get("staging",{}).get("w_coverage_oversize", -2.0))
        w_bbox_over = float(self.cfg.get("staging",{}).get("w_bbox_oversize", -1.0))
        topK   = int(self.cfg.get("staging",{}).get("topk_preclip", 5))

        # structure score knobs (soft bias)
        a = float(self.cfg.get("staging",{}).get("struct_a", 0.6))
        b = float(self.cfg.get("staging",{}).get("struct_b", 0.3))  # CLIP term (unavailable pre-CLIP)
        c = float(self.cfg.get("staging",{}).get("struct_c", 0.1))
        inlier_min = float(self.cfg.get("staging",{}).get("struct_geom_inlier_pct_min", 0.80))
        rms_inv_scale = float(self.cfg.get("staging",{}).get("struct_geom_rms_inv_scale_m", 0.005))
        rms_smooth_hi = float(self.cfg.get("staging",{}).get("struct_geom_rms_hi_m", 0.02))
        extent_lo = float(self.cfg.get("staging",{}).get("struct_extent_lo", 0.10))
        extent_hi = float(self.cfg.get("staging",{}).get("struct_extent_hi", 0.35))

        def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return max(lo, min(hi, x))

        def _smoothstep(edge0: float, edge1: float, x: float) -> float:
            if edge1 == edge0:
                return 1.0 if x >= edge1 else 0.0
            t = _clamp((x - edge0) / (edge1 - edge0))
            return t * t * (3.0 - 2.0 * t)

        cands: List[Candidate] = []
        img_area = float(H * W)
        for i, (m, s) in enumerate(zip(masks, stats)):
            x0,y0,x1,y1 = s.bbox
            bbox_area = max(0, (x1-x0)) * max(0, (y1-y0))
            # normalize features
            cov = s.coverage                        # 0..1
            edge = s.border_fraction                # 0..1 (lower better)
            vld = s.depth_valid                     # 0..1
            spr = 0.0 if s.depth_spread is None else float(s.depth_spread)  # meters, ~[0..1+]
            spr_n = max(0.0, min(1.0, spr))         # clamp
            bbox_n = min(1.0, bbox_area / img_area)
            # oversize penalties
            oversize_cov = max(0.0, cov - cov_cap)
            oversize_bbox = max(0.0, bbox_n - bbox_cap)

            # soft structure score (pre-CLIP: geom + extent; semantic term deferred)
            s_geom = 0.0
            if s.planar_inlier_pct is not None and s.planar_rms_m is not None:
                if float(s.planar_inlier_pct) >= inlier_min:
                    inv = rms_inv_scale / (float(s.planar_rms_m) + 1e-6)
                    s_geom = _smoothstep(0.0, rms_smooth_hi, inv)
            s_extent = _smoothstep(extent_lo, extent_hi, cov)
            s_sem = 0.0  # no CLIP pre-topK
            structure_score = _clamp(a * s_geom + b * s_sem + c * s_extent)

            score = (
                (w_cov*cov) + (w_edge*edge) + (w_valid*vld) + (w_spread*spr_n)
                + (bias_big_bbox*bbox_n)
                + (w_cov_over*oversize_cov) + (w_bbox_over*oversize_bbox)
                - (w_struct * structure_score)
            )

            # Dual-confirmation priority adjustment: boost dual-confirmed,
            # penalize YOLOE-only (no FastSAM confirmation)
            seg = getattr(self, '_last_seg_result', None)
            if seg is not None and seg.confirmation_source is not None:
                dual_cfg = self.cfg.get("segmentation", {}).get("dual", {})
                src_idx = s.idx  # original mask index from segmentation output
                if src_idx < len(seg.confirmation_source):
                    src = seg.confirmation_source[src_idx]
                    if src == "dual":
                        score += float(dual_cfg.get("priority_boost_dual", 0.3))
                    elif src == "yoloe_only":
                        score += float(dual_cfg.get("priority_penalty_yoloe_only", -0.1))

            cands.append(Candidate(idx=i, mask=m, stats=s, priority=float(score)))

        # pick top-K by priority
        cands.sort(key=lambda c: c.priority, reverse=True)

        # Build scoring trace (pre-dedup snapshot of all candidates with their priorities).
        # selected_topk is filled in after dedup at the end of this function.
        seg = getattr(self, '_last_seg_result', None)
        seg_sources = (
            seg.confirmation_source
            if seg is not None and seg.confirmation_source is not None
            else None
        )
        trace_entries: List[ScoringTraceEntry] = []
        for cd in cands:
            s = cd.stats
            bbox_n = min(1.0, (max(0, s.bbox[2]-s.bbox[0]) * max(0, s.bbox[3]-s.bbox[1])) / img_area) if img_area > 0 else 0.0
            src: Optional[str] = None
            if seg_sources is not None and 0 <= s.idx < len(seg_sources):
                src = seg_sources[s.idx]
            trace_entries.append(ScoringTraceEntry(
                mask_idx=int(s.idx),
                priority=float(cd.priority),
                coverage=float(s.coverage),
                border_fraction=float(s.border_fraction),
                depth_valid=float(s.depth_valid),
                depth_spread=0.0 if s.depth_spread is None else float(s.depth_spread),
                bbox_area_norm=float(bbox_n),
                confirmation_source=src,
                selected_topk=False,  # filled in at the end after dedup
            ))

        # Scoring diagnostics: log all candidates ranked by priority
        if cands:
            logger.info(
                f"[scoring] {len(cands)} candidates ranked "
                f"(top={cands[0].priority:.3f}, bottom={cands[-1].priority:.3f})"
            )
            for rank, cd in enumerate(cands):
                s = cd.stats
                cov = s.coverage
                edge = s.border_fraction
                vld = s.depth_valid
                spr = 0.0 if s.depth_spread is None else float(s.depth_spread)
                bbox_n = min(1.0, (max(0, s.bbox[2]-s.bbox[0]) * max(0, s.bbox[3]-s.bbox[1])) / img_area)
                oc = max(0.0, cov - cov_cap)
                ob = max(0.0, bbox_n - bbox_cap)
                # recompute structure for log
                sg = 0.0
                if s.planar_inlier_pct is not None and s.planar_rms_m is not None:
                    if float(s.planar_inlier_pct) >= inlier_min:
                        sg = _smoothstep(0.0, rms_smooth_hi, rms_inv_scale / (float(s.planar_rms_m) + 1e-6))
                se = _smoothstep(extent_lo, extent_hi, cov)
                ss = _clamp(a * sg + b * 0.0 + c * se)
                logger.debug(
                    f"  #{rank} mask={s.idx} score={cd.priority:.3f} | "
                    f"cov={cov:.1%} edge={edge:.1%} depth_vld={vld:.1%} "
                    f"spread={spr:.3f}m bbox={bbox_n:.1%} | "
                    f"penalties: oversize_cov={w_cov_over*oc:+.3f} oversize_bbox={w_bbox_over*ob:+.3f} "
                    f"struct={-w_struct*ss:+.3f} (geom={sg:.2f} extent={se:.2f})"
                )

        # ----------------------------------------------------------------
        # same-frame dedup: greedy suppression to ensure top-K are distinct objects
        #
        # Purpose: prevent duplicate masks of the same object (e.g., FastSAM and
        # YOLOE both segment the same chair) from consuming the CLIP budget.
        # Association handles duplicates gracefully (both match the same WM object),
        # so this is purely a top-K budget optimization — we want K distinct objects
        # in the CLIP batch, not 3 copies of the chair.
        # ----------------------------------------------------------------

        def _centroid_dist_px(sa: MaskStats, sb: MaskStats) -> float:
            ca = getattr(sa, 'centroid_px', None)
            cb = getattr(sb, 'centroid_px', None)
            if ca is None or cb is None:
                return 1e9
            dx = float(ca[0]) - float(cb[0])
            dy = float(ca[1]) - float(cb[1])
            return float((dx*dx + dy*dy) ** 0.5)

        def _area_ratio(sa: MaskStats, sb: MaskStats) -> float:
            """Ratio of smaller to larger area (1.0 = identical size, 0.0 = vastly different)."""
            a = max(1, getattr(sa, 'area_px', 0))
            b = max(1, getattr(sb, 'area_px', 0))
            return min(a, b) / max(a, b)

        def _bbox_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
            ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
            iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
            inter = iw * ih
            area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
            area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
            union = max(1, area_a + area_b - inter)
            return float(inter) / float(union)

        # TODO: performance test — compare mask IoU dedup vs lightweight dedup.
        #
        # Original approach: pixel-wise mask IoU on full-resolution boolean tensors.
        # Precise but O(N² × H × W) — with retina_masks=true at 1920x1440 and ~30
        # candidates, this takes ~1s per frame (1.2B boolean ops). The precision
        # is overkill for dedup since association already handles duplicate matches
        # gracefully (both masks update the same WM object). Dedup only needs to
        # ensure the top-K CLIP batch contains distinct objects.
        #
        # When to re-enable: if the lightweight dedup below produces measurably
        # worse object tracking (e.g., CLIP batch wastes slots on duplicates that
        # should have been suppressed), switch back by setting:
        #   staging.dedup_use_mask: true
        # in rtsm.yaml. Consider also downsampling masks to 64x64 before IoU
        # as a middle ground (~700x cheaper than full-resolution).
        def _mask_iou(ma: torch.Tensor, mb: torch.Tensor) -> float:
            """Pixel-wise mask IoU — expensive at high resolution. See TODO above."""
            ia = (ma & mb).sum().item()
            ua = (ma | mb).sum().item()
            if ua <= 0:
                return 0.0
            return float(ia) / float(ua)

        # Lightweight dedup: centroid proximity + area similarity + bbox IoU.
        # All features from MaskStats (pure float math, zero tensor ops).
        # Runs in microseconds regardless of mask resolution.
        def _is_duplicate_lightweight(sa: MaskStats, sb: MaskStats) -> bool:
            # Quick reject: if centroids are far apart, definitely different objects
            dist = _centroid_dist_px(sa, sb)
            if dist > dedup_centroid_thr:
                return False
            # Area check: if sizes differ a lot, different objects at same location
            if _area_ratio(sa, sb) < dedup_area_ratio_min:
                return False
            # Confirm with bbox IoU (cheap integer math)
            return _bbox_iou(sa.bbox, sb.bbox) >= dedup_bbox_iou_thr

        # Config — lightweight thresholds (intentionally permissive to start)
        dedup_centroid_thr = float(self.cfg.get("staging",{}).get("dedup_centroid_px", 50.0))
        dedup_area_ratio_min = float(self.cfg.get("staging",{}).get("dedup_area_ratio_min", 0.4))
        dedup_bbox_iou_thr = float(self.cfg.get("staging",{}).get("dedup_bbox_iou_thr", 0.50))

        # Legacy config for mask IoU path
        iou_thr = float(self.cfg.get("staging",{}).get("dedup_iou_thr", 0.80))
        use_mask_iou = bool(self.cfg.get("staging",{}).get("dedup_use_mask", False))  # off by default now

        kept: List[Candidate] = []
        for c in cands:
            suppress = False
            for k in kept:
                if use_mask_iou:
                    # Legacy path: pixel-wise mask IoU (expensive, see TODO above)
                    iou = _mask_iou(c.mask, k.mask)
                    if iou >= iou_thr:
                        suppress = True
                        break
                else:
                    # Lightweight path: centroid + area + bbox (default)
                    if _is_duplicate_lightweight(c.stats, k.stats):
                        suppress = True
                        break
            if not suppress:
                kept.append(c)

        final = kept[:topK]

        # Mark which trace entries actually survived dedup + topK
        selected_mask_idx = {int(c.stats.idx) for c in final}
        for entry in trace_entries:
            if entry.mask_idx in selected_mask_idx:
                entry.selected_topk = True
        self._last_scoring_trace = ScoringTrace(
            n_candidates=len(trace_entries),
            n_selected=len(final),
            entries=trace_entries,
        )

        return final

    def _make_crops_inplace(self, cands: List[Candidate], rgb: np.ndarray,
                            mask_to_rgb_sx: float = 1.0,
                            mask_to_rgb_sy: float = 1.0):
        pad = int(self.cfg.get("staging",{}).get("crop_pad_px", 6))
        size = int(self.cfg.get("staging",{}).get("clip_input", 224))
        H, W, _ = rgb.shape

        for c in cands:
            x0, y0, x1, y1 = c.stats.bbox  # in mask space

            # Scale bbox from mask space to RGB space (no-op when retina_masks=true)
            x0 = int(x0 * mask_to_rgb_sx)
            y0 = int(y0 * mask_to_rgb_sy)
            x1 = int(x1 * mask_to_rgb_sx)
            y1 = int(y1 * mask_to_rgb_sy)

            # pad & clamp to the *image* bounds
            x0 = max(0, x0 - pad);  y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)

            # guard for degenerate boxes
            if x1 <= x0 or y1 <= y0:
                c.crop = None
                continue

            # Judgment crop FIRST, before any masking (2026-08-30): the
            # snapshot gallery needs real pixels with context.
            pad_frac = float(self.cfg.get("staging", {})
                             .get("judge_crop_pad_frac", 0.20))
            max_px = int(self.cfg.get("staging", {})
                         .get("judge_crop_max_px", 640))
            c.crop_hires = cut_judgment_crop(rgb, x0, y0, x1, y1,
                                             pad_frac, max_px)

            crop = rgb[y0:y1, x0:x1].copy()    # copy needed for masking
            if crop.size == 0:
                c.crop = None
                continue

            # Apply segmentation mask to suppress background —
            # CLIP embeds the full crop; without masking, background dominates
            # and all objects converge to similar embeddings.
            if c.mask is not None:
                # Scale mask region to match RGB crop bounds
                mask_region = c.mask[
                    int(y0 / mask_to_rgb_sy) : int(y1 / mask_to_rgb_sy),
                    int(x0 / mask_to_rgb_sx) : int(x1 / mask_to_rgb_sx),
                ]
                if mask_region.numel() > 0:
                    mask_np = mask_region.cpu().numpy().astype(np.uint8)
                    mask_resized = cv2.resize(
                        mask_np, (crop.shape[1], crop.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    # Suppress background pixels using configured fill strategy
                    bg_fill = self.cfg.get("staging", {}).get("bg_fill", "mean")
                    if bg_fill == "white":
                        crop[mask_resized == 0] = 255
                    elif bg_fill == "black":
                        crop[mask_resized == 0] = 0
                    elif bg_fill == "blur":
                        blurred = cv2.GaussianBlur(crop, (31, 31), 0)
                        crop[mask_resized == 0] = blurred[mask_resized == 0]
                    else:  # "mean" (default)
                        bg_color = crop.mean(axis=(0, 1)).astype(np.uint8)
                        crop[mask_resized == 0] = bg_color

            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
            c.crop = crop

    @torch.no_grad()
    def _clip_batch_inplace(self, cands: List[Candidate]):
        imgs = []
        idxs = []
        for i, c in enumerate(cands):
            if c.crop is None:
                continue
            # Crops are cut from the BGR ingest frame; CLIP/SigLIP was
            # trained on RGB — swap ONCE here (this line sat commented out
            # while every embedding in the store was computed on swapped
            # channels; found 2026-08-15).
            crop = c.crop[..., ::-1]
            imgs.append(Image.fromarray(crop))  # already 224x224 uint8
            idxs.append(i)
        if not imgs:
            return
        batch_size = int(self.cfg.get("clip", {}).get("batch_size", 16))
        embs = self.clip.encode_images_batch(imgs, batch_size=batch_size)  # [N,D] torch (GPU if available)

        if self.vocab_clf is not None and embs.numel() > 0:
            self.vocab_clf.ensure_text_on_device()
            k = int(self.cfg.get("clip", {}).get("vocab_topk", 5))
            cls = self.vocab_clf.classify_feats(embs, return_topk=k)
            for row, i in enumerate(idxs):
                if row < len(cls):
                    # topk always contains candidate labels with scores, even if primary is "unknown"
                    label, tv, class_idx, topk = cls[row]
                    # Always store top-K labels with scores (frontend will pick best for display)
                    setattr(cands[i], 'label_topk', [(cid, float(sc)) for (cid, sc, _j) in topk])
        # Merge DETECTION labels with CLIP vocab labels — the detection
        # head's label (higher specificity; for prompted backends it is
        # matched against the operator's vocabulary) goes ahead of CLIP's
        # vocab-classifier labels. Sources: dual/YOLOE fill
        # detection_labels/label_confidence; grounded_sam2 and standalone
        # yoloe fill the base labels/scores fields (2026-08-28: grounded
        # labels were silently dropped here — objects kept only CLIP
        # classifier labels like 'card box' for a GDINO-detected
        # 'tissue box', so label search had nothing to match).
        seg = getattr(self, '_last_seg_result', None)
        if seg is not None:
            det_labels = seg.detection_labels
            det_conf = seg.label_confidence
            if det_labels is None and seg.labels is not None:
                det_labels = seg.labels
                det_conf = ([float(s) for s in seg.scores]
                            if seg.scores is not None else None)
            if det_labels is not None:
                for i, c in enumerate(cands):
                    src_idx = c.stats.idx  # original mask index from segmentation output
                    if src_idx < len(det_labels) and det_labels[src_idx]:
                        det_label = det_labels[src_idx]
                        # RAW measured confidence (0.5 only when the
                        # backend reports none) — a floor here saturates
                        # label_scores to a constant, which destroys
                        # label-search ranking (exact ties broken by
                        # object age) and fabricates every stored
                        # label_confidence (reviewed 2026-08-28).
                        det_score = (float(det_conf[src_idx])
                                     if det_conf and src_idx < len(det_conf)
                                     else 0.5)
                        existing = getattr(c, 'label_topk', None) or []
                        merged = [(det_label, det_score)]
                        for lbl, sc in existing:
                            if lbl != det_label:
                                merged.append((lbl, sc))
                        c.label_topk = merged[:5]

        # Single boundary cast: move whole batch to CPU numpy float32 for association/WM
        try:
            embs_np = embs.detach().to(device="cpu", dtype=torch.float32).numpy()
        except Exception:
            # If already on CPU, ensure float32 numpy
            embs_np = np.asarray(embs, dtype=np.float32)
        # assign back per candidate
        for row, i in enumerate(idxs):
            cands[i].emb_vis = embs_np[row]

    def _maybe_flush_vectors(self):
        if self.vectors is None or self.working_mem is None:
            return
        flush_every_s = float(self.cfg.get("vectors",{}).get("flush_period_s", 5.0))
        now = time.monotonic()
        if (now - self._last_flush_ts) < flush_every_s:
            return
        self._last_flush_ts = now
        # Let associator (or WM) provide a list of “ready” objects to upsert.
        # Guarded: run_forever() has no per-step catch, so an exception here
        # would kill the whole pipeline thread.
        try:
            ready = self.working_mem.collect_ready_for_upsert()
        except Exception:
            logger.error("[PIPE] collect_ready_for_upsert failed", exc_info=True)
            return
        if not ready:
            return
        try:
            self.vectors.upsert_batch(ready)  # implementation behind your interface
        except Exception:
            self._vector_flush_failures += 1
            # Roll back the upsert bookkeeping so these objects retry next
            # flush; otherwise they look freshly upserted and stay invisible
            # to /search/semantic until the force period (or forever).
            requeued = 0
            try:
                requeued = self.working_mem.mark_upsert_failed(
                    [r["object_id"] for r in ready]
                )
            except Exception:
                logger.error("[PIPE] upsert-failure requeue failed", exc_info=True)
            logger.error(
                f"[PIPE] vector upsert FAILED for {len(ready)} objects "
                f"(requeued={requeued}, failures_total={self._vector_flush_failures}); "
                f"semantic search will miss these objects until a flush succeeds",
                exc_info=True,
            )
            return
        st = self.working_mem.stats()
        vec_total = "?"
        try:
            vec_total = self.vectors.stats().get("count", "?")
        except Exception:
            pass
        logger.info(
            f"[PIPE] flushed {len(ready)} upserts; wm_confirmed={st.get('confirmed',0)} "
            f"vectors_total={vec_total} upserts_total={st.get('upserts_total',0)}"
        )


    # -------- teardown --------
    def shutdown(self):
        # free heavy models if desired
        try:
            self.segmenter.close()
        except Exception:
            pass
        try:
            self.clip.close()
        except Exception:
            pass
        try:
            self._event_log.close()
        except Exception:
            pass

    # -------- single test  step (hardcoded import) --------
    @torch.no_grad()
    def test_step(self):
        from rtsm.utils.load_depth_png_as_meters import load_depth_png_as_meters
        from PIL import Image
        print("--------------------------------")
        print("running test_step")
        print("--------------------------------")
        # image_path = "test_dataset/rgb/1754989062.627478.png"
        image_path = "test_dataset/simple_room.png"
        pil = Image.open(image_path).convert("RGB")
        rgb = np.array(pil)
        seg_result = self.segmenter.segment(pil)
        ann_bool = prepare_ann(seg_result.masks) if seg_result.has_masks else torch.empty(0, pil.height, pil.width, dtype=torch.bool)
        depth_m = load_depth_png_as_meters("test_dataset/depth/1754989062.627478.png")
        kept_masks, stats, _filter_diag = run_heuristics(
            ann_bool,
            depth_m,
            cfg=self.cfg,
        )
        cands = self._score_and_select(kept_masks, stats)

        self._make_crops_inplace(cands, rgb)
        self._clip_batch_inplace(cands)
        # Print total number of candidates
        print(f"Total candidates: {len(cands)}")
        # Print only candidate's idx, its stats, and priority (not the entire ndarray)
        for cand in cands:
            print(f"Candidate idx: {cand.idx}")
            print(f"  Stats: area_px={cand.stats.area_px}, bbox={cand.stats.bbox}, "
                f"coverage={cand.stats.coverage:.4f}, border_fraction={cand.stats.border_fraction:.4f}, "
                f"depth_valid={cand.stats.depth_valid:.4f}, depth_p50={cand.stats.depth_p50}, "
                f"depth_spread={cand.stats.depth_spread}, centroid_px={cand.stats.centroid_px}")
            print(f"  Priority: {cand.priority:.4f}")
        return cands.copy()

