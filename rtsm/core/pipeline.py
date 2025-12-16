from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import List, Optional, Dict, Any, Tuple
import time
import numpy as np
import torch
import cv2
from PIL import Image

from rtsm.utils.mask_staging import run_heuristics, MaskStats
from rtsm.utils.prepare_ann import prepare_ann
from rtsm.utils.periodic_logger import PeriodicLogger
from rtsm.models.fastsam.adapter import FastSAMAdapter
from rtsm.models.clip.adapter import CLIPAdapter
from rtsm.models.clip.vocab_classifier import ClipVocabClassifier
from rtsm.stores.working_memory import WorkingMemory
from rtsm.stores.proximity_index import ProximityIndex
from rtsm.core.association import Associator
from rtsm.core.ingest_gate import IngestGate
from rtsm.stores.sweep_cache import SweepCache
from rtsm.io.ingest_queue import IngestQueue
from rtsm.core.datamodel import FramePacket

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

class Pipeline:
    def __init__(
        self,
        cfg: Dict[str,Any],
        fastsam: FastSAMAdapter,
        clip: CLIPAdapter,
        working_mem: WorkingMemory,
        proximity_index: ProximityIndex,
        associator: Associator,
        ingest_gate: IngestGate,
        vocab_clf: Optional[ClipVocabClassifier] = None,
        vectors: Optional[Any] = None,
        ingest_q: Optional[IngestQueue] = None,
        sweep_cache: Optional[SweepCache] = None,
    ):
        self.cfg = cfg
        self.fastsam = fastsam
        self.clip = clip
        self.working_mem = working_mem
        self.proximity_index = proximity_index
        self.associator = associator
        self.ingest_gate = ingest_gate
        self.vocab_clf = vocab_clf
        self.vectors = vectors  # generic vector store client (FAISS or Milvus)
        self._running = False
        self._last_flush_ts = 0.0
        self.ingest_q = ingest_q
        self.sweep_cache = sweep_cache or SweepCache()

        # Periodic summary logger
        log_cfg = cfg.get("logging", {})
        self._periodic_logger = PeriodicLogger(
            interval_s=float(log_cfg.get("summary_interval_s", 5.0)),
            enabled=bool(log_cfg.get("periodic_summary", True)),
        )

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
                    zc = float(snap.depth_m[H//2, W//2])
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
            return

        # 1) segmentation -> masks
        # FastSAMAdapter expects a PIL.Image for segment_everything
        rgb_img = snap.rgb
        pil_img = Image.fromarray(rgb_img) if isinstance(rgb_img, np.ndarray) else rgb_img
        ann = self.fastsam.segment_everything(pil_img)
        ann_bool = prepare_ann(ann)                   # [N,H,W] bool CPU
        n_masks = int(ann_bool.shape[0]) if hasattr(ann_bool, 'shape') else 0

        # 2) heuristics -> keep + stats
        kept_masks, stats = run_heuristics(
            ann_bool,
            snap.depth_m,
            cfg=self.cfg,
        )
        logger.debug(f"staging: kept {len(kept_masks)}/{n_masks} masks after heuristics")

        # 3) compute priorities (soft-features) and pick top-K
        cands = self._score_and_select(kept_masks, stats)
        
        # 4) pre-CLIP crop only top-K, then batch encode
        self._make_crops_inplace(cands, snap.rgb)
        self._clip_batch_inplace(cands)

        # 5) associate and update memory/index (if components provided)
        m, c = 0, 0
        is_kf = bool(pkt.is_keyframe) if pkt is not None else False
        if self.associator is not None and self.working_mem is not None and self.proximity_index is not None:
            stats_assoc = self.associator.update_with_candidates(cands, snap, self.working_mem, self.proximity_index, is_keyframe=is_kf)
            try:
                if isinstance(stats_assoc, dict):
                    m = int(stats_assoc.get("matched", 0))
                    c = int(stats_assoc.get("created", 0))
                    logger.debug(f"assoc: matched={m} created={c}")
            except Exception:
                pass

        # Update periodic logger with this frame's stats
        self._periodic_logger.tick(matched=m, created=c, masks=len(kept_masks))

        # 6) periodic flush/upsert to vector store (if configured)
        self._maybe_flush_vectors()

        # 7) periodic summary log
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
        except Exception:
            pass

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
                # Debug: log T_wc before inversion (periodically)
                if not hasattr(self, '_pipe_pose_log_count'):
                    self._pipe_pose_log_count = 0
                self._pipe_pose_log_count += 1
                if self._pipe_pose_log_count % 30 == 1:
                    t_raw = pkt.pose.t_wc
                    t_from_matrix = T_wc[:3, 3]
                    logger.debug(f"[pipeline] pkt.pose.t_wc={t_raw}, T_wc[:3,3]={t_from_matrix}")
                # inverse to get T_cam_world
                R = T_wc[:3, :3]
                t = T_wc[:3, 3]
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R.T
                T[:3, 3] = (-R.T @ t)
        except Exception:
            T = None
        return Snapshot(rgb=rgb, depth_m=depth_m, intrinsics=intr, pose_cam_T_world=T), pkt

    def _score_and_select(
        self, masks: List[torch.Tensor], stats: List[MaskStats]
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
            cands.append(Candidate(idx=i, mask=m, stats=s, priority=float(score)))

        # pick top-K by priority
        cands.sort(key=lambda c: c.priority, reverse=True)

        # same-frame dedup: greedy suppression using mask/bbox IoU, centroid as fallback
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

        def _mask_iou(ma: torch.Tensor, mb: torch.Tensor) -> float:
            ia = (ma & mb).sum().item()
            ua = (ma | mb).sum().item()
            if ua <= 0:
                return 0.0
            return float(ia) / float(ua)

        def _centroid_dist_px(sa: MaskStats, sb: MaskStats) -> float:
            ca = getattr(sa, 'centroid_px', None)
            cb = getattr(sb, 'centroid_px', None)
            if ca is None or cb is None:
                return 1e9
            dx = float(ca[0]) - float(cb[0])
            dy = float(ca[1]) - float(cb[1])
            return float((dx*dx + dy*dy) ** 0.5)

        iou_thr = float(self.cfg.get("staging",{}).get("dedup_iou_thr", 0.80))
        use_mask_iou = bool(self.cfg.get("staging",{}).get("dedup_use_mask", True))
        cen_thr_px = float(self.cfg.get("staging",{}).get("dedup_centroid_px", 12.0))

        kept: List[Candidate] = []
        for c in cands:
            suppress = False
            for k in kept:
                # primary: IoU
                if use_mask_iou:
                    iou = _mask_iou(c.mask, k.mask)
                else:
                    iou = _bbox_iou(c.stats.bbox, k.stats.bbox)
                if iou >= iou_thr:
                    suppress = True
                    break
                # fallback: centroid distance
                if _centroid_dist_px(c.stats, k.stats) <= cen_thr_px:
                    suppress = True
                    break
            if not suppress:
                kept.append(c)

        return kept[:topK]

    def _make_crops_inplace(self, cands: List[Candidate], rgb: np.ndarray):
        pad = int(self.cfg.get("staging",{}).get("crop_pad_px", 6))
        size = int(self.cfg.get("staging",{}).get("clip_input", 224))
        H, W, _ = rgb.shape
        
        for c in cands:
            x0, y0, x1, y1 = c.stats.bbox  # x1/y1 are exclusive (as we computed earlier)

            # pad & clamp to the *image* bounds
            x0 = max(0, x0 - pad);  y0 = max(0, y0 - pad)
            x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)

            # guard for degenerate boxes
            if x1 <= x0 or y1 <= y0:
                c.crop = None
                continue

            crop = rgb[y0:y1, x0:x1]           # still a view; cheap
            if crop.size == 0:
                c.crop = None
                continue

            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
            c.crop = crop

    @torch.no_grad()
    def _clip_batch_inplace(self, cands: List[Candidate]):
        imgs = []
        idxs = []
        for i, c in enumerate(cands):
            if c.crop is None:
                continue
            # ensure RGB PIL for preprocess; if your crop is BGR, swap channels here once:
            # crop = c.crop[..., ::-1]
            crop = c.crop
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
        # Let associator (or WM) provide a list of “ready” objects to upsert
        ready = self.working_mem.collect_ready_for_upsert()
        if not ready:
            return
        try:
            self.vectors.upsert_batch(ready)  # implementation behind your interface
        except Exception as e:
            logger.warning(f"vectors upsert failed: {e}")
        st = self.working_mem.stats()
        logger.info(f"[PIPE] flushed {len(ready)} upserts; wm_confirmed={st.get('confirmed',0)} upserts_total={st.get('upserts_total',0)}")


    # -------- teardown --------
    def shutdown(self):
        # free heavy models if desired
        try:
            self.fastsam.close()
        except Exception:
            pass
        try:
            self.clip.close()
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
        ann = self.fastsam.segment_everything(pil)
        ann_bool = prepare_ann(ann)
        depth_m = load_depth_png_as_meters("test_dataset/depth/1754989062.627478.png")
        kept_masks, stats = run_heuristics(
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

