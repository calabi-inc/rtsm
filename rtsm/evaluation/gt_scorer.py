"""Scoring an ``rtsm eval`` run against a simulator ground-truth sidecar.

The sidecar is what ``rtsm-isaacsim`` writes (schema 1): ``objects.json``
(the labelled props: stable id, prim path, class, oriented world box),
``gt.jsonl`` (per sensor stamp: the true and the published camera pose and
every labelled object with pixels in the instance image, with its pixel
count, 2-D box and the 3-D box of that frame) and ``meta.json``.

The match rule, in one sentence: an RTSM object (a point in the world) is
assigned to the smallest true box it lies within ``match_margin_m`` of,
preferring boxes that were in view in a frame where the object was observed.
Geometry only; labels never take part in matching, so the label-agreement
numbers are not circular.

Everything here is pure (ledger rows + summary + sidecar in, dicts out); the
report module calls it per run and flattens ``summary`` into the run's
scalars, so the floors across repeats come from the existing aggregation.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from rtsm.evaluation.event_log import KIND_OBS, OBS_MATCHED
from rtsm.evaluation.ledger import _stamp, by_kind

GT_SCORER_SCHEMA = 1
SIDECAR_SCHEMA = 1

# true class -> labels an open-vocabulary detector may use for it (lower case). Exact matches
# and plural/singular variants are accepted without the map.
DEFAULT_SYNONYMS: Dict[str, Set[str]] = {
    "table": {"desk", "dining table", "coffee table", "side table", "work table", "counter", "workbench"},
    "chair": {"armchair", "office chair", "stool", "seat", "swivel chair", "bar stool"},
    "sofa": {"couch", "loveseat", "settee", "bed"},
    "shelf": {"bookshelf", "bookcase", "cabinet", "cupboard", "locker", "rack", "drawer", "dresser", "shelving", "file cabinet"},
    "bin": {"trash can", "garbage can", "waste basket", "wastebasket", "trash bin", "bucket", "recycling bin", "trash"},
    "monitor": {"screen", "computer monitor", "tv", "television", "display", "computer screen", "computer"},
    "cup": {"mug", "coffee cup", "glass", "coffee mug"},
    "lamp": {"ceiling light", "light", "desk lamp", "floor lamp", "chandelier", "light fixture"},
    "plant": {"potted plant", "houseplant", "flower", "flower pot", "vase", "planter"},
    "box": {"carton", "crate", "cardboard box", "storage box", "package", "container", "bin"},
    "book": {"notebook", "binder", "folder", "magazine", "booklet", "ring binder"},
    "picture": {"picture frame", "painting", "poster", "photo", "frame", "photograph"},
    "laptop": {"notebook computer", "computer"},
    "keyboard": {"computer keyboard"},
    "mouse": {"computer mouse"},
    "phone": {"telephone", "cell phone", "mobile phone", "landline"},
    "printer": {"copier", "copy machine", "scanner"},
    "whiteboard": {"board", "blackboard", "bulletin board"},
    "clock": {"wall clock"},
    "pen": {"pencil", "marker", "pens"},
    "tray": {"paper tray"},
    "bottle": {"water bottle"},
    "sink": {"washbasin", "basin"},
    "extinguisher": {"fire extinguisher"},
    "mousepad": {"mouse pad"},
    "microphone": {"mic"},
    "camera": {"security camera", "cctv", "surveillance camera"},
    "dispenser": {"soap dispenser"},
}


# ───────────────────────────── parameters ─────────────────────────────

@dataclass(frozen=True)
class GtParams:
    match_margin_m: float = 0.15        # point-to-box distance that still counts as "on the object"
    min_pixels: int = 1000              # a true object counts as visible in a frame above this many pixels
    min_frames: int = 2                 # ...and as observable over the run when visible in this many processed frames
    det_iou: float = 0.3                # 2-D box overlap for a detection to count as a hit
    revisit_gap_s: float = 5.0          # the report's revisit gap (MetricParams.revisit_gap_s)
    top_classes: int = 12


# ───────────────────────────── the sidecar ─────────────────────────────

def quat_xyzw_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


@dataclass
class TruthObject:
    id: int
    prim: str
    cls: str
    center: np.ndarray                  # (3,)
    half: np.ndarray                    # (3,) half extents along the box axes
    R: np.ndarray                       # (3, 3) box axes in world (columns)
    volume: float

    @classmethod
    def from_record(cls, rec: dict) -> Optional["TruthObject"]:
        box = rec.get("bbox3d_world")
        if not box:
            return None
        c = np.asarray(box["center"], dtype=np.float64).reshape(3)
        e = np.asarray(box["extents"], dtype=np.float64).reshape(3)
        R = quat_xyzw_to_matrix(box.get("q_xyzw") or (0.0, 0.0, 0.0, 1.0))
        half = np.maximum(e / 2.0, 1e-6)
        return cls(id=int(rec["id"]), prim=str(rec.get("prim", "")), cls=str(rec.get("class", "")), center=c, half=half, R=R,
                   volume=float(np.prod(2.0 * half)))


@dataclass
class Sidecar:
    dir: str
    meta: dict
    objects: Dict[int, TruthObject]
    unboxed: int                                      # labelled objects without a box (cannot be matched)
    frames: Dict[int, Dict[int, dict]]                # ts_ns -> {gt_id -> object line (pixels, bbox2d, ...)}
    pose_true: Dict[int, dict] = field(default_factory=dict)

    @property
    def n_labelled(self) -> int:
        return len(self.objects) + self.unboxed

    def pixels(self, ts: int, gid: int) -> int:
        return int((self.frames.get(ts, {}).get(gid) or {}).get("pixels") or 0)


def load_sidecar(gt_dir: Any) -> Sidecar:
    d = Path(gt_dir)
    objs_path, lines_path, meta_path = d / "objects.json", d / "gt.jsonl", d / "meta.json"
    if not objs_path.is_file() or not lines_path.is_file():
        raise FileNotFoundError(f"not a ground-truth sidecar (objects.json + gt.jsonl): {d}")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    schema = int(meta.get("schema", SIDECAR_SCHEMA))
    if schema != SIDECAR_SCHEMA:
        raise ValueError(f"sidecar schema {schema} is not supported (this scorer reads schema {SIDECAR_SCHEMA})")
    objects: Dict[int, TruthObject] = {}
    unboxed = 0
    for rec in json.loads(objs_path.read_text(encoding="utf-8")):
        t = TruthObject.from_record(rec)
        if t is None:
            unboxed += 1
        else:
            objects[t.id] = t
    frames: Dict[int, Dict[int, dict]] = {}
    pose_true: Dict[int, dict] = {}
    with open(lines_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            ts = int(row["ts_ns"])
            frames[ts] = {int(o["id"]): o for o in (row.get("objects") or [])}
            if row.get("T_wc_true"):
                pose_true[ts] = row["T_wc_true"]
    return Sidecar(dir=str(d), meta=meta, objects=objects, unboxed=unboxed, frames=frames, pose_true=pose_true)


# ───────────────────────────── geometry ─────────────────────────────

def box_distance(p: Sequence[float], t: TruthObject) -> float:
    """Distance from a world point to the box's surface, 0 inside."""
    local = t.R.T @ (np.asarray(p, dtype=np.float64).reshape(3) - t.center)
    excess = np.maximum(np.abs(local) - t.half, 0.0)
    return float(np.linalg.norm(excess))


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


# ───────────────────────────── labels ─────────────────────────────

def _norm_label(s: Any) -> str:
    return " ".join(str(s or "").strip().lower().replace("_", " ").split())


def label_agreement(true_cls: str, label: str, synonyms: Dict[str, Set[str]]) -> str:
    """'exact' | 'synonym' | 'none'."""
    a, b = _norm_label(true_cls), _norm_label(label)
    if not a or not b:
        return "none"
    if a == b or a + "s" == b or a == b + "s":
        return "exact"
    syn = {_norm_label(x) for x in synonyms.get(a, ())}
    if b in syn or (b.endswith("s") and b[:-1] in syn) or (b + "s") in syn:
        return "synonym"
    return "none"


# ───────────────────────────── matching ─────────────────────────────

@dataclass
class MatchRecord:
    id: str
    label: str
    confirmed: bool
    hits: int
    position: List[float]
    n_obs: int
    gt_id: Optional[int] = None
    gt_class: Optional[str] = None
    d_box_m: Optional[float] = None
    d_center_m: Optional[float] = None
    agreement: str = "none"
    n_candidates: int = 0


def match_objects(mem_objs: Sequence[dict], obs_by_id: Dict[str, List[dict]], sc: Sidecar, prm: GtParams,
                  synonyms: Dict[str, Set[str]]) -> List[MatchRecord]:
    visible_at: Dict[int, Set[int]] = {ts: {gid for gid, o in objs.items() if int(o.get("pixels") or 0) > 0} for ts, objs in sc.frames.items()}
    out: List[MatchRecord] = []
    for o in mem_objs:
        oid = str(o.get("id"))
        p = o.get("xyz_world")
        rows = obs_by_id.get(oid, [])
        rec = MatchRecord(id=oid, label=str(o.get("label_primary") or ""), confirmed=bool(o.get("confirmed")), hits=int(o.get("hits") or 0),
                          position=[float(v) for v in (p or [])], n_obs=len(rows))
        if p is None or len(p) != 3:
            out.append(rec)
            continue
        cands = []
        for t in sc.objects.values():
            d = box_distance(p, t)
            if d <= prm.match_margin_m:
                cands.append((d, t))
        rec.n_candidates = len(cands)
        if cands:
            stamps = {s for s in (_stamp(r) for r in rows) if s is not None}
            seen = [(d, t) for d, t in cands if any(t.id in visible_at.get(ts, ()) for ts in stamps)]
            pool = seen or cands
            d, t = min(pool, key=lambda dt: (dt[1].volume, dt[0]))
            rec.gt_id, rec.gt_class, rec.d_box_m = t.id, t.cls, round(d, 4)
            rec.d_center_m = round(float(np.linalg.norm(np.asarray(p, dtype=np.float64) - t.center)), 4)
            rec.agreement = label_agreement(t.cls, rec.label, synonyms)
        out.append(rec)
    return out


# ───────────────────────────── the score ─────────────────────────────

def _processed_stamps(kinds: Dict[str, List[dict]]) -> List[int]:
    stamps = [s for s in (_stamp(r) for r in kinds.get("frame", [])) if s is not None]
    if not stamps:
        stamps = [s for s in (_stamp(r) for r in kinds.get("dequeue", []) if r.get("outcome") == "processed") if s is not None]
    return sorted(set(stamps))


def _median(xs: Sequence[float]) -> Optional[float]:
    return round(float(np.median(xs)), 4) if len(xs) else None


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    return round(float(np.percentile(xs, q)), 4) if len(xs) else None


def _frac(n: float, d: float) -> Optional[float]:
    return round(n / d, 6) if d else None


def score_run(rows: Sequence[dict], summary: Optional[dict], sc: Sidecar, prm: Optional[GtParams] = None,
              synonyms: Optional[Dict[str, Set[str]]] = None) -> dict:
    """Everything the sidecar can say about one run. ``rows`` = the ledger,
    ``summary`` = the run's summary.json (its memory.objects are the survivors)."""
    prm = prm or GtParams()
    syn = synonyms if synonyms is not None else DEFAULT_SYNONYMS
    summary = summary or {}
    mem_objs: List[dict] = list(((summary.get("memory") or {}).get("objects")) or [])
    kinds = by_kind(rows)
    obs_by_id: Dict[str, List[dict]] = defaultdict(list)
    for r in kinds.get(KIND_OBS, []):
        if r.get("object_id"):
            obs_by_id[str(r["object_id"])].append(r)
    processed = _processed_stamps(kinds)
    processed_in = [ts for ts in processed if ts in sc.frames]

    # ── objects: RTSM -> truth ───────────────────────────────────────
    matches = match_objects(mem_objs, obs_by_id, sc, prm, syn)
    by_gt: Dict[int, List[MatchRecord]] = defaultdict(list)
    for m in matches:
        if m.gt_id is not None:
            by_gt[m.gt_id].append(m)
    n = len(matches)
    n_conf = sum(1 for m in matches if m.confirmed)
    n_matched = sum(1 for m in matches if m.gt_id is not None)
    n_matched_conf = sum(1 for m in matches if m.gt_id is not None and m.confirmed)

    # ── truth: observable, recalled, duplicated ──────────────────────
    visible_frames: Dict[int, List[int]] = defaultdict(list)       # gt id -> processed stamps where it is big enough
    for ts in processed_in:
        for gid, o in sc.frames[ts].items():
            if int(o.get("pixels") or 0) >= prm.min_pixels and gid in sc.objects:
                visible_frames[gid].append(ts)
    observable = {gid for gid, fs in visible_frames.items() if len(fs) >= prm.min_frames}
    recalled = {gid for gid in observable if by_gt.get(gid)}
    recalled_conf = {gid for gid in observable if any(m.confirmed for m in by_gt.get(gid, []))}
    dup_extra = sum(max(0, len(by_gt[gid]) - 1) for gid in by_gt)
    n_dup = sum(1 for gid in by_gt if len(by_gt[gid]) > 1)
    matched_outside_observable = sum(1 for gid in by_gt if gid not in observable)

    # ── position ──────────────────────────────────────────────────────
    d_box = [m.d_box_m for m in matches if m.d_box_m is not None]
    d_cen = [m.d_center_m for m in matches if m.d_center_m is not None]

    # ── labels ────────────────────────────────────────────────────────
    pairs = [m for m in matches if m.gt_id is not None]
    exact = sum(1 for m in pairs if m.agreement == "exact")
    synonym = sum(1 for m in pairs if m.agreement == "synonym")
    confusion: Counter = Counter((m.gt_class, _norm_label(m.label)) for m in pairs if m.agreement == "none")

    # ── revisits against truth (recalled objects only; an unbuilt object is a recall miss, not a re-id miss) ──
    gap_ns = int(prm.revisit_gap_s * 1e9)
    rev = Counter()
    truth_records: List[dict] = []
    for gid in sorted(observable):
        fs = visible_frames[gid]
        visits: List[List[int]] = []
        for ts in fs:
            if visits and ts - visits[-1][-1] < gap_ns:
                visits[-1].append(ts)
            else:
                visits.append([ts])
        ids = [m.id for m in by_gt.get(gid, [])]
        first_obs: Dict[str, int] = {}
        for oid in ids:
            stamps_o = [s for s in (_stamp(r) for r in obs_by_id.get(oid, [])) if s is not None]
            if stamps_o:
                first_obs[oid] = min(stamps_o)
        outcomes: List[str] = []
        for v in (visits[1:] if ids else []):            # re-identification is only a question for objects RTSM built
            lo, hi = v[0], v[-1]
            older = [oid for oid, f in first_obs.items() if f < lo]           # RTSM objects on this prop that existed before the gap
            born_here = [oid for oid, f in first_obs.items() if lo <= f <= hi]
            old_matched = any(r.get("outcome") == OBS_MATCHED and lo <= (_stamp(r) or -1) <= hi
                              for oid in older for r in obs_by_id.get(oid, []))
            if old_matched:
                o = "reidentified"                       # an object from before the gap was matched again on this prop
            elif born_here and older:
                o = "duplicated"                         # a new object was created on it although one already existed
            elif born_here:
                o = "first_seen_late"                    # RTSM first built it on this visit; nothing to re-identify
            else:
                o = "missed"
            outcomes.append(o)
            rev[o] += 1
        t = sc.objects[gid]
        truth_records.append({"gt_id": gid, "class": t.cls, "prim": t.prim, "n_visible_frames": len(fs), "n_visits": len(visits),
                              "n_rtsm_objects": len(ids), "rtsm_ids": ids, "revisits": outcomes,
                              "max_pixels": max(sc.pixels(ts, gid) for ts in fs)})
    n_revisits = sum(rev.values())

    # ── detections per processed frame ───────────────────────────────
    det_by_ts: Dict[int, dict] = {}
    for r in kinds.get("det", []):
        s = _stamp(r)
        if s is not None and s not in det_by_ts:
            det_by_ts[s] = r
    tp = n_truth = n_det = 0
    per_class_truth: Counter = Counter()
    per_class_hit: Counter = Counter()
    frames_scored = 0
    for ts in processed_in:
        truths = [(gid, o["bbox2d"]) for gid, o in sc.frames[ts].items()
                  if int(o.get("pixels") or 0) >= prm.min_pixels and o.get("bbox2d") and gid in sc.objects]
        dets = [d.get("bbox_rgb") for d in ((det_by_ts.get(ts) or {}).get("detections") or []) if d.get("bbox_rgb")]
        if not truths and not dets:
            continue
        frames_scored += 1
        n_truth += len(truths)
        n_det += len(dets)
        for gid, _ in truths:
            per_class_truth[sc.objects[gid].cls] += 1
        pairs_iou = sorted(((iou_xyxy(tb, db), i, j) for i, (_, tb) in enumerate(truths) for j, db in enumerate(dets)),
                           key=lambda x: -x[0])
        used_t: Set[int] = set()
        used_d: Set[int] = set()
        for v, i, j in pairs_iou:
            if v < prm.det_iou:
                break
            if i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            tp += 1
            per_class_hit[sc.objects[truths[i][0]].cls] += 1
    top = [c for c, _ in per_class_truth.most_common(prm.top_classes)]
    per_class = {c: {"n_truth": per_class_truth[c], "n_hit": per_class_hit[c], "recall": _frac(per_class_hit[c], per_class_truth[c])} for c in top}

    summary_block = {
        "objects": {"n": n, "n_confirmed": n_conf, "n_matched": n_matched, "n_matched_confirmed": n_matched_conf,
                    "n_phantom": n - n_matched, "n_phantom_confirmed": n_conf - n_matched_conf,
                    "precision": _frac(n_matched, n), "precision_confirmed": _frac(n_matched_conf, n_conf),
                    "matched_to_unobservable": matched_outside_observable},
        "truth": {"n_labelled": sc.n_labelled, "n_boxed": len(sc.objects), "n_observable": len(observable),
                  "n_recalled": len(recalled), "recall": _frac(len(recalled), len(observable)),
                  "n_recalled_confirmed": len(recalled_conf), "recall_confirmed": _frac(len(recalled_conf), len(observable)),
                  "n_duplicated": n_dup, "duplicates_extra": dup_extra},
        "position": {"d_box_median_m": _median(d_box), "d_box_p90_m": _pct(d_box, 90), "d_center_median_m": _median(d_cen),
                     "n": len(d_box)},
        "labels": {"n_pairs": len(pairs), "exact": exact, "exact_frac": _frac(exact, len(pairs)),
                   "synonym": synonym, "synonym_frac": _frac(synonym, len(pairs)),
                   "agree_frac": _frac(exact + synonym, len(pairs))},
        "revisits": {"n_revisits": n_revisits, "reidentified": rev["reidentified"], "duplicated": rev["duplicated"],
                     "missed": rev["missed"], "first_seen_late": rev["first_seen_late"],
                     "reid_frac": _frac(rev["reidentified"], n_revisits)},
        "detection": {"frames": frames_scored, "n_truth": n_truth, "n_det": n_det, "tp": tp,
                      "recall": _frac(tp, n_truth), "precision": _frac(tp, n_det)},
        "frames": {"n_processed": len(processed), "n_processed_in_sidecar": len(processed_in), "join_frac": _frac(len(processed_in), len(processed))},
    }
    return {
        "schema": GT_SCORER_SCHEMA, "sidecar": {"dir": sc.dir, "schema": SIDECAR_SCHEMA, "scene": sc.meta.get("scene"),
                                                "session_id": sc.meta.get("session_id"), "frames": len(sc.frames)},
        "params": asdict(prm),
        "summary": summary_block,
        "rtsm_objects": [asdict(m) for m in matches],
        "truth_objects": truth_records,
        "confusion": [{"true_class": a, "rtsm_label": b, "n": c} for (a, b), c in confusion.most_common(30)],
        "detection_by_class": per_class,
    }
