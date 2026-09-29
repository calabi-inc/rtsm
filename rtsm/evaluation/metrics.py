"""
Metrics over the ledgers of ONE ``rtsm eval`` run (Gate 4.5 plan, P3 task 3).

Pure functions over the rows of ``events.jsonl`` (``rtsm/evaluation/ledger.py``
reads them) and the run's ``summary.json``. No model, no GPU, no ground truth,
no labels needed: everything below is derived from the pose / obs / view
ledgers and the frame-flow trace, and every definition states its caveat.

  object_tracks(rows, survivor_ids)   -> {id: Track}   one track per object id the
                                                       associator matched or created
  leader_clusters(tracks, radius)     -> [Cluster]     spatial clusters of tracks (the
                                                       proxy for physical objects)
  compute_metrics(rows, summary, p)   -> dict          scalars + clusters + objects +
                                                       frames + moments for one run

Definitions (the report repeats them in its method notes):

* A **track** is every ``obs`` line that carries an ``object_id`` (outcome
  ``matched`` or ``created``): the RAW ``p_world`` of each observation, the
  stamp, the camera position, the range, the top-1 label. A track is a
  **survivor** when its id is in the final memory (``summary.memory.objects``)
  and **transient** otherwise (a proto that expired, or an object the memory
  evicted).
* A **cluster** groups tracks whose median raw positions lie within the
  cluster radius (default: the associator's own distance gate,
  ``assoc.gate_dist_base_m``). Leader clustering in creation order: a track
  joins the nearest existing leader within the radius, else starts a
  cluster. Deterministic, no chaining. It is a proxy for "one physical
  object" and over-merges neighbours closer than the radius; the sensitivity
  to the radius is reported.
* **Detection over views**: for every processed frame whose ``view`` line
  lists a member of a cluster (occlusion-agnostic frustum), the cluster is
  ``reidentified`` when a member was matched on that frame, ``duplicated``
  when a member was created instead (a segment was found there but the memory
  did not recognise it), ``missed`` when neither happened. A frame on which a
  member is matched or created counts as a presence frame even when the view
  line does not list the cluster (the view precedes association).
* **Label disagreement** of a track: ``1 - modal_count / n`` over the top-1
  labels of its observations (0 for a single observation).
* **Scatter**: per track with >= ``min_obs_scatter`` observations, the
  residual of each raw observation to the track median, split into the
  component along the camera-to-median ray (depth) and the lateral
  remainder; pooled RMS, per range bin, and a least-squares line
  ``|along| ~ a + b * range``.
* **Duplicate spawn**: a ``created`` line whose raw position lies within the
  cluster radius of an earlier track that was ALIVE at that stamp (a
  survivor, or last observed less than ``proto_ttl_s`` earlier). The reason
  class comes from the created line's audit fields: ``not_in_index``
  (``n_nearby`` 0), ``gated`` (nearby but no gate survivor), ``low_similarity``
  (survivors, but ``max_cos`` below ``cos_min``), ``other``.
* **Revisits**: the presence frames of a cluster split into visits at sensor
  gaps >= ``revisit_gap_s``; every visit after the first is a revisit, which
  is ``reidentified`` when a member was matched during it, ``duplicated``
  when one was created and none matched, ``missed`` otherwise. The
  re-identification LOWER BOUND counts missed and duplicated revisits as
  failures (an occluded object is a miss here too).
* **Worst moments**: processed frames scored by misses + 2 x duplicate spawns
  (+1 when the frame-quality gate rejected the frame), plus the pose gaps,
  discontinuities and tracking-limited episodes from ``pose_health``; each
  moment carries the sensor stamp and the time since the first pose line so
  an engineer can scrub to it.
* Under the sweep gate's shadow mode (``dense`` / ``every_frame``) every
  view-based number is also given restricted to the frames the deployed gate
  would have admitted (``gate_shadow`` null on the dequeue line): ``masked``.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from rtsm.evaluation.event_log import KIND_OBS, KIND_POSE, KIND_VIEW, OBS_CREATED, OBS_MATCHED
from rtsm.evaluation.ledger import _stamp, _stats, by_kind, observation_summary, outcome_histogram, pose_health

METRICS_SCHEMA = 1
DUP_REASONS = ("not_in_index", "gated", "low_similarity", "other")


# ───────────────────────────── parameters ─────────────────────────────

@dataclass(frozen=True)
class MetricParams:
    cluster_radius_m: float = 0.5               # None in the config -> assoc.gate_dist_base_m
    cluster_radii_m: Tuple[float, ...] = (0.25, 0.5, 1.0)
    revisit_gap_s: float = 5.0
    range_bin_m: float = 0.5
    min_obs_scatter: int = 3
    worst_n: int = 10
    moments_cap: int = 100                      # per-cluster seen / missed stamps kept in the JSON
    proto_ttl_s: float = 10.0                   # object.proto_ttl_s
    cos_min: float = 0.90                       # assoc.cos_min
    disc_base_m: float = 0.5
    disc_rate_mps: float = 1.0
    gap_factor: float = 2.0

    @classmethod
    def from_cfg(cls, cfg: Optional[dict]) -> "MetricParams":
        cfg = cfg or {}
        ev = (cfg.get("eval") or {})
        m = dict(ev.get("metrics") or {})
        assoc = cfg.get("assoc") or {}
        obj = cfg.get("object") or {}
        radius = m.get("cluster_radius_m")
        if radius is None:
            radius = assoc.get("gate_dist_base_m", 0.5)
        radius = float(radius)
        if radius <= 0:
            raise ValueError("eval.metrics.cluster_radius_m must be > 0")
        radii = m.get("cluster_radii_m")
        if radii is None:
            radii = (round(radius / 2, 6), radius, round(radius * 2, 6))
        radii = tuple(sorted({float(r) for r in radii if float(r) > 0} | {radius}))
        gap = float(m.get("revisit_gap_s", 5.0))
        if gap <= 0:
            raise ValueError("eval.metrics.revisit_gap_s must be > 0")
        return cls(
            cluster_radius_m=radius, cluster_radii_m=radii, revisit_gap_s=gap,
            range_bin_m=max(1e-3, float(m.get("range_bin_m", 0.5))),
            min_obs_scatter=max(2, int(m.get("min_obs_scatter", 3))),
            worst_n=max(1, int(m.get("worst_n", 10))), moments_cap=max(1, int(m.get("moments_cap", 100))),
            proto_ttl_s=float(obj.get("proto_ttl_s", 10.0)), cos_min=float(assoc.get("cos_min", 0.90)),
            disc_base_m=float(m.get("disc_base_m", 0.5)), disc_rate_mps=float(m.get("disc_rate_mps", 1.0)),
            gap_factor=float(m.get("gap_factor", 2.0)),
        )


# ───────────────────────────── helpers ─────────────────────────────

def _r(v: Any, nd: int = 6) -> Any:
    """Round a finite float for the JSON; ints and None pass through."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _stats_r(values: Iterable[Any]) -> Dict[str, Any]:
    return {k: _r(v) for k, v in _stats(values).items()}


def _top1(r: dict) -> Optional[str]:
    tk = r.get("label_topk")
    if isinstance(tk, list) and tk:
        first = tk[0]
        if isinstance(first, dict):
            lbl = first.get("label")
        elif isinstance(first, (list, tuple)) and first:
            lbl = first[0]
        else:
            lbl = first
        return str(lbl) if lbl is not None else None
    return None


def _vec3(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        a = np.asarray([float(x) for x in v], dtype=float)
    except (TypeError, ValueError):
        return None
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return None
    return a


def _div(n: float, d: float) -> Optional[float]:
    return _r(n / d) if d else None


def _range_bin(rng: float, width: float) -> str:
    lo = math.floor(rng / width) * width
    return f"{lo:.2f}-{lo + width:.2f}"


# ───────────────────────────── tracks ─────────────────────────────

@dataclass
class Track:
    id: str
    survivor: bool = False
    points: List[np.ndarray] = field(default_factory=list)   # raw p_world per observation (matched + created)
    stamps: List[int] = field(default_factory=list)
    frame_seqs: List[Optional[int]] = field(default_factory=list)
    cams: List[Optional[np.ndarray]] = field(default_factory=list)
    ranges: List[Optional[float]] = field(default_factory=list)
    labels: List[Optional[str]] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    created_ts: Optional[int] = None
    created_seq: Optional[int] = None
    created_order: Optional[int] = None                       # file position of the created line
    first_order: Optional[int] = None                         # file position of the first obs line (id-free tie-break)
    created_audit: Dict[str, Any] = field(default_factory=dict)
    n_matched: int = 0
    n_created: int = 0
    n_without_scoring: int = 0
    n_no_point: int = 0                                       # matched / created lines without a usable p_world

    @property
    def n_obs(self) -> int:
        return len(self.points)

    @property
    def first_ts(self) -> Optional[int]:
        return min(self.stamps) if self.stamps else self.created_ts

    @property
    def last_ts(self) -> Optional[int]:
        return max(self.stamps) if self.stamps else self.created_ts

    @property
    def median(self) -> Optional[np.ndarray]:
        if not self.points:
            return None
        return np.median(np.stack(self.points), axis=0)


def object_tracks(rows: Iterable[dict], survivor_ids: Iterable[str] = ()) -> Dict[str, Track]:
    """One track per object id over the matched + created obs lines, in file order."""
    survivors = set(survivor_ids)
    tracks: Dict[str, Track] = {}
    order = 0
    for r in rows:
        if r.get("kind") != KIND_OBS:
            continue
        oid = r.get("object_id")
        out = r.get("outcome")
        if not oid or out not in (OBS_MATCHED, OBS_CREATED):
            continue
        t = tracks.get(oid)
        if t is None:
            t = tracks[oid] = Track(id=str(oid), survivor=(oid in survivors), first_order=order)
        ts = _stamp(r)
        p = _vec3(r.get("p_world"))
        if out == OBS_CREATED:
            t.n_created += 1
            if t.created_ts is None:
                t.created_ts, t.created_seq, t.created_order = ts, r.get("frame_seq"), order
                t.created_audit = {"n_nearby": int(r.get("n_nearby") or 0), "n_gate_survivors": int(r.get("n_gate_survivors") or 0),
                                   "max_cos": _r(r.get("max_cos")), "view_bin": r.get("view_bin"), "range_m": _r(r.get("range_m"))}
        else:
            t.n_matched += 1
            if r.get("matched_without_scoring"):
                t.n_without_scoring += 1
        if p is None or ts is None:
            t.n_no_point += 1
        else:
            t.points.append(p)
            t.stamps.append(int(ts))
            t.frame_seqs.append(r.get("frame_seq"))
            t.cams.append(_vec3(r.get("cam_t_wc")))
            rng = r.get("range_m")
            t.ranges.append(float(rng) if isinstance(rng, (int, float)) and math.isfinite(float(rng)) else None)
            t.labels.append(_top1(r))
            t.outcomes.append(str(out))
        order += 1
    return tracks


# ───────────────────────────── clusters ─────────────────────────────

@dataclass
class Cluster:
    index: int
    leader: str
    position: np.ndarray
    members: List[str]

    @property
    def id(self) -> str:
        return f"c{self.index:04d}"


def leader_clusters(tracks: Dict[str, Track], radius: float) -> List[Cluster]:
    """Leader clustering of the track medians in creation order (first stamp,
    then id): a track joins the NEAREST existing leader within ``radius``,
    else starts a cluster. Tracks without a usable position are skipped."""
    # Creation order with an ID-FREE tie-break (object ids are random per run: sorting on them would make the
    # clusters of two runs with identical ledgers differ).
    ordered = sorted((t for t in tracks.values() if t.median is not None),
                     key=lambda t: (t.first_ts if t.first_ts is not None else float("inf"),
                                    t.first_order if t.first_order is not None else float("inf")))
    clusters: List[Cluster] = []
    leaders = np.zeros((0, 3), dtype=float)
    for t in ordered:
        m = t.median
        if leaders.shape[0]:
            d = np.linalg.norm(leaders - m[None, :], axis=1)
            j = int(np.argmin(d))
            if d[j] <= radius:
                clusters[j].members.append(t.id)
                continue
        clusters.append(Cluster(index=len(clusters), leader=t.id, position=m.copy(), members=[t.id]))
        leaders = np.vstack([leaders, m[None, :]])
    return clusters


# ───────────────────────────── per-run metrics ─────────────────────────────

def _frame_index(rows: Sequence[dict]) -> Dict[str, Any]:
    """Per-stamp joins: the dequeue line, the view line, the obs lines."""
    kinds = by_kind(rows)
    dq_by_ts: Dict[int, dict] = {}
    for r in kinds.get("dequeue", []):
        ts = _stamp(r)
        if ts is not None and ts not in dq_by_ts:
            dq_by_ts[ts] = r
    view_by_ts: Dict[int, dict] = {}
    for r in kinds.get(KIND_VIEW, []):
        ts = _stamp(r)
        if ts is not None and ts not in view_by_ts:
            view_by_ts[ts] = r
    obs_by_ts: Dict[int, List[dict]] = defaultdict(list)
    for r in kinds.get(KIND_OBS, []):
        ts = _stamp(r)
        if ts is not None:
            obs_by_ts[ts].append(r)
    return {"kinds": kinds, "dequeue": dq_by_ts, "view": view_by_ts, "obs": obs_by_ts}


def compute_metrics(rows: Sequence[dict], summary: Optional[dict], params: Optional[MetricParams] = None) -> dict:
    """Every metric of one run. ``summary`` is the run's ``summary.json`` (None
    tolerated: memory-derived numbers are then taken from the ledgers alone)."""
    p = params or MetricParams()
    summary = summary or {}
    memory = summary.get("memory") or {}
    mem_objs: List[dict] = list(memory.get("objects") or [])
    survivors = {o.get("id") for o in mem_objs if o.get("id")}
    mem_by_id = {o.get("id"): o for o in mem_objs if o.get("id")}
    idx = _frame_index(rows)
    kinds = idx["kinds"]
    tracks = object_tracks(rows, survivors)
    clusters = leader_clusters(tracks, p.cluster_radius_m)
    cluster_of: Dict[str, int] = {}
    for c in clusters:
        for oid in c.members:
            cluster_of[oid] = c.index
    n_at_radius = {f"{r:g}": len(leader_clusters(tracks, r)) for r in p.cluster_radii_m}

    # time origin: the first pose stamp (falls back to the first receiver / dequeue stamp)
    stamps = [s for s in (_stamp(r) for r in kinds.get(KIND_POSE, [])) if s is not None]
    if not stamps:
        stamps = [s for s in (_stamp(r) for r in kinds.get("receiver", []) + kinds.get("dequeue", [])) if s is not None]
    t0 = min(stamps) if stamps else None

    def rel(ts: Optional[int]) -> Optional[float]:
        return _r((ts - t0) / 1e9, 4) if (ts is not None and t0 is not None) else None

    shadow_mode = any(r.get("gate_shadow") for r in kinds.get("dequeue", []))

    # ── presence per cluster per frame ───────────────────────────────
    # per frame: in-frustum ids (view), matched ids, created ids
    frames: List[dict] = []
    per_id_views = Counter(); per_id_reid = Counter(); per_id_missed = Counter()
    per_id_range_bins: Dict[str, List[str]] = defaultdict(list)     # (bin, detected) at id level
    range_tab: Dict[str, Counter] = defaultdict(Counter)
    presence: Dict[int, List[Tuple[int, str, bool]]] = defaultdict(list)   # cluster -> [(ts, outcome, shadowed)]
    unknown_view_ids = 0
    dup_records: List[dict] = []
    dups_by_frame = Counter()
    created_lines: List[Tuple[int, dict]] = []
    for ts, obs in idx["obs"].items():
        for r in obs:
            if r.get("outcome") == OBS_CREATED and r.get("object_id"):
                created_lines.append((ts, r))

    # duplicate spawns (needs the tracks; independent of the frames loop)
    track_list = list(tracks.values())
    alive_medians = [(t, t.median) for t in track_list if t.median is not None]
    for ts, r in sorted(created_lines, key=lambda x: (x[0], x[1].get("cand_idx") or 0)):
        oid = r["object_id"]
        me = tracks.get(oid)
        p_new = _vec3(r.get("p_world"))
        if me is None or p_new is None or me.created_order is None:
            continue
        best = None
        for t, m in alive_medians:
            if t.id == oid or t.created_order is None or t.created_order >= me.created_order:
                continue
            last = t.last_ts
            alive = t.survivor or (last is not None and last >= ts - int(p.proto_ttl_s * 1e9)) or (t.created_ts is not None and t.created_ts >= ts - int(p.proto_ttl_s * 1e9))
            if not alive:
                continue
            d = float(np.linalg.norm(m - p_new))
            if d <= p.cluster_radius_m and (best is None or d < best[1]):
                best = (t, d)
        if best is None:
            continue
        t, d = best
        nn = int(r.get("n_nearby") or 0); ng = int(r.get("n_gate_survivors") or 0); mc = r.get("max_cos")
        if nn == 0:
            reason = "not_in_index"
        elif ng == 0:
            reason = "gated"
        elif isinstance(mc, (int, float)) and float(mc) < p.cos_min:
            reason = "low_similarity"
        else:
            reason = "other"
        view = idx["view"].get(ts)
        in_view = bool(view and any(e.get("id") == t.id for e in (view.get("objects") or [])))
        dup_records.append({"id": oid, "dup_of": t.id, "cluster": (f"c{cluster_of[oid]:04d}" if oid in cluster_of else None),
                            "t_sensor_ns": ts, "t_rel_s": rel(ts), "frame_seq": r.get("frame_seq"), "dist_m": _r(d, 4),
                            "reason": reason, "n_nearby": nn, "n_gate_survivors": ng, "max_cos": _r(mc), "original_in_view": in_view,
                            "original_survivor": t.survivor, "duplicate_survivor": me.survivor})
        dups_by_frame[ts] += 1

    # frames: every processed frame = a view line (one per processed frame); obs lines without a view line still count
    frame_ts = sorted(set(idx["view"]) | set(idx["obs"]))
    for ts in frame_ts:
        view = idx["view"].get(ts)
        dq = idx["dequeue"].get(ts) or {}
        shadowed = bool(dq.get("gate_shadow"))
        in_ids: List[str] = []
        depth_by_id: Dict[str, Optional[float]] = {}
        if view is not None:
            for e in (view.get("objects") or []):
                oid = e.get("id")
                if oid is None:
                    continue
                in_ids.append(str(oid))
                ed = e.get("expected_depth")
                depth_by_id[str(oid)] = float(ed) if isinstance(ed, (int, float)) and math.isfinite(float(ed)) else None
        matched_ids = {str(r["object_id"]) for r in idx["obs"].get(ts, []) if r.get("outcome") == OBS_MATCHED and r.get("object_id")}
        created_ids = {str(r["object_id"]) for r in idx["obs"].get(ts, []) if r.get("outcome") == OBS_CREATED and r.get("object_id")}
        n_obs = len(idx["obs"].get(ts, []))
        in_clusters: Dict[int, List[str]] = defaultdict(list)
        for oid in in_ids:
            ci = cluster_of.get(oid)
            if ci is None:
                unknown_view_ids += 1
                continue
            in_clusters[ci].append(oid)
        matched_clusters = {cluster_of[o] for o in matched_ids if o in cluster_of}
        created_clusters = {cluster_of[o] for o in created_ids if o in cluster_of}
        # id level (cross-checks with observation_summary's view join)
        for oid in in_ids:
            per_id_views[oid] += 1
            hit = oid in matched_ids
            if hit:
                per_id_reid[oid] += 1
            else:
                per_id_missed[oid] += 1
            d = depth_by_id.get(oid)
            if d is not None and d > 0:
                b = _range_bin(d, p.range_bin_m)
                range_tab[b]["views"] += 1
                range_tab[b]["reid" if hit else "missed"] += 1
                if not shadowed:
                    range_tab[b]["views_masked"] += 1
                    range_tab[b]["reid_masked" if hit else "missed_masked"] += 1
        # cluster level
        n_reid = n_dup = n_missed = 0
        for ci in in_clusters:
            if ci in matched_clusters:
                out = "reidentified"; n_reid += 1
            elif ci in created_clusters:
                out = "duplicated"; n_dup += 1
            else:
                out = "missed"; n_missed += 1
            presence[ci].append((ts, out, shadowed))
        n_outside = 0
        for ci in (matched_clusters | created_clusters) - set(in_clusters):
            if ci in matched_clusters:
                n_outside += 1                     # matched although the view line did not list it (a view, counted)
            presence[ci].append((ts, "reidentified" if ci in matched_clusters else "created", shadowed))
        frames.append({
            "t_sensor_ns": ts, "t_rel_s": rel(ts), "frame_seq": (view or dq or {}).get("frame_seq"),
            "is_keyframe": (view or dq or {}).get("is_keyframe"), "gate_shadow": dq.get("gate_shadow"),
            "n_obs": n_obs, "n_in_frustum_ids": len(in_ids), "n_in_frustum_clusters": len(in_clusters),
            "n_reidentified": n_reid, "n_duplicated": n_dup, "n_missed": n_missed, "n_reidentified_outside_frustum": n_outside, "n_created": len(created_ids),
            "n_duplicate_spawns": int(dups_by_frame.get(ts, 0)),
            "frame_rejected": (str(dq.get("reason")) if dq.get("outcome") == "frame_rejected" else None),
        })

    # ── detection over views (cluster level) ─────────────────────────
    det = Counter(); det_masked = Counter()
    cluster_det: Dict[int, Counter] = {c.index: Counter() for c in clusters}
    for ci, seq in presence.items():
        for ts, out, shadowed in seq:
            if out == "created":
                continue                           # the creation frame is presence, not a view
            det["views"] += 1; det[out] += 1
            cluster_det[ci]["views"] += 1; cluster_det[ci][out] += 1
            if not shadowed:
                det_masked["views"] += 1; det_masked[out] += 1
                cluster_det[ci]["views_masked"] += 1; cluster_det[ci][out + "_masked"] += 1

    def det_block(c: Counter, suffix: str = "") -> Dict[str, Any]:
        v = c.get("views" + suffix, 0)
        reid = c.get("reidentified" + suffix, 0); dup = c.get("duplicated" + suffix, 0); miss = c.get("missed" + suffix, 0)
        return {"views": int(v), "reidentified": int(reid), "duplicated": int(dup), "missed": int(miss),
                "detection_rate": _div(reid + dup, v), "reid_rate": _div(reid, v)}

    detection = {"all": det_block(det), "by_range": {}}
    for b in sorted(range_tab, key=lambda s: float(s.split("-")[0])):
        c = range_tab[b]
        detection["by_range"][b] = {"views": int(c["views"]), "reidentified": int(c["reid"]), "reid_rate": _div(c["reid"], c["views"])}
        if shadow_mode:
            detection["by_range"][b].update({"views_masked": int(c["views_masked"]), "reid_rate_masked": _div(c["reid_masked"], c["views_masked"])})
    if shadow_mode:
        detection["masked"] = det_block(det_masked)
    detection["id_level"] = {"views": int(sum(per_id_views.values())), "reidentified": int(sum(per_id_reid.values())),
                             "missed": int(sum(per_id_missed.values())), "unknown_view_ids": int(unknown_view_ids)}

    # ── revisits ─────────────────────────────────────────────────────
    gap_ns = int(p.revisit_gap_s * 1e9)
    rev = Counter()
    cluster_visits: Dict[int, dict] = {}
    for ci, seq in presence.items():
        seq = sorted(set(seq))
        visits: List[List[Tuple[int, str, bool]]] = []
        for item in seq:
            if visits and item[0] - visits[-1][-1][0] < gap_ns:
                visits[-1].append(item)
            else:
                visits.append([item])
        outs = []
        for v in visits:
            kinds_v = {o for _, o, _ in v}
            if "reidentified" in kinds_v:
                outs.append("reidentified")
            elif "duplicated" in kinds_v or "created" in kinds_v:
                outs.append("duplicated" if "duplicated" in kinds_v else "created")
            else:
                outs.append("missed")
        revisit_outs = outs[1:]
        cluster_visits[ci] = {"n_visits": len(visits), "n_revisits": len(revisit_outs),
                              "revisits": dict(Counter(revisit_outs)), "visit_starts_s": [rel(v[0][0]) for v in visits]}
        rev["n_visits"] += len(visits)
        if len(visits) > 1:
            rev["clusters_with_revisit"] += 1
        for o in revisit_outs:
            rev["n_revisits"] += 1
            rev[o] += 1
    revisits = {"clusters_with_revisit": int(rev["clusters_with_revisit"]), "n_visits": int(rev["n_visits"]), "n_revisits": int(rev["n_revisits"]),
                "reidentified": int(rev["reidentified"]), "duplicated": int(rev["duplicated"] + rev["created"]), "missed": int(rev["missed"]),
                "reid_lower_bound": _div(rev["reidentified"], rev["n_revisits"])}

    # ── labels ───────────────────────────────────────────────────────
    track_dis: Dict[str, dict] = {}
    for t in track_list:
        labs = [l for l in t.labels if l is not None]
        n = len(labs)
        if n == 0:
            track_dis[t.id] = {"n": 0, "modal": None, "modal_count": 0, "distinct": 0, "disagreement": None}
            continue
        cnt = Counter(labs)
        modal, mc = cnt.most_common(1)[0]
        track_dis[t.id] = {"n": n, "modal": modal, "modal_count": int(mc), "distinct": len(cnt), "disagreement": _r(1.0 - mc / n)}
    ge2 = [d["disagreement"] for d in track_dis.values() if d["n"] >= 2 and d["disagreement"] is not None]
    cluster_labels: Dict[int, dict] = {}
    n_cluster_conflicts = 0
    for c in clusters:
        labs = [l for oid in c.members for l in tracks[oid].labels if l is not None]
        cnt = Counter(labs)
        surv_primary = sorted({str(mem_by_id[oid].get("label_primary")) for oid in c.members if oid in mem_by_id and mem_by_id[oid].get("label_primary")})
        if len(surv_primary) > 1:
            n_cluster_conflicts += 1
        modal_count = cnt.most_common(1)[0][1] if cnt else 0
        cluster_labels[c.index] = {"n": len(labs), "top": [{"label": l, "n": int(k)} for l, k in cnt.most_common(3)], "distinct": len(cnt),
                                   "disagreement": (_r(1.0 - modal_count / len(labs)) if labs else None), "survivor_label_primary": surv_primary}
    labels = {"tracks_with_labels": sum(1 for d in track_dis.values() if d["n"] > 0), "tracks_ge2": len(ge2),
              "disagreement": _stats_r(ge2), "frac_tracks_disagreeing": _div(sum(1 for v in ge2 if v > 0), len(ge2)),
              "clusters_with_survivor_label_conflict": int(n_cluster_conflicts),
              "distinct_per_track_ge2": _stats_r(d["distinct"] for d in track_dis.values() if d["n"] >= 2)}

    # ── scatter ──────────────────────────────────────────────────────
    along_all: List[float] = []; lat_all: List[float] = []; rng_all: List[float] = []
    bins: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"along": [], "lat": []})
    track_scatter: Dict[str, dict] = {}
    for t in track_list:
        if t.n_obs < p.min_obs_scatter:
            continue
        m = t.median
        al: List[float] = []; la: List[float] = []; rg: List[float] = []
        for pt, cam, rng in zip(t.points, t.cams, t.ranges):
            resid = pt - m
            if cam is None:
                continue
            ray = m - cam
            nrm = float(np.linalg.norm(ray))
            if nrm < 1e-6:
                continue
            u = ray / nrm
            a = float(np.dot(resid, u))
            l = float(np.linalg.norm(resid - a * u))
            al.append(a); la.append(l)
            r_eff = float(rng) if rng is not None else nrm
            rg.append(r_eff)
            bins[_range_bin(r_eff, p.range_bin_m)]["along"].append(a)
            bins[_range_bin(r_eff, p.range_bin_m)]["lat"].append(l)
        if not al:
            continue
        along_all += al; lat_all += la; rng_all += rg
        track_scatter[t.id] = {"n": len(al), "along_rms_m": _r(math.sqrt(float(np.mean(np.square(al))))),
                               "lateral_rms_m": _r(math.sqrt(float(np.mean(np.square(la))))), "mean_range_m": _r(float(np.mean(rg)))}

    def _fit(y: List[float], x: List[float]) -> Dict[str, Any]:
        if len(y) < 3 or len(set(round(v, 6) for v in x)) < 2:
            return {"slope_per_m": None, "intercept_m": None, "r2": None, "n": len(y)}
        X = np.asarray(x, dtype=float); Y = np.asarray(y, dtype=float)
        b, a = np.polyfit(X, Y, 1)
        pred = a + b * X
        ss_res = float(np.sum((Y - pred) ** 2)); ss_tot = float(np.sum((Y - Y.mean()) ** 2))
        return {"slope_per_m": _r(b), "intercept_m": _r(a), "r2": (_r(1.0 - ss_res / ss_tot) if ss_tot > 0 else None), "n": len(y)}

    along_rms = math.sqrt(float(np.mean(np.square(along_all)))) if along_all else None
    lat_rms = math.sqrt(float(np.mean(np.square(lat_all)))) if lat_all else None
    scatter = {"n_tracks": len(track_scatter), "n_obs": len(along_all), "along_rms_m": _r(along_rms), "lateral_rms_m": _r(lat_rms),
               "along_over_lateral": (_r(along_rms / lat_rms) if (along_rms is not None and lat_rms) else None),
               "along_abs_vs_range": _fit([abs(v) for v in along_all], rng_all), "lateral_vs_range": _fit(lat_all, rng_all),
               "by_range": {b: {"n": len(v["along"]), "along_rms_m": _r(math.sqrt(float(np.mean(np.square(v["along"]))))),
                                "lateral_rms_m": _r(math.sqrt(float(np.mean(np.square(v["lat"])))))}
                            for b, v in sorted(bins.items(), key=lambda kv: float(kv[0].split("-")[0]))}}

    # ── memory ───────────────────────────────────────────────────────
    n_created_ids = sum(1 for t in track_list if t.n_created > 0)
    n_transient = sum(1 for t in track_list if t.n_created > 0 and not t.survivor)
    hits_hist = Counter()
    for o in mem_objs:
        h = int(o.get("hits") or 0)
        hits_hist["1" if h <= 1 else "2" if h == 2 else "3-5" if h <= 5 else "6-10" if h <= 10 else "11+"] += 1
    mem = {"objects": int(memory.get("objects_count", len(mem_objs))), "confirmed": int(memory.get("confirmed_count", sum(1 for o in mem_objs if o.get("confirmed")))),
           "fingerprint": memory.get("fingerprint"), "ids_created": int(n_created_ids), "ids_seen": len(track_list), "transient": int(n_transient),
           "survivors_single_hit": sum(1 for o in mem_objs if int(o.get("hits") or 0) <= 1),
           "matched_without_scoring": int(sum(t.n_without_scoring for t in track_list)),
           "hits_hist": {k: int(hits_hist[k]) for k in ("1", "2", "3-5", "6-10", "11+")},
           "view_bins_hist": {str(k): int(v) for k, v in sorted(Counter(int(o.get("view_bins") or 0) for o in mem_objs).items())}}

    # ── admission ────────────────────────────────────────────────────
    rx = Counter((r.get("decision"), r.get("reason")) for r in kinds.get("receiver", []))
    dq_c = Counter((r.get("outcome"), r.get("reason")) for r in kinds.get("dequeue", []))
    shadow = Counter(r.get("gate_shadow") for r in kinds.get("dequeue", []) if r.get("gate_shadow"))
    processed_ts = sorted(s for s in (_stamp(r) for r in kinds.get("dequeue", []) if r.get("outcome") == "processed") if s is not None)
    gaps = [(b - a) / 1e9 for a, b in zip(processed_ts, processed_ts[1:])]
    n_rx = sum(rx.values())
    n_processed = sum(n for (o, _), n in dq_c.items() if o == "processed")
    admission = {
        "receiver": {f"{d}:{r}" if r else str(d): int(n) for (d, r), n in sorted(rx.items(), key=lambda kv: str(kv[0]))},
        "dequeue": {f"{o}:{r}" if r else str(o): int(n) for (o, r), n in sorted(dq_c.items(), key=lambda kv: str(kv[0]))},
        "outcomes": {k: int(v) for k, v in sorted(outcome_histogram(rows).items())},
        "receiver_lines": int(n_rx), "enqueued": int(rx.get(("enqueued", ""), 0)),
        "throttled": int(rx.get(("dropped", "throttle"), 0)), "throttled_frac": _div(rx.get(("dropped", "throttle"), 0), n_rx),
        "tracking_dropped": int(rx.get(("dropped", "tracking_state"), 0)),
        "processed": int(n_processed), "keyframes_processed": int(sum(1 for r in kinds.get("dequeue", []) if r.get("outcome") == "processed" and r.get("is_keyframe"))),
        "gate_rejected": int(sum(n for (o, _), n in dq_c.items() if o == "gate_rejected")),
        "frame_rejected": int(sum(n for (o, _), n in dq_c.items() if o == "frame_rejected")),
        "processed_gap_s": _stats_r(gaps), "longest_unprocessed_s": (_r(max(gaps)) if gaps else None),
        "processed_span_s": (_r((processed_ts[-1] - processed_ts[0]) / 1e9) if len(processed_ts) > 1 else None),
        "shadowed": int(sum(shadow.values())), "shadowed_frac": _div(sum(shadow.values()), n_processed), "shadow_reasons": {str(k): int(v) for k, v in sorted(shadow.items())},
    }

    # ── pose health ──────────────────────────────────────────────────
    ph = pose_health(rows, disc_base_m=p.disc_base_m, disc_rate_mps=p.disc_rate_mps, gap_factor=p.gap_factor)
    groups = ph.get("groups") or {}
    main_group = max(groups.values(), key=lambda g: g.get("n_frames", 0)) if groups else {}
    pose = {"total": ph.get("total"), "groups": groups, "params": ph.get("params"),
            "sensor_hz": main_group.get("sensor_hz"), "span_s": main_group.get("span_s"), "jitter_ms": main_group.get("jitter_ms"),
            "n_stream": main_group.get("n_stream"), "limited_frames_frac": main_group.get("limited_frames_frac"),
            "delivery_lag_end_s": (main_group.get("delivery_lag") or {}).get("end_s"),
            "depth_valid_frac_p50": (main_group.get("depth_valid_frac") or {}).get("p50"),
            "conf2_frac_p50": (main_group.get("conf2_frac") or {}).get("p50")}

    # ── worst moments ────────────────────────────────────────────────
    moments: List[dict] = []
    for f in frames:
        score = f["n_missed"] + 2 * f["n_duplicate_spawns"] + (1 if f["frame_rejected"] else 0)
        if score > 0:
            moments.append({"kind": "frame", "t_sensor_ns": f["t_sensor_ns"], "t_rel_s": f["t_rel_s"], "frame_seq": f["frame_seq"], "score": _r(float(score), 3),
                            "detail": {"missed": f["n_missed"], "duplicate_spawns": f["n_duplicate_spawns"], "in_frustum": f["n_in_frustum_clusters"],
                                       "reidentified": f["n_reidentified"], "frame_rejected": f["frame_rejected"], "gate_shadow": f["gate_shadow"]}})
    for r in kinds.get("dequeue", []):
        if r.get("outcome") == "frame_rejected":
            ts = _stamp(r)
            if ts is not None and ts not in idx["view"] and ts not in idx["obs"]:
                moments.append({"kind": "frame_rejected", "t_sensor_ns": ts, "t_rel_s": rel(ts), "frame_seq": r.get("frame_seq"), "score": 1.0,
                                "detail": {"reason": r.get("reason")}})
    for g in groups.values():
        for gap in g.get("gaps") or []:
            ts = gap.get("after_t_sensor_ns")
            moments.append({"kind": "pose_gap", "t_sensor_ns": ts, "t_rel_s": rel(ts), "frame_seq": None, "score": _r(float(gap.get("gap_s") or 0), 3),
                            "detail": {"gap_s": gap.get("gap_s")}})
        for d in g.get("discontinuities") or []:
            ts = d.get("t_sensor_ns")
            moments.append({"kind": "pose_discontinuity", "t_sensor_ns": ts, "t_rel_s": rel(ts), "frame_seq": None, "score": _r(float(d.get("jump_m") or 0), 3),
                            "detail": {"jump_m": d.get("jump_m"), "dt_s": d.get("dt_s")}})
        for e in g.get("limited_episodes") or []:
            ts = e.get("start_t_sensor_ns")
            moments.append({"kind": "tracking_limited", "t_sensor_ns": ts, "t_rel_s": rel(ts), "frame_seq": None, "score": _r(float(e.get("n_frames") or 0), 3),
                            "detail": {"n_frames": e.get("n_frames"), "duration_s": e.get("duration_s"), "states": e.get("states")}})
    moments.sort(key=lambda m: (-(m["score"] or 0), m["t_sensor_ns"] if m["t_sensor_ns"] is not None else 0, m["kind"]))
    worst = moments[:p.worst_n]

    # ── per-cluster / per-object records ─────────────────────────────
    cluster_records: List[dict] = []
    for c in clusters:
        seq = sorted(set(presence.get(c.index, [])))
        seen = [ts for ts, o, _ in seq if o in ("reidentified", "duplicated", "created")]
        missed = [ts for ts, o, _ in seq if o == "missed"]
        cd = cluster_det[c.index]
        rec = {"cluster": c.id, "leader": c.leader, "members": list(c.members), "n_members": len(c.members),
               "n_survivors": sum(1 for oid in c.members if tracks[oid].survivor), "position": [_r(v, 4) for v in c.position],
               "n_obs": int(sum(tracks[oid].n_obs for oid in c.members)),
               "first_seen_s": rel(min((tracks[oid].first_ts for oid in c.members if tracks[oid].first_ts is not None), default=None)),
               "last_seen_s": rel(max((tracks[oid].last_ts for oid in c.members if tracks[oid].last_ts is not None), default=None)),
               "detection": det_block(cd), "visits": cluster_visits.get(c.index, {"n_visits": 0, "n_revisits": 0, "revisits": {}, "visit_starts_s": []}),
               "labels": cluster_labels.get(c.index),
               "moments": {"n_seen": len(seen), "n_missed": len(missed), "seen_s": [rel(t) for t in seen[:p.moments_cap]], "missed_s": [rel(t) for t in missed[:p.moments_cap]]}}
        if shadow_mode:
            rec["detection_masked"] = det_block(cd, "_masked")
        cluster_records.append(rec)
    object_records: List[dict] = []
    for t in sorted(track_list, key=lambda t: (t.first_ts if t.first_ts is not None else float("inf"),
                                               t.first_order if t.first_order is not None else float("inf"))):
        o = mem_by_id.get(t.id) or {}
        d = track_dis.get(t.id) or {}
        s = track_scatter.get(t.id) or {}
        object_records.append({
            "id": t.id, "survivor": t.survivor, "cluster": (f"c{cluster_of[t.id]:04d}" if t.id in cluster_of else None),
            "n_obs": t.n_obs, "n_matched": t.n_matched, "n_created": t.n_created, "first_seen_s": rel(t.first_ts), "last_seen_s": rel(t.last_ts),
            "position": ([_r(v, 4) for v in t.median] if t.median is not None else None),
            "views": int(per_id_views.get(t.id, 0)), "reidentified": int(per_id_reid.get(t.id, 0)), "missed": int(per_id_missed.get(t.id, 0)),
            "label_modal": d.get("modal"), "n_distinct_labels": d.get("distinct"), "label_disagreement": d.get("disagreement"),
            "along_rms_m": s.get("along_rms_m"), "lateral_rms_m": s.get("lateral_rms_m"), "mean_range_m": s.get("mean_range_m"),
            "hits": (int(o.get("hits")) if o.get("hits") is not None else None), "confirmed": (bool(o.get("confirmed")) if o else None),
            "label_primary": o.get("label_primary"), "created_audit": (t.created_audit or None),
        })

    duplicates = {"n": len(dup_records), "rate_of_created": _div(len(dup_records), n_created_ids),
                  "original_in_view": int(sum(1 for d in dup_records if d["original_in_view"])),
                  "reasons": {k: int(sum(1 for d in dup_records if d["reason"] == k)) for k in DUP_REASONS}}
    cluster_block = {"radius_m": p.cluster_radius_m, "n": len(clusters), "n_with_survivor": sum(1 for c in clusters if any(tracks[o].survivor for o in c.members)),
                     "n_multi_member": sum(1 for c in clusters if len(c.members) > 1),
                     "duplicates_all": int(len([t for t in track_list if t.median is not None]) - len(clusters)),
                     "duplicates_survivors": int(sum(1 for t in track_list if t.survivor and t.median is not None) - sum(1 for c in clusters if any(tracks[o].survivor for o in c.members))),
                     "n_at_radius": n_at_radius, "tracks_without_position": sum(1 for t in track_list if t.median is None)}

    scalars = _flatten({
        "memory": {k: v for k, v in mem.items() if k not in ("fingerprint", "hits_hist", "view_bins_hist")},
        "clusters": {k: v for k, v in cluster_block.items() if k != "n_at_radius"},
        "clusters.n_at_radius": n_at_radius,
        "detection": {"views": detection["all"]["views"], "reidentified": detection["all"]["reidentified"], "duplicated": detection["all"]["duplicated"],
                      "missed": detection["all"]["missed"], "detection_rate": detection["all"]["detection_rate"], "reid_rate": detection["all"]["reid_rate"],
                      **({"masked": detection["masked"]} if shadow_mode else {})},
        "labels": {"tracks_ge2": labels["tracks_ge2"], "disagreement_mean": labels["disagreement"]["mean"], "disagreement_p50": labels["disagreement"]["p50"],
                   "disagreement_p95": labels["disagreement"]["p95"], "frac_tracks_disagreeing": labels["frac_tracks_disagreeing"],
                   "clusters_with_survivor_label_conflict": labels["clusters_with_survivor_label_conflict"]},
        "scatter": {"n_tracks": scatter["n_tracks"], "n_obs": scatter["n_obs"], "along_rms_m": scatter["along_rms_m"], "lateral_rms_m": scatter["lateral_rms_m"],
                    "along_over_lateral": scatter["along_over_lateral"], "along_slope_per_m": scatter["along_abs_vs_range"]["slope_per_m"],
                    "along_intercept_m": scatter["along_abs_vs_range"]["intercept_m"], "along_r2": scatter["along_abs_vs_range"]["r2"],
                    "lateral_slope_per_m": scatter["lateral_vs_range"]["slope_per_m"]},
        "duplicates": {"n": duplicates["n"], "rate_of_created": duplicates["rate_of_created"], "original_in_view": duplicates["original_in_view"], **{f"reason.{k}": v for k, v in duplicates["reasons"].items()}},
        "revisits": revisits,
        "admission": {"receiver_lines": admission["receiver_lines"], "enqueued": admission["enqueued"], "throttled": admission["throttled"], "throttled_frac": admission["throttled_frac"],
                      "tracking_dropped": admission["tracking_dropped"], "processed": admission["processed"], "keyframes_processed": admission["keyframes_processed"],
                      "gate_rejected": admission["gate_rejected"], "frame_rejected": admission["frame_rejected"], "processed_gap_p50_s": admission["processed_gap_s"]["p50"],
                      "processed_gap_p95_s": admission["processed_gap_s"]["p95"], "longest_unprocessed_s": admission["longest_unprocessed_s"],
                      "processed_span_s": admission["processed_span_s"], "shadowed": admission["shadowed"], "shadowed_frac": admission["shadowed_frac"]},
        "pose": {"n_frames": (ph.get("total") or {}).get("n_frames"), "sensor_hz": pose["sensor_hz"], "span_s": pose["span_s"], "jitter_ms": pose["jitter_ms"],
                 "n_gaps": (ph.get("total") or {}).get("n_gaps"), "n_limited_episodes": (ph.get("total") or {}).get("n_limited_episodes"),
                 "n_discontinuities": (ph.get("total") or {}).get("n_discontinuities"), "pose_errors": (ph.get("total") or {}).get("pose_errors"),
                 "limited_frames_frac": pose["limited_frames_frac"],
                 "depth_valid_frac_p50": pose["depth_valid_frac_p50"], "conf2_frac_p50": pose["conf2_frac_p50"]},
        "moments": {"n": len(moments), "worst_score": (worst[0]["score"] if worst else 0.0)},
    })

    return {
        "schema": METRICS_SCHEMA, "params": asdict(p), "shadow_mode": bool(shadow_mode), "t0_sensor_ns": t0,
        "scalars": scalars,
        "memory": mem, "clusters_summary": cluster_block, "detection": detection, "labels": labels, "scatter": scatter,
        "duplicates": duplicates, "revisits": revisits, "admission": admission, "pose": pose,
        "worst_moments": worst, "clusters": cluster_records, "objects": object_records, "frames": frames, "duplicate_spawns": dup_records,
        "observation_summary": observation_summary(rows),
    }


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Nested dicts -> dotted keys; only numbers / None are kept (strings and lists are not scalars)."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            out[key] = _r(v)
        elif isinstance(v, bool):
            out[key] = int(v)
    return out
