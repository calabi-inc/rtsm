"""
Planner — E1 condition (a): one RTSM query, one LLM target pick.

Locked design (code-plan-demo2-rc-car.md §5.0, 2026-06-28):
  * ONE /search/semantic call assembles the snapshot (results + robot_pose).
  * A single Claude Haiku forced-tool call picks the target from the
    candidates — `tool_choice` forces `select_target{target_id}` so the
    model can ONLY answer with an id we gave it (grounded; a hallucinated
    id is rejected and falls back). Stateless; ~1 s per trial.
  * Deterministic top-1 fallback whenever the LLM path is unavailable or
    misbehaves. `planner_path` records which path ran (logged per trial):
        haiku         — LLM picked, id validated
        top1_fallback — LLM attempted but failed/invalid → ranked top-1
        top1_no_llm   — no API key / client → ranked top-1 directly
  * frame_epoch (resolved 2026-07-20, rtsm c75e787): RTSM serves a
    world-frame discontinuity counter on every robot_pose. The plan records
    the plan-time value (`frame_epoch`, may be None on epoch-less paths)
    plus `plan_pose` — the robot pose snapshot from the SAME query
    response — so the monitor can gate on epoch equality (ints only) with
    the pose-discontinuity heuristic as belt-and-braces.

The planner decides INTENT only (which object). It computes no motion —
heading/distance are nav's deterministic geometry (invariant 3).
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from config import Config, load_config
from rtsm_client import (PoseSample, RtsmClient, SemanticHit,
                         SemanticResult)

logger = logging.getLogger("rc_car_agent.planner")

_SYSTEM_PROMPT = (
    "You are the target-selection step of a robot navigation planner. "
    "Given the user's goal and candidate objects from the robot's spatial "
    "memory (id, label, similarity score, confirmed flag, stability, and "
    "usually the candidate's latest camera crop), pick the single best "
    "target. When a candidate has an image, the IMAGE is the primary "
    "evidence — labels come from an automatic captioner and are often "
    "wrong for the right object (a real teddy bear has been labeled "
    "'audio' and 'bichon'), so decide from what the crop actually shows "
    "and treat the label as a hint only. Prefer confirmed, high-stability "
    "candidates. If NO candidate could plausibly be the goal object "
    "(check every image first), set no_match=true instead of settling for "
    "the best-scoring wrong object — driving to a wrong object is worse "
    "than admitting the target is not visible. Respond ONLY via the tool."
)

_SELECT_TOOL = {
    "name": "select_target",
    "description": ("Select the single best target object for the goal, or "
                    "declare that no candidate plausibly matches it."),
    "input_schema": {
        "type": "object",
        "properties": {
            "target_id": {"type": "string",
                          "description": "id of the chosen candidate; omit "
                                         "when no_match is true"},
            "no_match": {"type": "boolean",
                         "description": "true when NO candidate could "
                                        "plausibly be the goal object"},
            "reason": {"type": "string", "description": "one short sentence"},
        },
        "required": [],
    },
}

# Sentinel returned by _pick_with_haiku when the model declares no_match —
# distinct from None (call failed), which falls back to ranked top-1.
NO_MATCH = "__no_match__"


@dataclass(frozen=True)
class Candidate:
    id: str
    label: Optional[str]
    score: float
    confirmed: bool
    stability: float
    xyz_world: List[float]
    last_seen_wall_utc: Optional[float] = None   # observation provenance


@dataclass(frozen=True)
class PlanResult:
    status: str                      # "ok" | "not_found"
    goal: str
    query: str
    target_id: Optional[str] = None
    label: Optional[str] = None
    xyz_world: Optional[List[float]] = None
    score: float = 0.0
    confirmed: bool = False
    stability: float = 0.0
    planner_path: str = "none"       # haiku | top1_fallback | top1_no_llm | none
    plan_pose: Optional[PoseSample] = None
    frame_epoch: Optional[int] = None  # plan-time epoch (from plan_pose)
    # When the chosen target was last observed (server wall clock) — makes
    # the memory condition's coordinate age at plan time AUDITABLE (the
    # map keeps ingesting during trials; reviewers can check whether plans
    # ran on scan-age or approach-refreshed coordinates).
    target_last_seen_wall_utc: Optional[float] = None
    reason: Optional[str] = None     # not_found detail or LLM's one-liner
    # Which retrieval sources produced the candidate set (2026-08-28):
    # "label" | "semantic" | "label+semantic" — same audit field the
    # baseline stamps per round, so both arms are inspectable.
    retrieval: Optional[str] = None


def extract_query(goal: str) -> str:
    """'go to the red mug' -> 'red mug'; anything else passes through."""
    g = goal.strip()
    for prefix in ("go to the ", "go to ", "navigate to the ", "navigate to "):
        if g.lower().startswith(prefix):
            return g[len(prefix):].strip()
    return g


def _eligible(hits: List[SemanticHit]) -> List[SemanticHit]:
    """Navigable candidates only: must have a 3D position. Keeps FAISS
    ranking order; prefers confirmed hits by stable partition."""
    with_xyz = [h for h in hits if h.xyz_world is not None]
    confirmed = [h for h in with_xyz if h.confirmed]
    unconfirmed = [h for h in with_xyz if not h.confirmed]
    return confirmed + unconfirmed


def _pick_with_haiku(candidates: List[Candidate], goal: str, cfg: Config,
                     anthropic_client,
                     snapshots: Optional[dict] = None) -> Optional[tuple]:
    """Forced-tool Haiku pick. Returns (target_id, reason), (NO_MATCH,
    reason) when the model declares nothing plausibly matches, or None on
    any failure — caller falls back deterministically. Never raises.
    `snapshots`: optional {id: base64 JPEG} — candidates with a crop are
    judged visually (labels lie; see _SYSTEM_PROMPT)."""
    try:
        snapshots = snapshots or {}
        content: list = [{"type": "text", "text": f"Goal: {goal}\nCandidates:"}]
        for c in candidates:
            content.append({
                "type": "text",
                "text": f"- id={c.id} label={c.label or 'unknown'} "
                        f"score={c.score:.4f} confirmed={c.confirmed} "
                        f"stability={c.stability:.2f}",
            })
            b64 = snapshots.get(c.id)
            if b64:
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg",
                               "data": b64},
                })
        resp = anthropic_client.messages.create(
            model=cfg.planner.model,
            max_tokens=200,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_SELECT_TOOL],
            tool_choice={"type": "tool", "name": "select_target"},
            messages=[{"role": "user", "content": content}],
            timeout=cfg.planner.api_timeout_s,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                reason = block.input.get("reason")
                reason = str(reason) if reason else None
                if block.input.get("no_match"):
                    return NO_MATCH, reason
                tid = str(block.input.get("target_id", ""))
                return tid, reason
        return None
    except Exception:  # noqa: BLE001 — LLM path must never break planning
        return None


def _default_anthropic_client(cfg: Config):
    """Real client iff the SDK and an API key are available, else None."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:  # noqa: BLE001
        return None


def select_target_from_hits(hits, goal: str, rtsm: RtsmClient, cfg: Config,
                            anthropic_client=None, use_llm: bool = True):
    """THE target-selection rule, shared by BOTH E1 conditions so the
    comparison masks persistence and nothing else: eligibility filter
    (3D position, confirmed-first), best-effort label enrichment, one
    forced-tool Haiku pick over the given candidates, deterministic
    ranked-top-1 fallback.

    Returns (picked: Candidate, planner_path, reason); picked is None with
    planner_path "haiku_no_match" when the LLM explicitly declared that no
    candidate plausibly matches the goal (respected, NOT overridden by the
    top-1 fallback — settling on a wrong object is the failure mode this
    exists to prevent, observed live 2026-08-11: baseline drove at a
    smartphone for 160 s on goal 'teddy bear'). Returns None only when no
    candidate is even eligible. `hits` is whatever candidate set the
    caller is ALLOWED to see — all of memory for condition (a), the
    freshness-gated currently-visible set for condition (b)."""
    eligible = _eligible(hits)
    if not eligible:
        return None
    candidates = [
        Candidate(
            id=h.id,
            label=rtsm.get_object_label(h.id),   # best-effort enrichment
            score=h.score,
            confirmed=h.confirmed,
            stability=h.stability,
            xyz_world=list(h.xyz_world or []),
            last_seen_wall_utc=h.last_seen_wall_utc,
        )
        for h in eligible
    ]
    by_id = {c.id: c for c in candidates}

    picked: Optional[Candidate] = None
    planner_path = "top1_no_llm"
    reason: Optional[str] = None

    client = anthropic_client if anthropic_client is not None else (
        _default_anthropic_client(cfg) if use_llm else None
    )
    if client is not None:
        snapshots = None
        if cfg.planner.include_snapshots:
            snapshots = {
                c.id: rtsm.get_object_snapshot_b64(c.id)
                for c in candidates[:cfg.planner.snapshot_max_candidates]
            }
        out = _pick_with_haiku(candidates, goal, cfg, client,
                               snapshots=snapshots)
        if out is not None and out[0] == NO_MATCH:
            return None, "haiku_no_match", out[1]   # respected, no fallback
        if out is not None and out[0] in by_id:
            picked = by_id[out[0]]
            planner_path = "haiku"
            reason = out[1]
        else:
            planner_path = "top1_fallback"   # LLM tried: failed or invalid id

    if picked is None:
        picked = candidates[0]               # ranked top-1 (confirmed first)
    return picked, planner_path, reason


def query_memory(rtsm: RtsmClient, cfg: Config, query: str, top_k: int):
    """The SHARED retrieval policy (2026-08-28, reviewed same day), used
    by BOTH conditions so the comparison masks persistence and nothing
    else: the UNION of detector-label search (prompted-vocabulary labels
    are the reliable signal on this rig; reaches protos) and semantic
    embedding search, label hits ranked first, deduped by id, capped at
    top_k. A fall-back-on-miss predicate was reviewed out — one
    contaminated label hit would have suppressed the semantic candidates
    entirely, and condition (a) is one-shot with no masking loop to
    self-heal. Returns (SemanticResult, retrieval_path). A label-side
    error alone (older server) degrades to pure semantic, logged."""
    lres = None
    if cfg.planner.retrieval == "label_first":
        try:
            lres = rtsm.label_query(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001 — label side is optional
            logger.warning("label_query failed (%s: %s) — semantic only",
                           type(e).__name__, e)
    res = rtsm.semantic_query(query, top_k=top_k)
    if lres is None or not lres.results:
        return res, "semantic"
    seen = {h.id for h in lres.results}
    merged = list(lres.results) + [h for h in res.results
                                   if h.id not in seen]
    merged = merged[:top_k]
    path = "label+semantic" if len(merged) > len(lres.results[:top_k]) \
        else "label"
    return SemanticResult(query=res.query, robot_pose=res.robot_pose,
                          results=merged), path


def plan(goal: str, rtsm: RtsmClient, cfg: Config,
         anthropic_client=None, use_llm: bool = True) -> PlanResult:
    """One-shot plan: query RTSM, pick a target, return it with provenance."""
    query = extract_query(goal)
    res, retrieval_path = query_memory(rtsm, cfg, query, top_k=5)

    plan_epoch = res.robot_pose.frame_epoch if res.robot_pose else None
    sel = select_target_from_hits(res.results, goal, rtsm, cfg,
                                  anthropic_client=anthropic_client,
                                  use_llm=use_llm)
    if sel is None:
        detail = "no results" if not res.results else "no candidate has a 3D position"
        return PlanResult(status="not_found", goal=goal, query=query,
                          plan_pose=res.robot_pose, frame_epoch=plan_epoch,
                          reason=detail, retrieval=retrieval_path)
    picked, planner_path, reason = sel
    if picked is None:                       # LLM declared no plausible match
        return PlanResult(status="not_found", goal=goal, query=query,
                          planner_path=planner_path, plan_pose=res.robot_pose,
                          frame_epoch=plan_epoch,
                          reason=reason or "no candidate plausibly matches",
                          retrieval=retrieval_path)

    return PlanResult(
        status="ok", goal=goal, query=query,
        target_id=picked.id, label=picked.label,
        xyz_world=picked.xyz_world, score=picked.score,
        confirmed=picked.confirmed, stability=picked.stability,
        planner_path=planner_path, plan_pose=res.robot_pose,
        frame_epoch=plan_epoch,
        target_last_seen_wall_utc=picked.last_seen_wall_utc,
        reason=reason, retrieval=retrieval_path,
    )


if __name__ == "__main__":
    goal_arg = sys.argv[1] if len(sys.argv) > 1 else "go to the red mug"
    config = load_config()
    result = plan(goal_arg, RtsmClient(config.rtsm.url), config)
    print(f"goal:         {result.goal!r}  (query: {result.query!r})")
    print(f"status:       {result.status}" + (f"  ({result.reason})" if result.reason else ""))
    if result.status == "ok":
        print(f"target:       {result.target_id}  label={result.label}  "
              f"score={result.score:.4f}  confirmed={result.confirmed}")
        print(f"xyz_world:    {result.xyz_world}")
        print(f"planner_path: {result.planner_path}")
    print(f"plan_pose:    {'present' if result.plan_pose else 'None (no frames yet)'}")
    sys.exit(0 if result.status == "ok" else 2)
