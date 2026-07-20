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

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from config import Config, load_config
from rtsm_client import PoseSample, RtsmClient, SemanticHit

_SYSTEM_PROMPT = (
    "You are the target-selection step of a robot navigation planner. "
    "Given the user's goal and candidate objects from the robot's spatial "
    "memory (id, label, similarity score, confirmed flag, stability), pick "
    "the single best target. Prefer confirmed, high-stability, high-score "
    "candidates whose label matches the goal. Respond ONLY via the tool."
)

_SELECT_TOOL = {
    "name": "select_target",
    "description": "Select the single best target object for the goal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "target_id": {"type": "string", "description": "id of the chosen candidate"},
            "reason": {"type": "string", "description": "one short sentence"},
        },
        "required": ["target_id"],
    },
}


@dataclass(frozen=True)
class Candidate:
    id: str
    label: Optional[str]
    score: float
    confirmed: bool
    stability: float
    xyz_world: List[float]


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
    reason: Optional[str] = None     # not_found detail or LLM's one-liner


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
                     anthropic_client) -> Optional[tuple]:
    """Forced-tool Haiku pick. Returns (target_id, reason) or None on any
    failure — caller falls back deterministically. Never raises."""
    try:
        lines = [
            f"- id={c.id} label={c.label or 'unknown'} score={c.score:.4f} "
            f"confirmed={c.confirmed} stability={c.stability:.2f}"
            for c in candidates
        ]
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
            messages=[{
                "role": "user",
                "content": f"Goal: {goal}\nCandidates:\n" + "\n".join(lines),
            }],
            timeout=cfg.planner.api_timeout_s,
        )
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                tid = str(block.input.get("target_id", ""))
                reason = block.input.get("reason")
                return tid, (str(reason) if reason else None)
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


def plan(goal: str, rtsm: RtsmClient, cfg: Config,
         anthropic_client=None, use_llm: bool = True) -> PlanResult:
    """One-shot plan: query RTSM, pick a target, return it with provenance."""
    query = extract_query(goal)
    res = rtsm.semantic_query(query, top_k=5)

    hits = _eligible(res.results)
    plan_epoch = res.robot_pose.frame_epoch if res.robot_pose else None
    if not hits:
        detail = "no results" if not res.results else "no candidate has a 3D position"
        return PlanResult(status="not_found", goal=goal, query=query,
                          plan_pose=res.robot_pose, frame_epoch=plan_epoch,
                          reason=detail)

    candidates = [
        Candidate(
            id=h.id,
            label=rtsm.get_object_label(h.id),   # best-effort enrichment
            score=h.score,
            confirmed=h.confirmed,
            stability=h.stability,
            xyz_world=list(h.xyz_world or []),
        )
        for h in hits
    ]
    by_id = {c.id: c for c in candidates}

    picked: Optional[Candidate] = None
    planner_path = "top1_no_llm"
    reason: Optional[str] = None

    client = anthropic_client if anthropic_client is not None else (
        _default_anthropic_client(cfg) if use_llm else None
    )
    if client is not None:
        out = _pick_with_haiku(candidates, goal, cfg, client)
        if out is not None and out[0] in by_id:
            picked = by_id[out[0]]
            planner_path = "haiku"
            reason = out[1]
        else:
            planner_path = "top1_fallback"   # LLM tried: failed or invalid id

    if picked is None:
        picked = candidates[0]               # ranked top-1 (confirmed first)

    return PlanResult(
        status="ok", goal=goal, query=query,
        target_id=picked.id, label=picked.label,
        xyz_world=picked.xyz_world, score=picked.score,
        confirmed=picked.confirmed, stability=picked.stability,
        planner_path=planner_path, plan_pose=res.robot_pose,
        frame_epoch=plan_epoch, reason=reason,
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
