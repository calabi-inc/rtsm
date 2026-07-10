"""
Live LLM smoke test — verifies the real Haiku forced-tool path end to end.

NOT part of pytest (needs network + ANTHROPIC_API_KEY). Run before trial
days to confirm the condition-(a) planner path is healthy:

    .venv/Scripts/python.exe smoke_llm.py

Cost per run: ~$0.001. Exercises: API auth, model id, forced tool_choice,
prompt-cache flag, grounding (the pick must be one of OUR candidate ids),
and the planner's _pick_with_haiku code path.
"""

from __future__ import annotations

import os
import sys
import time

from config import load_config
from planner import Candidate, _SELECT_TOOL, _SYSTEM_PROMPT, _pick_with_haiku

CANDIDATES = [
    Candidate(id="obj_7", label="mug", score=0.1412, confirmed=True,
              stability=0.71, xyz_world=[0.5, 0.8, 1.5]),
    Candidate(id="obj_2", label="cup", score=0.2000, confirmed=False,
              stability=0.25, xyz_world=[2.0, 0.1, 2.0]),
    Candidate(id="obj_4", label="backpack", score=0.0300, confirmed=True,
              stability=0.66, xyz_world=[-1.0, 0.2, 0.5]),
]
GOAL = "go to the red mug"


def main() -> int:
    cfg = load_config()  # also loads .env if present
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("FAIL: ANTHROPIC_API_KEY not set (setx + new terminal, or .env)")
        return 1

    import anthropic
    client = anthropic.Anthropic()

    # Step 1 — raw call, so auth/model/billing errors surface loudly
    # (the planner's own path swallows errors by design and falls back).
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=cfg.planner.model,
            max_tokens=200,
            system=[{"type": "text", "text": _SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            tools=[_SELECT_TOOL],
            tool_choice={"type": "tool", "name": "select_target"},
            messages=[{"role": "user", "content":
                       f"Goal: {GOAL}\nCandidates:\n" + "\n".join(
                           f"- id={c.id} label={c.label} score={c.score:.4f} "
                           f"confirmed={c.confirmed} stability={c.stability:.2f}"
                           for c in CANDIDATES)}],
            timeout=cfg.planner.api_timeout_s,
        )
    except Exception as e:  # noqa: BLE001 — smoke test: show the real error
        print(f"FAIL: raw API call errored: {type(e).__name__}: {e}")
        return 1
    dt_ms = (time.perf_counter() - t0) * 1000.0

    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        print("FAIL: no tool_use block despite forced tool_choice")
        return 1
    picked = block.input.get("target_id")
    print(f"raw call:     model={cfg.planner.model}  {dt_ms:.0f} ms  "
          f"tokens in/out={resp.usage.input_tokens}/{resp.usage.output_tokens}")
    print(f"picked:       {picked}  reason={block.input.get('reason')!r}")

    grounded = picked in {c.id for c in CANDIDATES}
    sensible = picked == "obj_7"   # the confirmed mug is the obvious answer

    # Step 2 — through the planner's own code path
    out = _pick_with_haiku(CANDIDATES, GOAL, cfg, client)
    path_ok = out is not None and out[0] in {c.id for c in CANDIDATES}
    print(f"planner path: {'ok' if path_ok else 'FAILED'}"
          + (f"  -> {out[0]}" if out else ""))

    if grounded and path_ok:
        note = "" if sensible else "  (note: grounded but picked a non-obvious id)"
        print(f"\nPASS{note}")
        return 0
    print("\nFAIL: ungrounded pick or planner path broke")
    return 1


if __name__ == "__main__":
    sys.exit(main())
