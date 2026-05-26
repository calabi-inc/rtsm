"""
RTSM evaluation utilities — diagnostics, harness, metrics, reports.

See `.claude/plans/permanent-plan/eval-paper-plan-2026.md` for the full plan.

Phase 0 (this module's current scope):
- event_log: append-only JSONL writer for per-frame diagnostic events.
  Used together with FilterDiagnostics (rtsm/utils/mask_staging.py) and
  ScoringTrace (rtsm/core/pipeline.py).
"""
