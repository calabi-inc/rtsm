"""
RTSM evaluation utilities — diagnostics, harness, metrics, reports.

See `.claude/plans/permanent-plan/eval-paper-plan-2026.md` for the full plan.

Current scope:
- event_log: append-only JSONL frame-flow trace (schema_version 2) — one
  line per receiver decision, per dequeued frame (gate outcome + reason) and
  per processed frame (FilterDiagnostics from rtsm/utils/mask_staging.py and
  ScoringTrace from rtsm/core/pipeline.py). Off by default
  (cfg.diagnostics.enabled); the record P1's determinism gate and the P2
  ledgers build on.
"""
