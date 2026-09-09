# Pre-campaign trials (EXCLUDED from E1 analysis)

Everything here predates the frozen campaign apparatus:

- `t20260811-*`, `t20260815-*`, `t20260828-*` — development and
  calibration runs under earlier code/configs.
- `t20260830-*` — the L1 SHAKEDOWN session, excluded from analysis by
  the pre-registered protocol amendment 2026-08-30 (it motivated the
  selection-model and perception-package changes).

`aggregate.py` scans `paper/demo2_data/*.jsonl` non-recursively, so
this folder is invisible to the analysis by construction. Kept for
failure-taxonomy material and audit, not for statistics.

Campaign trials (2026-09-01 onward, frozen stack) live one level up.
