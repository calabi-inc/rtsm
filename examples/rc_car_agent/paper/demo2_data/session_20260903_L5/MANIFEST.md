# Session 2026-09-03/04 (night) — Layout L5 — trial manifest (COMPLETE: 12/12 — CAMPAIGN COMPLETE 60/60)

Operator: solo, ~23:30–00:20. The campaign's only PERFECT layout:
12 arrivals, 0 failures, 1 infra void. Sheet order = memory-first
(matches operator standing order natively). Dumbbell open floor,
sight-line sweep done (L4-D6 rule), all five crops eyeballed at gate
(L4-D7 rule) — the full lesson stack, and it shows.

## Sheet mapping (L5 table order)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| 1 | memory | scissors | t20260903-233912-001 | arrived | 31.3 | 18 (0) | premium-dwell record (18 dets, 0.97) |
| 2 | search | scissors | t20260903-234029-002 | arrived | 486.1 | 26 direct | LONGEST successful search of campaign (447 s searching, locked just inside cap). Car also contacted wall at stop; operator safety-pull BEFORE measuring → tape ~26 cm measured drive-center→cross DIRECTLY (no +18), approximate. Lock reused the cat-tree ghost id after live re-association overwrote its crop — arbiter judged pixels, correct outcome. |
| 3 | memory | teddy bear | t20260903-235351-003 | arrived | 36.2 | 18 (0) | rank-1 blurry rejected, rank-2 taken |
| 4 | search | teddy bear | t20260903-235515-004 | arrived | 44.6 | 18 (0) | narrowest bear gap of campaign (1.2×) |
| 5 | memory | water bottle | t20260903-235647-005 | arrived | 28.2 | 23 direct | BOTTLE CURSE BROKEN: first memory-bottle arrival since L2 — the 10-hit crisp-crop record did it. Controlled contrast with L3/L4 failures: scan quality was the whole story. |
| 6 | search | water bottle | t20260903-235827-006 | arrived | 41.9 | 18 (0) | memory won a bottle pair for the first time in 3 layouts |
| 7 | memory | dumbbell | t20260903-235956-007 | arrived | 37.5 | 18 (0) | open floor; no drift |
| 8 | search | dumbbell | t20260904-000112-008 | arrived | 71.8 | 18 (0) | 2nd consecutive layout with clean dumbbell pair |
| 9 | memory | tissue box | t20260904-000311-009 | arrived | 25.1 | 18 (0) | straight to the gate-eyeballed sharp record |
| 10 | search | tissue box | t20260904-000423-010 | arrived | 30.1 | 22 direct | spotted in first sweep steps |
| 11 | memory | scissors (repeat) | t20260904-000601-011 | arrived | 34.5 | 18 (0) | repeat consistency: row 1 was 31.3 s, same record |
| 12 | search | scissors (repeat) | t20260904-001209-013 | arrived | 373.0 | 18 (0) | 2nd attempt (void below); ran SCANLESS post-reset (L4 precedent — condition (b) masked from store; empty-map preflight guard satisfied by a 10 s ambient pan, panned records inert to masked search). 354 s searching. Scissors search variance: 486/373 s. |

Voided / non-sheet runs (excluded from rows):

| Trial file | What happened |
|-----------|---------------|
| t20260904-000718-012 | search/scissors repeat → stale_stop 4 ticks in: Lens died at the ~28 min thermal mark (feed 30+ s stale, epoch survived until relaunch). Relaunch → RTSM restart → /reset → scanless redo. |

## Protocol deviations this session

- **L5-D1 — row 12 ran scanless** on a reset, empty map after the
  Lens death (L4-D2 precedent): legal because condition (b) is masked
  from the store; the agent server's empty-map preflight guard was
  satisfied with a ~10 s ambient pan whose proto records are inert to
  the masked search (freshness window admits only the trial's own
  sweep observations).
- **L5-D2 — row 2 wall contact + operator safety pull before tape**
  (row note): tape recorded as direct center-to-cross ~26 cm,
  approximate, convention flagged in the trial JSONL.
- **L5-D3 — e-stop waiver continues** (L2-D5 lineage), full campaign
  ran without wired pad or verified gamepad estop; operator physical
  intervention used 3× total (L3×2, L4 crash... see prior manifests)
  — none in L5.
- **L5-D4 — full lesson-stack gate** (sight-line sweep, open-floor
  dumbbell, premium dwells, all-five crop eyeballs) preceded the only
  zero-failure layout — plausibly causal, honestly confounded with a
  rested phone and practiced operator.

## Tape + photos

RESOLVED 2026-09-04: full set dictated (contact everywhere except row 2 = 26 direct, row 5 = 23 direct, row 10 = 22 direct). All 12 rows pass the 50 cm threshold. Photos pending transfer.
