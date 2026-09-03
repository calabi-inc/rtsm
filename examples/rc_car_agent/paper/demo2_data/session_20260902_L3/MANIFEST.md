# Session 2026-09-02 (evening) — Layout L3 — trial manifest (IN PROGRESS: 6/12 rows)

Operator: solo. Objects repositioned from L2, tape crosses laid.
Session ~21:00–22:00, paused on accelerating Lens delivery stalls
(25 → 14 → 7 min intervals — thermal curve). Resume plan at bottom.
L3 sheet order is MEMORY-FIRST (matches printed counterbalancing).

## Sheet mapping (L3 table order)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| 1 | memory | dumbbell | t20260902-212227-001 | arrived | 34.6 | ___ | 2nd attempt (void below); CLIP-net record, arbiter picked on image despite label_primary "water bottle" |
| 2 | search | dumbbell | t20260902-212400-002 | drift | — | — | VALID FAILURE, see L3-D3: correct bearing, depth-error centroid (Y=-0.14 ≈ 1 m above floor), car overran/crashed through the object at speed, operator safety-stop; monitor issued drift. Counts against baseline; not redone. Crash photo taken. |
| 3 | memory | tissue box | t20260902-213005-003 | arrived | 36.5 | ___ | arbiter took candidate 2 (rank deference) |
| 4 | search | tissue box | t20260902-213132-004 | arrived | 152.5 | ___ | ~131 s search w/ relocation |
| 5 | memory | scissors | t20260902-213550-005 | arrived | 40.0 | ___ | rank-1 ghost ("blurry wall") rejected on image |
| 6 | search | scissors | t20260902-214503-007 | arrived | 365.1 | ___ | 2nd attempt (void below); longest successful search of campaign; 1 blocked-walk skip |
| 7 | memory | teddy bear | PENDING (redo) | — | — | — | 1st attempt infra void below |
| 8 | search | teddy bear | PENDING | — | — | — | |
| 9 | memory | water bottle | PENDING | — | — | — | |
| 10 | search | water bottle | PENDING | — | — | — | |
| 11 | memory | dumbbell (repeat) | PENDING | — | — | — | dumbbell restored to its cross after row-2 crash |
| 12 | search | dumbbell (repeat) | PENDING | — | — | — | |

Voided / non-sheet runs (excluded from rows):

| Trial file | What happened |
|-----------|---------------|
| t20260902-211258-001 | memory/dumbbell → stale_stop at believed 1.04 m: Lens froze mid-drive, ARKit session lost (frame_epoch 1→2) → full rescan. Correct pick ("black weight plates on a bar"). |
| t20260902-213718-006 | search/scissors → stale_stop 75 s into search: ~3 s pose-delivery stall (Lens app alive, same epoch; likely WiFi/throttle, operator confirmed app never froze). |
| t20260902-215221-008 | memory/bear → stale_stop at believed 1.99 m: blind 2.8 s; feed then died fully (Lens death, session paused). Correct pick. |

## Protocol deviations this session (so far)

- **L3-D1 — scan gate failures on the dumbbell:** two scans produced
  zero-to-one GDINO dumbbell detections. Fixed PRE-TRIAL (legal —
  layout not yet frozen) by rotating the dumbbell plates-out + a 10 s
  steady dwell; record minted via the CLIP safety net and was
  eyeball-verified from its judgment crop before any trial. Ghost
  records (blurred floor/wall frames minted as "water bottle") noted;
  image arbiter rejected one live in row 5.
- **L3-D2 — Lens death after row 1's first attempt** bumped the ARKit
  frame epoch → map invalidated → RTSM restart + full rescan (L3 rows
  ran on scan #2 of the layout; rows 7–12 will run on scan #3 at
  resume). frame_epoch is logged per trial for audit.
- **L3-D3 — row 2 crash-through (valid failure, fully disclosed):**
  the baseline locked a live detection of the real dumbbell with a
  depth-corrupted centroid (~1 m above floor, range never converging),
  so the slow zone (keyed on believed distance) never engaged and the
  car physically overran the object at speed; the operator grabbed it
  to prevent damage. The monitor independently issued the drift
  verdict. Classified VALID baseline failure: the failure was fully
  materialized before the touch, and both interpretations (system
  drift verdict / operator stop of an overrun) are failures. This is
  the 08-30 amendment's "contact severity mitigation, not prevention"
  clause demonstrated. Dumbbell restored to its tape cross afterward.
- **L3-D4 — accelerating pose-delivery stalls** (25/14/7 min apart)
  ended the session after row 7's attempt; classic thermal pattern.
  Rule reaffirmed: pose age gate before every fire caught none of
  these (all fired healthy); the stalls hit mid-trial.
- **L3-D5 — e-stop waiver continues** (see L2-D5): no PS4, no wired
  pad, operator physical power-cut only, ≤0.08 m/s rig.

## Resume checklist

1. Phone fully cooled + charged, Low Power Mode off.
2. RTSM: /reset (or fresh process) → verify indexed 0 → rescan L3
   (objects UNTOUCHED except dumbbell restored to its cross; slow
   dwell on dumbbell, bottle, tissue box).
3. Five-object verification incl. eyeballing dumbbell crop.
4. Agent server up + preflight + pose gate.
5. Rows in order: 7 (mem bear redo), 8, 9, 10, 11, 12.
6. Tapes: operator batches numbers at day end (gap convention,
   0 = contact, +18 cm rig constant); stop photos per trial;
   crash photo from row 2 to be transferred with them.
