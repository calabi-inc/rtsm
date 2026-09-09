# Session 2026-09-02 (evening) — Layout L3 — trial manifest (COMPLETE: 12/12 rows)

Operator: solo. Objects repositioned from L2, tape crosses laid.
Session ~21:00–22:20 in two blocks (thermal pause mid-session).
L3 sheet order is MEMORY-FIRST (matches printed counterbalancing).

## Sheet mapping (L3 table order)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| 1 | memory | dumbbell | t20260902-212227-001 | arrived | 34.6 | 18 (0) | 2nd attempt (void below); CLIP-net record, arbiter picked on image despite label_primary "water bottle" |
| 2 | search | dumbbell | t20260902-212400-002 | drift | — | 51 (33) | VALID FAILURE, see L3-D3: correct bearing, depth-error centroid (Y=-0.14 ≈ 1 m above floor), car overran/CRASHED THROUGH the object at speed, ended 33 cm past its cross; operator safety-stop; monitor issued drift. Counts against baseline; not redone. Crash photo taken. Tape belongs to any-verdict secondary only (51 cm converted — over threshold anyway). |
| 3 | memory | tissue box | t20260902-213005-003 | arrived | 36.5 | 18 (0) | arbiter took candidate 2 (rank deference) |
| 4 | search | tissue box | t20260902-213132-004 | arrived | 152.5 | 36 (18) | ~131 s search w/ relocation; no contact |
| 5 | memory | scissors | t20260902-213550-005 | arrived | 40.0 | 18 (0) | rank-1 ghost ("blurry wall") rejected on image |
| 6 | search | scissors | t20260902-214503-007 | arrived | 365.1 | 18 (0) | 2nd attempt (void below); longest successful search of campaign; 1 blocked-walk skip |
| 7 | memory | teddy bear | t20260902-215836-009 | arrived | 33.9 | 18 (0) | 2nd attempt (void below); rows 7–12 on scan #3 (post-cooldown) |
| 8 | search | teddy bear | t20260902-215948-010 | arrived | 60.4 | 25 (7) | ~36 s sweep; live re-sight of same record (conservative masking mechanic); no contact |
| 9 | memory | water bottle | t20260902-220203-011 | not_found | — | — | VALID MEMORY FAILURE (first of campaign): arbiter refused dark/ambiguous stored crop from rushed scan-#3 bottle look; car never moved (0 ticks). Memory quality = scan quality. Not redone. No tape/photo (car never moved). |
| 10 | search | water bottle | t20260902-220317-012 | arrived | 30.7 | 18 (0) | baseline beat memory on this pair — fresh close-range crop accepted instantly; honest asymmetry, reported as-is |
| 11 | memory | dumbbell (repeat) | t20260902-220821-013 | drift | — | — | VALID MEMORY FAILURE: association drift under approach — re-observations swept in adjacent furniture (snapshots 1–2 show dumbbell as corner sliver), centroid dragged to Y=-0.50; operator grab prevented hard crash into furniture; monitor issued drift. Tape N/A (ended far off). |
| 12 | search | dumbbell (repeat) | t20260902-221901-015 | drift | — | — | 2nd attempt (void below). VALID BASELINE FAILURE: live observation, correct object, same corrupted-centroid divergence (best 0.47 → 1.29 m); operator-witnessed IDENTICAL trajectory to row 11 — same crash site (adjacent furniture), operator grab before hard impact. Proves the dumbbell failure is in the SHARED perception layer (detector box spill on partial views + sparse depth on black plates → centroid in the furniture), not the memory layer. Tape N/A. |

Tape column = converted protocol metric `gap + 18` cm (raw gap in
parentheses; 0 = nose contact). Dictated by operator 2026-09-02 late
evening: rows 2/4/8 non-zero (33/18/7 raw), rows 9/11/12 N/A, all
other arrivals contact. All 8 arrivals within the 50 cm threshold
(max 36 cm, also within the 40 cm sensitivity band). Values written
into trial JSONLs (`tape_gap_cm`/`tape_cm`).

Voided / non-sheet runs (excluded from rows):

| Trial file | What happened |
|-----------|---------------|
| t20260902-211258-001 | memory/dumbbell → stale_stop at believed 1.04 m: Lens froze mid-drive, ARKit session lost (frame_epoch 1→2) → full rescan. Correct pick ("black weight plates on a bar"). |
| t20260902-213718-006 | search/scissors → stale_stop 75 s into search: ~3 s pose-delivery stall (Lens app alive, same epoch; likely WiFi/throttle, operator confirmed app never froze). |
| t20260902-215221-008 | memory/bear → stale_stop at believed 1.99 m: blind 2.8 s; feed then died fully (Lens death, session paused). Correct pick. |
| t20260902-221439-014 | search/dumbbell repeat → stale_stop 6 ticks into sweep: pose-delivery stall (warm phone, post-cooldown degradation band 2.2–2.5 s); same epoch, map survived. |

## Protocol deviations this session

- **L3-D1 — scan gate failures on the dumbbell:** two scans produced
  zero-to-one GDINO dumbbell detections. Fixed PRE-TRIAL (legal —
  layout not yet frozen) by rotating the dumbbell plates-out + a 10 s
  steady dwell; record minted via the CLIP safety net and was
  eyeball-verified from its judgment crop before any trial. Ghost
  records (blurred floor/wall frames minted as "water bottle") noted;
  image arbiter rejected one live in row 5.
- **L3-D2 — Lens death after row 1's first attempt** bumped the ARKit
  frame epoch → map invalidated → RTSM restart + full rescan (rows
  1–6 ran on scan #2 of the layout; rows 7–12 on scan #3 after the
  thermal pause). frame_epoch is logged per trial for audit.
- **L3-D3 — row 2 crash-through (valid failure, fully disclosed):**
  the baseline locked a live detection of the real dumbbell with a
  depth-corrupted centroid (~1 m above floor, range never converging),
  so the slow zone (keyed on believed distance) never engaged and the
  car physically overran the object at speed, ending 33 cm past its
  cross; the operator grabbed it to prevent damage. The monitor
  independently issued the drift verdict. Classified VALID baseline
  failure: the failure was fully materialized before the touch, and
  both interpretations (system drift verdict / operator stop of an
  overrun) are failures. This is the 08-30 amendment's "contact
  severity mitigation, not prevention" clause demonstrated. Dumbbell
  restored to its tape cross afterward.
- **L3-D4 — accelerating pose-delivery stalls** (25/14/7 min apart,
  and again ~18 min after the cooldown): classic thermal pattern.
  The pose-age gate before every fire caught none of these (all
  fired healthy); the stalls hit mid-trial.
- **L3-D5 — e-stop waiver continues** (see L2-D5): no PS4, no wired
  pad, operator physical power-cut only, ≤0.08 m/s rig. Operator
  hand-grabs served as de facto e-stop twice (rows 11, 12).
- **L3-D6 — dumbbell = depth/association-adversarial object (both
  arms):** four dumbbell failures share one mechanism — black
  IR-absorbing plates + adjacent wooden furniture → partial-object
  observations associate in background structure → corrupted centroid
  (recorded Y up to ~1 m above floor) → drift/overrun. Failures: L2
  search (drift/crash-through, 17 cm past cross), L3 row 2 search
  (crash-through, 33 cm past cross), L3 row 11 memory (drift,
  operator grab), L3 row 12 search (drift, operator grab, same crash
  site as row 11). Arrivals (L2 memory, L3 row 1 memory) came from
  records minted off deliberate stationary dwells. Symmetric across
  arms — a shared-perception-layer limitation (detector box geometry
  + depth sparsity, NOT recognition: every failure named the object
  correctly), not a memory-layer one. Also: the pre-trial gate can
  verify a record's crop but NOT its depth/coordinate quality —
  honest scope limit.
- **L3-D7 — L3 ran across three scans** (#1 pre-Lens-death aborted
  after row 1's first attempt; #2 rows 1–6; #3 rows 7–12 after
  cooldown). Row 9's memory failure traces to scan #3's rushed bottle
  look. frame_epoch logged per trial.

## Photos

Operator to transfer with the day's batch: stop photos for the 8
arrivals, crash photo(s) for rows 2/11/12, L2+L3 layout shots.
Convert HEIC → JPEG and STRIP EXIF/GPS before any publication.
