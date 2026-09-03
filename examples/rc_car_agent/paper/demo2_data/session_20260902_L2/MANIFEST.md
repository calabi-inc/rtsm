# Session 2026-09-02 — Layout L2 — trial manifest (COMPLETE: 12/12 rows)

Operator: solo. Same roster, objects repositioned from L1. Session ran
2026-09-02 ~00:50–01:30, ended early on phone degradation (Lens
delivering pose 2.3–2.6 s stale → relocations fail closed → baseline
spun in place). Resume plan at bottom.

## Sheet mapping (L2 table order — sheet prescribes SEARCH-FIRST)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| 1 | search | water bottle | t20260902-005340-005 | arrived | 54.4 | 18 (0) | ran AFTER row 2 (order deviation L2-D1) |
| 2 | memory | water bottle | t20260902-005209-004 | arrived | 38.9 | 18 (0) | |
| 3 | search | dumbbell | t20260902-005647-007 | drift | — | 35 (17) | VALID FAILURE: drift guard tripped (believed 1.73 m > best 0.77 + 0.8). Operator account (2026-09-02, with L3's crashes in hand): car physically crashed THROUGH the dumbbell and ended 17 cm past its cross — same corrupted-centroid mechanism as L3-D6. Tape reported under any-verdict secondary only; drift ≠ completion. |
| 4 | memory | dumbbell | t20260902-005532-006 | arrived | 33.4 | 18 (0) | |
| 5 | search | tissue box | t20260902-010250-009 | arrived | 57.5 | 23 (5) | |
| 6 | memory | tissue box | t20260902-010138-008 | arrived | 26.5 | 18 (0) | |
| 7 | search | scissors | t20260902-204701-001 | arrived | 196.8 | 18 (0) | redo after 3 infra voids (below); ran on scan #3, rested phone; 184 s searching before lock |
| 8 | memory | scissors | t20260902-010440-010 | arrived | 33.2 | 18 (0) | |
| 9 | search | teddy bear | t20260902-205349-003 | arrived | 49.3 | 18 (0) | scan #3; found during sweep (~32 s search), ran after row 10 |
| 10 | memory | teddy bear | t20260902-205204-002 | arrived | 24.1 | 18 (0) | scan #3; locked in ~4 s; ran before row 9 (memory-first kept for all of L2, see L2-D1) |
| 11 | search | water bottle (repeat) | t20260902-205701-005 | arrived | 53.7 | 18 (0) | scan #3; ran after row 12; live re-sight of the scan record during sweep (see masking note below) |
| 12 | memory | water bottle (repeat) | t20260902-205530-004 | arrived | 38.9 | 18 (0) | scan #3; ran before row 11; repeat consistency: row 2 (scan #1, prev. day) also 38.9 s |

Tape column = converted protocol metric `gap + 18` cm (raw gap in
parentheses; 0 = nose contact). Dictated by operator 2026-09-02
(evening) covering all 12 rows; contact rows corroborated by stop
photos where taken. 11/11 arrivals within the 50 cm threshold (max
23 cm); row 3's 35 cm belongs to a drift failure and does not count
as completion.

Voided / non-sheet runs (excluded from rows):

| Trial file | What happened |
|-----------|---------------|
| t20260902-010546-011 | search/scissors → stale_stop: pose feed died during search. Infra void (phone degradation). |
| t20260902-010754-012 | search/scissors → operator-cancelled: spun in place, no exploration (stale clearance stamps fail relocation closed). |
| t20260902-011944-013 | search/scissors → operator-cancelled: same spin symptom; session ended. Root cause confirmed: Lens freeze/degraded delivery on hot phone. |

## Protocol deviations this session (so far)

- **L2-D1 — within-pair order ran MEMORY-FIRST** (matching L1's habit)
  instead of the sheet's prescribed search-first counterbalancing.
  Masked retrieval makes condition order mechanically independent
  (baseline candidates never come from the map), and objects are only
  moved between layouts; disclosed as an order-effects caveat. Rows
  9–12 + the row-7 redo will follow the sheet's order as printed.
- **L2-D2 — session split across two scans:** rows 1–6, 8 ran on scan
  #1 (2026-09-02 ~00:50); the remaining rows will run on a fresh scan
  at resume (RTSM was shut down). Same disclosure pattern as L1-D6.
- **L2-D3 — three row-7 infra voids** (table above): phone thermal/
  uptime degradation, not a search-algorithm failure. Fix = phone rest
  + charge; pose age verified < 2.0 s before every fire at resume.
  Resume (2026-09-02 evening) confirmed the diagnosis: same trial on a
  rested phone arrived cleanly (row 7, 196.8 s).
- **L2-D4 — resume rows ran on scan #3** (2026-09-02 ~20:40): fresh
  RTSM process, persisted stale index cleared via /reset before scan
  (52 old vectors from scan #2's frame). Scan #3 verified scissors /
  teddy bear / water bottle (the only objects remaining rows target);
  dumbbell and tissue box were not re-verified — their rows completed
  on scan #1. Same disclosure pattern as L1-D6.
- **L2-D5 — e-stop waived for the resume session by operator
  instruction:** no PS4 controller detected, wired pad not restored;
  `require_verified_estop: false` (local). Operator's kill = physical
  power cut, standing within reach. Rig speed ≤ 0.08 m/s. Recorded
  for the paper's deviations note.

## Masking audit note (rows 9, 11)

Both baseline trials locked the SAME record ids the memory arm used
(bear b92819a1, bottle 0488f12d). This is not a masking violation:
the freshness window admits only records upserted during the current
round, so these entered as LIVE re-sightings during the sweep — the
object was physically observed from the current standpoint, which is
exactly what a memoryless agent gets. The one asymmetry (the record's
consolidated multi-view coordinate) ASSISTS the baseline arm, i.e. it
is conservative w.r.t. the memory-advantage claim. Row 7's search
locked a fresh record (3d647131), showing the mechanism takes fresh
mints when the old record is not re-sighted first.

## Tape + photos

RESOLVED 2026-09-02 (evening): operator dictated the full 12-row set
(table above). Convention: bumper gap, 0 = contact, +18 cm rig
constant, written into each trial JSONL (`tape_gap_cm`/`tape_cm`).
Stop photos: operator to transfer with the day's photo batch.

## Resume checklist (before any trial)

1. Objects UNTOUCHED since yesterday (operator confirms).
2. Phone rested/charged; pose age < 2.0 s verified before every fire.
3. Wired e-stop pad restored (D4 follow-up from L1).
4. Fresh RTSM process + scan; scan check must fire all 5 objects.
5. Remaining rows in sheet order: 7 (search scissors), 9 (search
   bear), 10 (memory bear), 11 (search bottle repeat), 12 (memory
   bottle repeat).
6. Tape + photo EVERY row: gap to floor cross, "0 contact", laid-tape
   photo when not touching.
