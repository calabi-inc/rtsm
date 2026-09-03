# Session 2026-09-02 — Layout L2 — trial manifest (IN PROGRESS: 7/12 rows)

Operator: solo. Same roster, objects repositioned from L1. Session ran
2026-09-02 ~00:50–01:30, ended early on phone degradation (Lens
delivering pose 2.3–2.6 s stale → relocations fail closed → baseline
spun in place). Resume plan at bottom.

## Sheet mapping (L2 table order — sheet prescribes SEARCH-FIRST)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| 1 | search | water bottle | t20260902-005340-005 | arrived | 54.4 | ___ | ran AFTER row 2 (order deviation L2-D1) |
| 2 | memory | water bottle | t20260902-005209-004 | arrived | 38.9 | ___ | |
| 3 | search | dumbbell | t20260902-005647-007 | drift | — | ___ | VALID FAILURE: drift guard tripped (believed 1.73 m > best 0.77 + 0.8); counts against baseline; not infra (3 later trials ran clean) |
| 4 | memory | dumbbell | t20260902-005532-006 | arrived | 33.4 | ___ | |
| 5 | search | tissue box | t20260902-010250-009 | arrived | 57.5 | ___ | |
| 6 | memory | tissue box | t20260902-010138-008 | arrived | 26.5 | ___ | |
| 7 | search | scissors | PENDING (redo) | — | — | — | 3 infra voids below |
| 8 | memory | scissors | t20260902-010440-010 | arrived | 33.2 | ___ | |
| 9 | search | teddy bear | PENDING | — | — | — | |
| 10 | memory | teddy bear | PENDING | — | — | — | |
| 11 | search | water bottle (repeat) | PENDING | — | — | — | |
| 12 | memory | water bottle (repeat) | PENDING | — | — | — | |

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

## Tape + photos for rows 1–6, 8

PENDING operator answer: whether tapes/stop photos were taken during
the session. Convention (amendment 2026-09-02): bumper tip → floor
cross, 0 = contact, +18 cm rig constant. If no records exist, those
rows stay TTA-valid; tape-TCR marks them untaped (excluded from tape
denominator, disclosed) — TTA is the primary pre-registered claim.

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
