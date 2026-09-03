# Session 2026-09-01 — Layout L1 (candidate) — trial manifest

Operator: solo. Roster: teddy bear, water bottle, dumbbell, tissue box,
scissors. Start pose: taped X + arrow, constant all session. Roster
objects untouched all session; one NON-roster distractor (purple wipes
box, on furniture) was hidden after trial row 7's two blocked attempts.

## Sheet mapping (L1 table order)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| 1 | memory | teddy bear | t20260901-192904-001 | arrived | 38.9 | ___ | |
| 2 | search | teddy bear | t20260901-193125-002 | arrived | 37.9 | ___ | early-exit step 3 |
| 3 | memory | water bottle | t20260901-193406-003 | arrived | 33.8 | ___ | |
| 4 | search | water bottle | t20260901-193526-004 | arrived | 55.8 | ___ | 2 no-match rejections then correct pick |
| 5 | memory | dumbbell | t20260901-193729-005 | arrived | 25.1 | ___ | found via CLIP label; GDINO silent; image-confirmed |
| 6 | search | dumbbell | t20260901-193831-006 | arrived | 34.5 | ___ | early-exit step 5 |
| 7 | memory | tissue box | t20260901-195220-002 | arrived | 30.1 | ___ | THIRD attempt; see deviations D1–D3 |
| 8 | search | tissue box | t20260901-195410-003 | arrived | 55.5 | ___ | 1 rejection then correct pick |
| 9 | memory | scissors | t20260901-195737-004 | arrived | 26.3 | ___ | label_primary wrong ("water bottle"); image won |
| 10 | search | scissors | t20260901-200333-006 | arrived | 252.8 | ___ | no-match @ standpoint 1 → relocate → found; see D4 |
| 11 | memory | teddy bear (repeat) | t20260902-004117-002 | arrived | 34.8 | ___ | ran on scan #2 (fresh map, see D6); pick verified pre-fire |
| 12 | search | teddy bear (repeat) | t20260902-004235-003 | arrived | 40.5 | ___ | early-exit step 3; fresh detection 0.67 |

Voided / non-sheet runs (kept for failure analysis, excluded from rows):

| Trial file | What happened |
|-----------|---------------|
| t20260901-194040-007 | memory/tissue box → blocked 0.18 m (wall guard save). Cause: picked non-roster purple wipes box (real second tissue-class object in view). |
| t20260901-194555-001 | memory/tissue box → blocked 0.26 m (guard save). Same purple box picked after roster box's snapshot degraded by drive-by re-observation. |
| t20260901-195854-005 | search/scissors → cancelled by operator: ESP32 WiFi drop mid-run, physical power-cut stop. |
| t20260901-201056-007 | memory/bear repeat (1st attempt) → believed-arrived at a DRIFTED DUPLICATE record ~1 m off (map polluted by ~18 drives); operator judged wrong-direction failure → voided, map reset (D5/D6). |
| t20260902-003935-001 | memory/bear repeat (2nd attempt) → stale_stop: phone pose feed froze 2.6 s mid-drive; monitor safe-stopped at 1.92 m. Infra void; stream recovered. |

## Protocol deviations this session (report in the paper's deviations note)

- **D1 — selection-rule change mid-session:** rank-deference tie-break
  (commit 3877f91) landed between row 6 and row 7 after the purple-box
  mis-picks. Rows 1–6 ran pre-fix, rows 7–12 post-fix. All other code
  identical (stack frozen at 1504b39/88a61fe otherwise).
- **D2 — venue change mid-session:** non-roster purple wipes box hidden
  after the two voided tissue-box attempts. Roster objects untouched.
- **D3 — map resets mid-session:** /reset + rescan after D2, and again
  before the row-11 redo (map pollution: ~18 drives minted duplicate
  records with drifted coordinates — 5 "teddy bear" entries at 2+
  locations). Campaign rule going forward: one reset+scan per evening,
  12 drives max.
- **D4 — one hardware void:** ESP32 WiFi drop during first row-10
  attempt (physical power-cut; firmware watchdog behavior uncertain —
  wired e-stop pad to be restored before further sessions).
- **D5 — row 11's first attempt voided** (wrong direction: drifted
  duplicate coordinate ~1 m off; believed-arrived 0.40 m; map had
  accumulated 5 bear records across ~18 drives).
- **D6 — rows 11–12 ran on a SECOND scan** (map reset after D5 + an
  iPhone charge break, session resumed 2026-09-02 ~00:40). Scan #2 was
  bear-complete but thin elsewhere (scissors unscanned — irrelevant to
  rows 11–12). Memory repeat consistency held ACROSS scans: row 1
  38.9 s (scan #1) vs row 11 34.8 s (scan #2); search rows 2/12:
  37.9/40.5 s. Campaign rule reaffirmed: one scan per evening, ≤12
  drives per map.

## Photos

Operator folder: `C:\Users\konam\Desktop\E1_data` — 12 stop photos +
1 layout wide shot (HEIC; convert to JPEG + STRIP EXIF/GPS before any
publication). Row mapping (photo timestamps corroborate trial logs):

row 1 L1_bear_mem · row 2 L1_bear_search · row 3 L1_water_bottle_mem ·
row 4 L1_water_bottle_search · row 5 L1_dumbbell_mem ·
row 6 L1_dumbbelll_search (sic) · row 7 L1_tissue_box_mem ·
row 8 L1_tissue_box_search · row 9 L1_scissors_mem ·
row 10 L1_scissors_search · row 11 L1_bear_mem_repeat ·
row 12 L1_bear_search_repeat · layout L1_layout.

## Tape numbers — PROVISIONAL (operator-dictated 2026-09-02)

Readings appear to be BUMPER-TO-OBJECT GAP, not the protocol's
wheel-center-to-base (ten zeros are impossible for the latter given
the 0.40 m believed stop radius). PENDING: operator confirms
measurement origin + one-time rig constant (wheel-center → front
bumper, measure once on the parked car). Protocol tape = gap +
constant, applied uniformly and documented here.

Gap readings by row: 1: 0 · 2: 19 · 3: 0 · 4: 0 · 5: 0 · 6: 0 ·
7: 0 · 8: 1 · 9: 0 · 10: 0 · 11: 0 · 12: 0 (cm).
All rows pass the 50 cm success threshold under any plausible
constant (max = 19 + const).
