# Session 2026-09-03 — Layout L4 — trial manifest (COMPLETE: 12/12 rows)

Operator: solo, afternoon session (~15:20–17:15) in three thermal
blocks. Dumbbell placed on OPEN FLOOR (L3-D6 lesson) — decisive, see
L4-D5. Operator standing order (2026-09-03): MEMORY-FIRST within
pairs, campaign-wide (sheet printed search-first; see L4-D1).

## Sheet mapping (running order; sheet row in parens)

| Row | Cond | Goal | Trial file | Result | TTA (s) | Tape (cm) | Notes |
|----|------|------|-----------|--------|---------|-----------|-------|
| (2) | memory | tissue box | t20260903-152321-001 | arrived | 40.7 | ___ | arbiter skipped ghost record, picked real box |
| (1) | search | tissue box | t20260903-153951-003 | arrived | 87.2 | ___ | 2nd attempt (livelock void below) |
| (4) | memory | scissors | t20260903-154311-004 | arrived | 29.2 | ___ | |
| (3) | search | scissors | t20260903-154518-005 | arrived | 120.2 | ___ | fastest scissors search of campaign |
| (6) | memory | teddy bear | t20260903-154926-006 | arrived | 31.8 | ___ | |
| (5) | search | teddy bear | t20260903-160622-001 | arrived | 58.0 | ___ | post-reset scan #2; operator observed enlarged standoff (pose drift) — tape decides |
| (8) | memory | water bottle | t20260903-161105-002 | not_found | — | — | VALID MEMORY FAILURE (2nd of campaign, replicates L3): thin 3-hit bottle record, arbiter refused all crops; 0 ticks. Operator requested redo; REFUSED per symmetry rule. |
| (7) | search | water bottle | t20260903-161429-003 | arrived | 71.3 | ___ | live crop accepted instantly — L3 bottle asymmetry replicated in full |
| (10) | memory | dumbbell | t20260903-161635-004 | arrived | 43.2 | ___ | first GDINO-native dumbbell record of campaign (6 hits, 0.93); open-floor placement |
| (9) | search | dumbbell | t20260903-162431-006 | arrived | 87.8 | ___ | 2nd attempt (void below). FIRST successful dumbbell search of campaign (was 0-for-3) — open floor fixed the centroid corruption for BOTH arms |
| (12) | memory | tissue box (repeat) | t20260903-163005-007 | not_found | — | — | VALID MEMORY FAILURE (3rd): GHOST DOMINANCE — mat+trash-bag corner minted 41-hit stab-1.00 "tissue box" record (crop = gray mat, operator-confirmed); real box unminted; arbiter correctly refused to drive to a mat. |
| (11) | search | tissue box (repeat) | t20260903-170714-011 | not_found | — | — | 4th attempt (3 voids below). FIRST GENUINE SEARCH-CAP TIMEOUT of campaign: full 480 s exhausted, last standpoint judged no-match; feed healthy throughout. VALID BASELINE FAILURE. Repeat pair failed in BOTH arms (mem: ghost dominance; search: cap). |

Voided / non-sheet runs (excluded from rows):

| Trial file | What happened |
|-----------|---------------|
| t20260903-152523-002 | search/tissue → operator soft-stop ~5.7 min: GHOST-LABEL LIVELOCK — GDINO hallucinated "tissue box" on furniture 27 rounds straight; each early-exit rejection re-observed in place (ghosts re-mint with new ids, defeating the rejected-id mask); sweep never completed → never relocated. Peek-suppression did not break the cycle (round state resets on rejection — apparatus anomaly, logged for POST-campaign investigation, no mid-campaign fix). Bounded by the 480 s cap had it run. |
| t20260903-161809-005 | search/dumbbell → stale_stop 40 ticks in (thermal window closed). |
| t20260903-165030-008 | search/tissue repeat → stale_stop mid-DRIVE at believed 1.23 m after 222 s search had FOUND the real box — transient ~2.5 s stall at the worst moment. |
| t20260903-165638-009 | search/tissue repeat → blocked 0.21 m: locked a tissue-lookalike OUTSIDE the venue (elevated ~0.65 m on furniture, ~5 m out; arbiter: "purple box with tissue pulled from opening"); wall guard saved the furniture. VENUE-SIGHT-LINE VOID (new deviation flavor, see L4-D6); lookalike hidden, redo. |
| t20260903-170451-010 | search/tissue repeat → stale_stop 9 ticks in (WiFi micro-stall; feed recovered seconds later, same epoch). |

## Protocol deviations this session

- **L4-D1 — order amendment (operator standing order 2026-09-03):**
  MEMORY-FIRST within every pair, campaign-wide, overriding the
  printed search-first counterbalancing (L2/L4 sheets). Consequence:
  the campaign has NO search-first layouts; condition-order
  counterbalancing is abandoned. Defense: masked retrieval makes the
  baseline mechanically independent of trial order; uniform order
  removes order variance between layouts; disclosed as a limitation.
- **L4-D2 — three scans/epochs:** scan #1 (rows 2,1,4,3,6 sheet-order
  numbering), thermal reset → scan #2 (rows 5,8,7,10,9,12), Lens
  relaunch → epoch 3 map RESET ONLY, NO rescan for the final row (11)
  — legal because condition (b) is masked from the store by design
  and no memory rows remained. Scanless-search note in row 11.
- **L4-D3 — phone thermal duty cycle ~20-25 min** with in-place
  recovery twice (Lens alive → map survived) and two full resets.
  Micro-stalls (~2.5 s) killed two otherwise-healthy trials.
- **L4-D4 — memory-bottle failure replicated** (L3→L4, same
  mechanism: thin scan record of the transparent bottle → arbiter
  refuses). Now a 2-datapoint pattern: memory quality = scan quality.
- **L4-D5 — dumbbell open-floor placement fixed BOTH arms** (mem 43.2
  s + search 87.8 s, first search success after 0-for-3): completes
  L3-D6 — the depth/association adversary was the
  placement-x-furniture interaction, not the object.
- **L4-D6 — outside-venue lookalike contamination** (new flavor):
  roster-lookalike objects VISIBLE from inside the venue but outside
  the drivable boundary can be locked by search (unwinnable by
  design; wall guard blocked). L5 setup rule: sweep sight lines
  beyond the boundary for lookalikes, not just inside.
- **L4-D7 — scan-gate lesson:** the 41-hit stab-1.00 "tissue box"
  record was a MAT (ghost dominance; operator-confirmed crop).
  Numbers lie; pixels don't. Gate now requires eyeballing ALL FIVE
  crops every scan (was: only historically-bad objects).
- **L4-D8 — e-stop waiver continues** (L2-D5/L3-D5 lineage).

## Tape + photos

PENDING operator batch: tapes for the 9 arrivals (gap convention,
0 = contact, +18 cm; failures need none) + stop photos + today's
layout shot. Note for sheet-row (5) search bear: operator observed
enlarged standoff (pose drift) — tape number decides pass/fail
honestly (believed-arrived-but-tape-failed bucket exists for this).
