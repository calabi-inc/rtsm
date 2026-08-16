# E1 Campaign Runbook — simple operator guide

Follow this top to bottom. Claude does all computer work; you do hands
and tape measure. Full rules live in E1_PROTOCOL.md — this is the
follow-along version.

**The plan: 5 evenings. Each evening = 1 layout = 12 trials. 60 total.**

Objects (same 5 all campaign, locked 2026-08-15 after Check A): **teddy
bear, soda can, water bottle, scissors, humidifier**.

---

## Evening setup (do once per evening, ~20 min)

- [ ] 1. Charge check: car battery full, phone charged, (controller wired if using)
- [ ] 2. Place the 5 objects at NEW spots for this layout
      - spread them out (at least 50 cm apart)
      - on the floor / low, so the car camera can see them
      - they stay EXACTLY there all evening — never touch them again
- [ ] 3. Tape an X on the floor = car start spot. Add an arrow = car direction.
      Same X, same arrow, every trial tonight.
- [ ] 4. Take ONE wide photo of the whole room setup
- [ ] 5. Tell Claude "layout L_ setup done" → Claude starts RTSM
- [ ] 6. Phone: Low Power Mode OFF, Auto-Lock NEVER, start Lens, stream
- [ ] 7. Scan walk: one slow lap around the area, then look at each of the
      5 objects from ~1 m, two angles each. About 3 minutes.
- [ ] 8. Say "scan done" → Claude checks: all 5 objects findable, pose
      feed healthy, car reachable, battery voltage
- [ ] 8b. Wall guard: automatic — the car senses walls with the phone's
      depth camera and turns away on its own during search. Nothing for
      you to do. (Corner capture is gone.)
- [ ] 9. Claude starts the agent server. If using the controller: press X
      once when asked
- [ ] 10. Claude says "ready for trial 1" → go

---

## One trial (5–15 min each)

- [ ] 1. Car on the X, nose along the arrow
- [ ] 2. Phone mounted on car, still streaming
- [ ] 3. Say "trial N ready" → Claude sends the command (see tonight's
      table for which condition + object)
- [ ] 4. HANDS OFF. Watch. Only touch the car if it's about to break
      something (that voids the trial — we redo it, no big deal)
- [ ] 5. Car stops → wait for Claude's verdict message
- [ ] 6. TAPE MEASURE: from the point between the car's wheels to the
      bottom of the target object. Say the number in cm.
      Do this EVERY trial where the car ended anywhere near the object.
- [ ] 7. One photo of where the car stopped
- [ ] 8. Put car back on the X. Next trial.

**If the phone disconnects or Lens crashes mid-trial:** say so. Claude
marks it INVALID, we redo the same trial. It never counts against us.

**Battery rule:** if Claude reports car battery below 7.0 V — finish the
current PAIR of trials, then stop and charge.

**Stopping early is fine** — but only between pairs, and we finish the
layout next time before touching any object.

---

## Trial tables (Claude tracks these too — just follow along)

"memory" = car plans from the scan. "search" = car must find it live.
Each object gets a back-to-back pair (memory then search, or flipped).

### Layout L1 (starts with memory)
| # | condition | goal | tape cm | done |
|---|-----------|------|---------|------|
| 1 | memory | teddy bear | | |
| 2 | search | teddy bear | | |
| 3 | memory | soda can | | |
| 4 | search | soda can | | |
| 5 | memory | water bottle | | |
| 6 | search | water bottle | | |
| 7 | memory | scissors | | |
| 8 | search | scissors | | |
| 9 | memory | humidifier | | |
| 10 | search | humidifier | | |
| 11 | memory | teddy bear (repeat) | | |
| 12 | search | teddy bear (repeat) | | |

### Layout L2 (starts with search)
| # | condition | goal | tape cm | done |
|---|-----------|------|---------|------|
| 1 | search | soda can | | |
| 2 | memory | soda can | | |
| 3 | search | water bottle | | |
| 4 | memory | water bottle | | |
| 5 | search | scissors | | |
| 6 | memory | scissors | | |
| 7 | search | humidifier | | |
| 8 | memory | humidifier | | |
| 9 | search | teddy bear | | |
| 10 | memory | teddy bear | | |
| 11 | search | soda can (repeat) | | |
| 12 | memory | soda can (repeat) | | |

### Layout L3 (starts with memory)
| # | condition | goal | tape cm | done |
|---|-----------|------|---------|------|
| 1 | memory | water bottle | | |
| 2 | search | water bottle | | |
| 3 | memory | scissors | | |
| 4 | search | scissors | | |
| 5 | memory | humidifier | | |
| 6 | search | humidifier | | |
| 7 | memory | teddy bear | | |
| 8 | search | teddy bear | | |
| 9 | memory | soda can | | |
| 10 | search | soda can | | |
| 11 | memory | water bottle (repeat) | | |
| 12 | search | water bottle (repeat) | | |

### Layout L4 (starts with search)
| # | condition | goal | tape cm | done |
|---|-----------|------|---------|------|
| 1 | search | scissors | | |
| 2 | memory | scissors | | |
| 3 | search | humidifier | | |
| 4 | memory | humidifier | | |
| 5 | search | teddy bear | | |
| 6 | memory | teddy bear | | |
| 7 | search | soda can | | |
| 8 | memory | soda can | | |
| 9 | search | water bottle | | |
| 10 | memory | water bottle | | |
| 11 | search | scissors (repeat) | | |
| 12 | memory | scissors (repeat) | | |

### Layout L5 (starts with memory)
| # | condition | goal | tape cm | done |
|---|-----------|------|---------|------|
| 1 | memory | humidifier | | |
| 2 | search | humidifier | | |
| 3 | memory | teddy bear | | |
| 4 | search | teddy bear | | |
| 5 | memory | soda can | | |
| 6 | search | soda can | | |
| 7 | memory | water bottle | | |
| 8 | search | water bottle | | |
| 9 | memory | scissors | | |
| 10 | search | scissors | | |
| 11 | memory | humidifier (repeat) | | |
| 12 | search | humidifier (repeat) | | |

---

## What Claude records automatically

Every trial writes a data file (trial id like `t20260812-...`). Claude
fills in: layout id, trial number, your tape number, photo name, and any
notes. You never touch files — just say the cm number out loud.

## Video (only 3 times all campaign)

Film with the second phone, whole trial start to stop:
- one clean memory arrival
- one clean search-sweep-find arrival
- one more of whichever looks best

That's it. When L5 row 12 is done, the campaign is over and the rest is
writing.
