# E1 Trial Protocol — Memory vs. Memoryless Object-Goal Navigation

Operator protocol for the E1 campaign (v2 — adversarially reviewed
2026-08-06; 20 findings fixed before any campaign data). One page per
section; print or keep open on a second screen during trial evenings.
The apparatus (server, conditions, logging, statistics) is implemented in
this directory; this document is everything the *human* must do for the
data to be valid.

**Conditions.** (a) `rtsm` — memory: plan from the scanned map, drive.
(b) `baseline` — memoryless: observe-then-confirm search rounds (only
observations made during the current standpoint's round are usable —
see the 2026-08-28 amendment), then drive.

> **AMENDMENT 2026-08-17 (pre-campaign, no trial data): steered
> relocation.** The baseline's search sweep records the forward depth
> clearance at each dwell heading; after a fruitless 360° sweep it
> rotates the SHORTEST way to the most-open recorded heading before its
> relocate walk (live walk guard still applies; blocked/stale falls
> back to rotate-until-open). Rationale: with no obstacle sensors the
> searcher previously walked whatever direction the sweep ended on —
> into narrow gaps and corners where the camera captures little. Each
> steer is logged (`relocate_steered`: rotation steps, best clearance).
> This STRENGTHENS the memoryless comparator; conservative w.r.t. the
> memory-advantage claim.
> **Stride (same amendment): the relocate walk covers HALF the measured
> open depth** toward the chosen heading (capped 1.0 m, floored 0.12 m),
> walked in 0.3 m chunks with a live clearance re-check between chunks —
> the next sweep happens mid-open-area with visibility all around,
> replacing the legacy fixed 12 cm hop from the blind-walk era.
> **Leash (same amendment):** depth sees PAST the venue tape (the
> boundary is not a wall), so the searcher stays within 2.0 m of its
> START pose: outbound strides are trimmed at the leash, and when every
> open heading leads outside, it walks back toward the start instead
> (mode logged per relocation: open / return / open_unleashed). Sized
> for a car starting centered in the ~4.9 × 3 m venue. Same perception, same pose stream, same
controller, same target-selection rule; the ONLY masked capability is
persistence. **Budgets are hard TOTAL clocks from command receipt** —
planning/selection time counts in (a) exactly as search time counts in
(b): **900 s (a) / 900 s (b)**.

> **AMENDMENT 2026-08-28 (pre-campaign, no trial data): observe-then-
> confirm rounds, batched selection, search cap.** Live shakedown
> (t20260828-160929-001) exposed two failures of per-dwell confirmation
> in a cluttered venue: (1) every visible object triggered its own 4-12 s
> image-verified LLM rejection — 12 calls in 150 s with the car
> effectively stalled mid-sweep; (2) the measured single-standpoint score
> band is FLAT (top-15 for "tissue box": 0.028-0.045 with the true
> target mid-pack at rank ~6), so no relevance floor can pre-filter junk
> without filtering the target (a 0.05 floor briefly deployed that
> morning would have removed even rank 1 — caught by the operator, never
> used in a trial). The baseline search is therefore restructured into
> ROUNDS: **3 full in-place 360° sweeps** (multi-view observation, no
> queries), then ONE deep query whose candidates are the observations
> made DURING the round (round-scoped freshness = the current
> standpoint's observation episode; the old 4 s gate survives as the
> upsert-lag margin), ranked **top-10**, judged in ONE batched
> image-verified selection call (same shared rule; every candidate
> carries its crop). No-match masks the judged ids and the searcher
> relocates ~1 m toward the most open in-leash heading (stride = measured
> clearance minus the 0.60 m wall-guard buffer, capped 1.0 m) before the
> next round; a round with no candidates relocates without any LLM call.
> **Search cap:** the acquisition phase of EITHER condition is capped at
> **480 s** of the 900 s budget; exhausting it ends the trial as
> `not_found` — an explicit, analyzable outcome that also keeps the
> remaining budget sufficient for a completable drive at the calibrated
> ~0.04 m/s (condition (a)'s acquisition is a single query + selection
> call, so the cap only ever binds on (b)). `not_found` requires that at
> least one round query actually succeeded (or a selection verdict was
> rendered): a retrieval outage producing zero judged standpoints stays
> a plain `timeout`, with `round_query_failed` events in the trial log —
> an infrastructure fault is never coded as "target absent". Round
> queries are retried (3 attempts) so one dropped request cannot
> silently discard an observation round; the leash is anchored ONCE at
> the trial's start pose and persists across no-match re-entries, so
> cumulative relocation drift stays inside the venue; and every
> relocation requires live pose evidence (a timestamp change) before
> the car moves.

> **AMENDMENT 2026-08-28b (pre-campaign, no trial data): grounded
> detection + label-first retrieval.** Perception backend switched from
> FastSAM+YOLOE (dual) to GroundingDINO+SAM2 prompted with the venue
> vocabulary (teddy bear, water bottle, scissors, tissue box, dumbbell
> — disclosed; identical for both conditions). Motivation, measured
> live: the RC-mounted low camera angle made generic-proposal retrieval
> unstable, and single-standpoint SigLIP scores are non-discriminative
> (flat 0.03-0.08 band); meanwhile a per-label detector monitor showed
> GDINO firing reliably on the roster (e.g. 82 tissue-box detections,
> peak conf 0.71) while the object sat proto/unindexed — invisible to
> embedding search for minutes (confirmation gates + upsert lag).
> Retrieval policy is therefore the UNION of label search and embedding
> search for BOTH conditions, label hits ranked first, deduped
> (`/search/label`: normalized substring, query-inside-label, over
> working memory INCLUDING protos; GDINO span-merged compound phrases
> are normalized to their single best vocabulary entry at the detector).
> A fall-back-on-miss variant was reviewed out the same day: one stale
> or masked label hit could suppress the embedding candidates entirely.
> A label-side error alone (older server) degrades to pure embedding
> search; the served sources are logged per plan/round. The freshness/
> round window, rejected-id masking, and the single batched
> image-verified selection call are unchanged — the detector label
> GENERATES candidates; the image check remains the arbiter (GDINO
> false positives observed live: a wall light switch offered as a
> roster candidate and rejected by image). Symmetry preserved: same
> detector, same vocabulary, same retrieval policy, same selection rule
> in both arms; the ONLY masked capability remains persistence. The memoryless definition is
> unchanged in spirit and stricter in mechanism: candidates never
> persist across standpoints or trials; only what the round's sweeps
> observed from the CURRENT position is actionable. This STRENGTHENS the
> comparator (systematic coverage + multi-view evidence per standpoint)
> and is conservative w.r.t. the memory-advantage claim.

> **AMENDMENT 2026-08-10 (pre-campaign, no trial data collected under the
> old budgets):** originally 60 s (a) / 180 s (b), set before Phase G
> measured the real rig. Calibration showed 0.08 m/s at full command
> (~0.04 m/s at the 0.5 nav cap) — a 2 m approach alone needs ~50 s, so
> 60 s censored trials mid-drive (observed live: t20260810-232334-001).
> Now SYMMETRIC 900 s for both conditions, which also removes the
> asymmetric-budget critique; the MW comparison horizon becomes 900 s.
> The timeout only binds on failures — arrivals end trials early.

**Ground truth & success (pre-registered, fixed before any data):**
tape-measured distance, drive center → object floor cross. HEADLINE
completion = monitor verdict `arrived` AND tape ≤ 50 cm. Rationale for
50: the 40 cm commanded stop radius + ≤ 10 cm allowance for the object's
3D-centroid-vs-floor-cross projection and tape-reading granularity.
A safety-stop or timeout that happens to halt near the object is NOT a
completion (reported separately as the any-verdict secondary).
`aggregate.py` always prints TCR at 40/50/60 cm as a sensitivity sweep.

**Coordinate-age honesty note:** the phone streams continuously, so the
map keeps updating during trials — a memory-condition plan may run on a
coordinate refreshed by a recent approach rather than the session scan.
This is logged per trial (`planner.target_age_at_plan_s`) and reported in
the paper; the claim is persistence *across tasks*, not staleness of a
specific scan.

---

## 1. Venue setup (once, before the first trial evening)

- One room, fixed furniture for the whole campaign. No moving chairs
  between trials; if anything large must move, re-scan and note it.
- Constant lighting: overhead lights ON, blinds fixed. Do not mix
  lighting states within a layout's trials.
- Floor: clear drivable area ≥ 2.5 × 2.5 m. Remove cables/rug edges.
- **Keep-out perimeter**: tape a line around any fragile/breakable zone
  and around the drivable area's edge. This line defines the e-stop rule
  (§3 step 4) — decide it now, not mid-trial.
- **Layouts L1–L5**: five goal-object × placement combinations. For each,
  mark with painter's tape:
  - the object's floor position (small tape cross under/beneath it),
  - the start pose: a tape outline of the car's footprint + an arrow for
    its nose direction.
- Layout requirements (all five):
  - straight-line start→object distance 1.5–3.0 m,
  - target NOT in the camera's view at the start pose (nose ≥ 90° away),
  - at least two layouts with the target BEHIND the start pose, at least
    one with the target well off-axis (mild detour),
  - ≥ 0.8 m clearance around the object's approach side.
- Objects: distinct categories, one per layout (e.g. plush teddy bear,
  backpack, shoe, watering can, ball). No same-category duplicates in
  view. Record the exact goal phrase per layout — it is part of the
  layout definition and never changes.
- **Two-sided layout acceptance** (both must pass BEFORE committing):
  1. *Close verify*: scan the object at 1–1.5 m, two angles; semantic-
     query its goal phrase; require a confirmed hit with a sane
     coordinate.
  2. *Start-geometry check*: place the car (phone mounted) on the start
     tape and run ONE baseline acquisition capped at a single sweep
     (send the goal with `condition: baseline`, cancel after the first
     sweep if unacquired). Require at least one freshness-gated hit
     during that sweep. This prevents admitting objects the baseline is
     structurally blind to at range — which would inflate the memory
     advantage by object selection. Record pass/fail + the sweep step of
     first hit in Appendix A, and list every REJECTED candidate object
     with the reason (close-verify fail vs start-geometry fail).
- Fill in the layout table (Appendix A) once and commit it with the data.

## 2. Pre-session checklist (every trial evening)

1. Calabi Lens deployed within the last 7 days — else redeploy the night
   before. **iPhone Auto-Lock set to NEVER** (Settings → Display &
   Brightness) for the whole session — an auto-lock kills the ARKit
   session, shifts the world origin, and forces a full re-scan (cost us
   one live: 2026-08-07).
2. Car battery fully charged; spare charged if available. Read voltage
   with `.venv/Scripts/python.exe preflight_check.py` (read-only, prints
   battery mV, sends no motion commands). **`/status` does NOT report
   voltage.** Swap/stop rule: below 7000 mV, finish the current PAIR of
   trials, then swap or end the session. (The server only refuses below
   6800 mV and only at preflight — the 7000 rule is operator-enforced.)
3. PS4 controller paired and AWAKE (solid light) BEFORE starting the
   agent server; verify `/status` shows `gamepad_available: true`.
   **LIVE-FIRE CHECK (mandatory, before the first trial): press X with
   the car idle → `/status` must show ESTOPPED with source
   gamepad-button0 AND `estop.binding_verified: true` → `/reset_estop` →
   READY.** `gamepad_available: true` alone is NOT proof the button
   works: a controller that slept and reconnected mid-run can bind as
   detected-but-deaf (a real mid-drive kill press was silently lost to
   this, 2026-08-07). `binding_verified` flips true only when a real
   press is seen on the CURRENT binding and resets on every reconnect —
   it is the honest signal; `last_button_press_mono` gives the age of the
   newest observed press. **If the controller ever sleeps or reconnects
   during a session: wait for `gamepad_available: true`, repeat the
   live-fire check, and confirm `binding_verified: true` before the next
   trial.** estop.py now rebinds via SDL hotplug events instead of the
   subsystem-restart rescan that produced the deaf binding — but until
   that path is live-fire validated on this rig (`hw_estop_check.py`,
   full sleep→wake→press sequence), restart the agent server as the
   conservative fallback whenever `binding_verified` will not flip true.
4. Boot order: **FRESH RTSM process** (`python -m rtsm`, wait for models)
   → phone streams (`ws://<desktop-ip>:8765/stream`) → **scan** → agent
   server (with ANTHROPIC_API_KEY injected) → `READY`.
   **Never build a campaign map on top of `POST /reset`** — reset clears
   working memory but leaves ghost entries in the vector index that crowd
   retrieval (found live 2026-08-07). Fresh process = fresh index.
4b. **Pose-feed health gate (added 2026-08-10, after the RTSM wedge):**
   with the phone streaming and the car idle, poll `/stats` and confirm
   **≥ 3 fresh poses/s over 10 s** before the first trial and again
   between layouts. Session 4 saw RTSM degrade progressively (pose rate
   fell from ~4.5/s to ~1/s, then the process wedged at ~3 GB RSS and
   stopped answering HTTP entirely — the phone "disconnecting" was the
   symptom, and the car's motion turned stepwise long before that).
   A slow feed = jerky control and eventual `stale_stop` aborts; do not
   burn trials on it. If the rate is low: check iPhone Low Power Mode
   (throttles ARKit), phone thermals, then restart the RTSM process
   (fresh scan — session rules apply). Launch RTSM with stdout/stderr
   redirected to a log file so a wedge/crash leaves a trace. Also check
   iOS Low Power Mode is OFF at session start — charging from a dead
   battery tends to switch it on.
5. **Scan procedure (reproducibility):** walk the SAME loop each session —
   perimeter of the drivable area, one slow lap, then a direct look at
   each layout object from ~1–1.5 m, two angles each. Target ~2–4 min.
   Then semantic-query every layout's goal phrase and confirm a confirmed
   hit with a stable coordinate. No trials until all five verify.
6. Calibration: `config.yaml` must carry the current rig's calibration
   (`is_calibrated: true`, correct `rig_id`). If the phone was remounted,
   re-run `calibrate.py` first. (`aggregate.py` EXCLUDES uncalibrated
   trials — they would be wasted evenings.)
7. Assign the session id (S1, S2, …) and start the session log row
   (Appendix B): date, session id, boot mV, scan duration, verification
   results, calibration rig_id.

## 3. Per-trial procedure

0. **Schedule.** Trials strictly alternate (a, b, a, b, …) within a
   layout. The STARTING condition alternates between layouts (L1 starts
   with a, L2 with b, …). Never re-order to "make up" a failed trial —
   a failure IS a data point; continue the alternation.
1. Place the car exactly on the start-pose tape. **Start photo** with a
   SECOND camera: `<task_id>_start.jpg` framing car-on-tape (this is the
   evidence that placement was correct — mis-placement invalidation is
   only allowed if this photo shows the car off the tape).
2. **Glance at the Lens screen — streaming and tracking normal — AND
   confirm `/status` is `READY`** before every command. (READY is cached
   from the last preflight; it does not re-verify the phone. The glance
   is the phone check.) If ESTOPPED: see the re-arm bullet in step 5.
   Thumb NEXT TO the X button — not resting on it.
3. Send the command with the layout's exact goal phrase and the scheduled
   condition — **ALWAYS pass `condition` explicitly** (the server
   defaults omissions to `rtsm` with no error). Read the response:
   confirm the echoed condition matches the schedule BEFORE writing the
   task_id on the sheet. **Never POST /command while a trial is RUNNING**
   — a second command preempts the live trial and starts a new one from
   wherever the car stopped. If a duplicate fires: BOTH resulting trials
   are INVALID; re-place on the start tape and run the scheduled trial.
4. Observe. **E-stop rule (objective, same for both conditions): press X
   ONLY when the car crosses the taped keep-out line, snags a cable,
   leaves the drivable area, or contact is already unavoidable — never
   because it merely looks close.** ANY X press during a trial is a
   valid FAILED trial, regardless of intent — including accidental taps
   (note the accident in `notes`; the trial still counts). There is no
   "mistake e-stop" invalidation.
5. **A trial is over when `GET /status` shows `state` no longer RUNNING;
   the verdict is `last_result.result`.** Nothing is printed anywhere —
   always read `/status`; never judge by eye (arrived, stale_stop, drift
   and timeout all look like "car stopped"). Then, without moving the car:
   - Tape-measure: drive-center chassis sticker → object floor cross,
     nearest cm. **Evidentiary photo**: the laid tape with the reading
     legible, `<task_id>_tape.jpg` (second camera). Stop photo
     `<task_id>.jpg` (car + object in frame).
   - **Second camera only — NEVER unmount or background the streaming
     iPhone.** Backgrounding Lens restarts its ARKit session and silently
     shifts the world frame, poisoning every remaining rtsm trial of the
     evening. Photos are backup evidence; under time pressure the JSONL
     edit (step 7) is the non-skippable step, the photo is not.
   - **Verdict handling:**
     - `arrived / timeout / stale_stop / frame_reset / drift / estopped`
       → valid trial (success or failure), record and continue.
     - `estopped` additionally: after measuring, `POST /reset_estop`
       (a 409 means the mission hasn't finalized — wait a second and
       retry), confirm `/status` READY. **Never restart the agent server
       or RTSM to clear an e-stop** — that forces §4's full re-scan.
     - `not_found / plan_error / search_error / nav_error / worker_error`
       → valid FAILURE (car may never have moved): tape only if it moved,
       copy `last_result.detail` into `notes`, continue the schedule.
     - `preempted / cancelled` → INVALID (operator-induced): see
       Trial validity below.
6. Record on the session sheet: task_id, session id, layout, condition,
   verdict, tape cm, battery mV (if checked this cycle), notes.
7. **Immediately** fill the JSONL: open
   `paper/demo2_data/<task_id>.jsonl`, first line (`trial_start`), set
   `"tape_cm"`, `"layout_id"`, `"start_pose_id"`, `"session_id"`,
   `"video_file"` (if any), `"notes"` (if any). Save. The `trial_start`
   line is the ONLY operator-editable record; never touch tick/end lines.
8. Reset for the next trial: carry the car back **slowly, camera
   unobstructed** (the Lens session keeps tracking during the carry).
   Glance that the object didn't move; if bumped: re-place on its tape
   cross and RE-SCAN it (5 s direct look) before the next rtsm trial.
9. Every 6 trials (falls on a pair boundary, so both conditions see the
   same cadence): check battery via `preflight_check.py`, apply the
   swap/stop rule, AND **rest the phone ~5 min** (car parked, streaming
   may continue idle or stop — if you stop Lens, that is a Lens restart:
   see §4). Note any iOS thermal warning on the sheet.

### Trial validity

- Valid: every trial run per schedule, including all failures, e-stops
  and timeouts. Failures enter the statistics at the censoring cap.
- **INVALID (excluded, counted, never silently):** a bystander or
  operator physically contacted the car or object mid-trial; the wrong
  goal phrase or condition was sent; a duplicate command preempted a live
  trial (both trials invalid); the pose stream was already dead or Lens
  had restarted BEFORE the command was sent (symptom: stale_stop or
  frame_reset within seconds, no meaningful motion); the start photo
  shows the car off the tape; any `preempted`/`cancelled` result.
- **Mechanism (this is what makes exclusion real):** set
  `notes: "INVALID — <reason>"` in the trial_start line AND move the
  JSONL into `paper/demo2_data/invalid/` (keep it forever; never delete).
  `aggregate.py` also auto-excludes INVALID-notes and
  preempted/cancelled/shutdown results and PRINTS every exclusion count —
  the sheet and the printout must reconcile (§5).
- An invalid trial, unlike a failure, is not a data point: run the SAME
  scheduled slot again next.

## 4. Between layouts / stream discipline

- Move to the next layout's tapes. If the object or placement changed
  since the scan: re-scan that area (two angles), re-verify with a
  semantic query, THEN start the layout's trials.
- Do NOT reset RTSM between layouts within a session (shared map is by
  design; the baseline's gate masks it).
- **RTSM restart OR Lens/phone restart (new ARKit session — including
  backgrounding the app) = same treatment:** full re-scan + re-verify all
  remaining layouts before continuing, note it in the session log. This
  is why photos use a second camera and carries keep Lens foregrounded.

## 5. After each session

1. Run `aggregate.py --dir paper/demo2_data` — reconcile: analysis n =
   sheet's VALID rows; every printed EXCLUDED count matches the sheet's
   INVALID rows; warnings only for intentionally deferred tapes.
2. Back up: commit the JSONLs (including `invalid/`), copy photos/videos.
3. Note battery/lighting/thermal/anything-unusual in the session log.

## 6. Campaign size, sessions, and stopping

- Target: **30 valid trials per condition** (6 per layout × 5 layouts).
- **A session may END only at a layout boundary** (all 12 of a layout's
  trials complete). Pre-plan evenings as whole layouts (2+2+1 or 3+2).
  If a session must abort mid-layout (crash, fatigue): mark that layout's
  already-run trials INVALID, re-run the full layout next session (fresh
  scan), alternation restarts at that layout's original starting
  condition. Rationale: a straddled layout hands fresh
  battery/scan/thermal state to whichever condition resumes — an
  uncontrolled, one-sided nuisance.
- Statistics (pre-registered here): pooled one-sided Mann-Whitney on TTA
  with ALL failures of BOTH arms at the common 900 s horizon (was 180 s —
  see the 2026-08-10 budget amendment in the header); arrivals-
  only sensitivity; **clustering acknowledged** — trials share layout
  geometry, so the pooled p is reported alongside the per-layout win
  table and exact one-sided sign test across the 5 layouts (all printed
  by `aggregate.py`). Power, stated prospectively (no pilot exists): at
  n=30/arm, one-sided α=0.05, ~80 % power requires
  P(memory faster) ≳ 0.68 (d ≳ 0.65) — the design detects only large
  effects, which matches the ≥ 3× median-speedup go/no-go expectation;
  smaller effects are a null result and will be reported as such.
- Good-signal gate: ≥ 70 % headline tape-TCR (memory), ≥ 3× median TTA
  speedup, video reads clearly.

---

## Appendix A — Layout table (fill once; commit with the data)

| layout | object | goal phrase | object pos (desc + tape) | start pose (desc) | dist (m) | target-behind? | start-geom check (pass / sweep-step) | notes |
|---|---|---|---|---|---|---|---|---|
| L1 | plush teddy bear | "go to the teddy bear" | | | | | | |
| L2 | | | | | | | | |
| L3 | | | | | | | | |
| L4 | | | | | | | | |
| L5 | | | | | | | | |

Rejected candidate objects (object — reason: close-verify fail / start-geometry fail):
-

## Appendix B — Session sheet (one row per trial)

| # | task_id | session | layout | cond | verdict | tape cm | mV | notes |
|---|---|---|---|---|---|---|---|---|
