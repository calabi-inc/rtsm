# P1 task 1 gates — ingest clock (2026-09-09, headless `dual`, `recordings/session1`, trace on)

Runs via `scripts/benchmark_datasheet.py dual --profile <label>.profile.yaml [--set ingest.clock=wall] [--replay-speed 5]`
on branch `feature/sensor-clock` (tree = main `90c01d5` + this change). Empty FAISS store per run. `gates.out` is the
driver's full output; `<label>.json` the harness datasheet; `<label>.events.jsonl` the frame-flow trace (schema 2).

| run | ingest.clock | speed | objects/confirmed | frames | full multiset sha | enqueued / throttled | dequeue outcomes |
|---|---|---|---|---|---|---|---|
| A  | wall   | 1x | 121/66 | 53 | `92e0a8f1206d77da` = the headless anchor | 88 / 152 | processed 53 (keyframe 9, hard_max_s 31, parallax 13), gate_rejected 35 (skip 31, near_recent_keyframe 4) |
| B1 | sensor | 1x | 124/65 | 53 | `ad6f71a5b89c8506` | 86 / 154 | processed 53 (keyframe 9, hard_max_s 30, parallax 13, ttl+novelty 1), gate_rejected 33 (skip 29, near_recent_keyframe 4) |
| B2 | sensor | 1x | 124/65 | 53 | `ad6f71a5b89c8506` | 86 / 154 | identical to B1 |
| B3 | sensor | **5x** | 124/65 | 53 | `ad6f71a5b89c8506` | 86 / 154 | identical to B1 |

**G1-A (refactor parity) — PASS, with a caveat the review surfaced.** With `ingest.clock=wall` every decision
equals main: same 88 admitted frames, same dequeue sequence and reasons, same multiset as
`../2026-09-post-reconcile-headless/` (121/66 @53, `92e0a8f1`). The mechanism is NOT "stamp-on-admit ==
stamp-on-enqueue": wall mode now stamps the throttle ~3 ms earlier than main did (at the admit decision instead of
after decode + enqueue; decode of a session1 frame measures 2–5 ms). Two admit gaps in this trace sit at exactly
500.0 ms, so a 3 ms shift could have flipped them. It did not, because Python 3.12 on Windows implements
`time.monotonic()` with a 15.625 ms tick (GetTickCount64): old and new stamps quantise to the same value. On a
ns-resolution monotonic clock (Linux, Jetson, or Python 3.13, which moved Windows to QueryPerformanceCounter)
wall-mode replays were never bit-reproducible, for main and for this branch alike. Consequence: wall parity is a
this-box result; **the sensor anchor below is the portable reference from now on.**

**G1-B (semantic change) — PASS.** With the sensor clock, two 1x runs and one 5x run give an identical enqueued
sequence (86), an identical throttled sequence (154), an identical dequeue sequence of (frame_seq, t_sensor_ns,
outcome, reason) (86) and an identical full object multiset (`ad6f71a5b89c8506`, 124/65). Replay speed no longer
changes what RTSM remembers.

**Sensor-vs-wall delta (the anchor P2–P4 cite for `dual`):** 124/65 (sensor) vs 121/66 (wall). The sensor-time
throttle admits two fewer non-keyframes (first divergence at the third admitted frame: seq 5 under wall vs seq 6
under sensor — wall admitted seq 5 because ~0.5 s of *processing* time had elapsed since the previous admit, sensor
saw only ~0.4 s of *sensor* time); processed count is 53 in both, gate reasons shift by one frame (`ttl+novelty`
appears once). `upserts_total` (233 wall, 206 sensor at 1x, 177 at 5x) is the vector-flush cadence, which stays on
wall time by design and is not an evaluation metric; end-of-replay FAISS content is force-flushed either way.

The dual sensor anchor **124/65 @53, `ad6f71a5b89c8506`** is the reference for the remaining P1 tasks (G1-A of
task 2/3 compares against it); `../2026-09-post-reconcile-headless/` remains the wall reference.

Scope notes from the review: (1) the determinism guarantee is **headless**: with the viz server on, recorded
`pose_corrections` text messages are applied to memory from the replay thread on wall pacing, so a viz-on run of a
recording whose corrections move positions can differ (session1's do not). (2) `ingest.non_kf_grace_s` (0.03 s) was
effectively dead under wall (the next dequeue after a keyframe is always a processing step later) and is live under
sensor time, sitting 3 ms below the 30 Hz frame period; it produced no rejections on session1 (the 200 ms
`dup_window_ns` covers the same frames) and is deterministic per recording, but task 6 should define it relative to
the frame period or drop it. (3) A `SensorClock` re-anchors on `POST /reset` and re-bases on a backwards jump larger
than 5 s inside one epoch (a re-replayed recording), so a long-running process cannot freeze its timing gates.

## Task 2 reproduction (admit before decode, 2026-09-09) — `task2-admit-before-decode/`

Same harness, branch `feature/admit-before-decode` (main `50e4a9e` + the reorder). Sensor run **S**: 124/65 @53,
full sha `ad6f71a5b89c8506` = this anchor; dequeue sequence identical to B1 (86), receiver sequence identical (240);
`depth_valid_frac` present on all 240 receiver lines (throttled frames included) and uniform: 1.000 on enqueued and
throttled lines alike (pre-confidence-filter statistic; ARKit depth has no zero pixels on the wire). Record = the re-run
after the review fixes (packet-carried statistic, ZMQ stamp-before-admission, window 90) — same numbers as before them. Wall run **W** (info only):
121/66 @53 `92e0a8f1206d77da` = A, dequeue identical (88). Moving the queue admission in front of the RGB decode and
the viz broadcast behind it changes nothing when nothing is dropped, as designed.

## Task 3 reproduction (ingest lanes, 2026-09-09) — `task3-ingest-lanes/`

Same harness, branch `feature/ingest-lanes` (main `c75a8c5` + the lanes). **L** = packaged default (`ingest.policy: auto`
→ lossless under replay): 124/65 @53 `ad6f71a5b89c8506` = this anchor; dequeue (86) and receiver (240) sequences identical
to B1; all receiver lines `source: replay`; per-line `(decision, reason, frame_seq, depth_valid_frac)` identical to
`task2-admit-before-decode/S.events.jsonl`. FIFO peak depth 2, `blocked_puts` 0 (the record is the re-run after the code-review fixes; the first run peaked at 3). **G** = `--set ingest.policy=legacy`:
identical. **G1-A PASS.** **T1–T3** = `--set ingest.policy=latest`, info only (not reproducible by construction, never
compared to the anchor): each superseded exactly one waiting non-keyframe (max depth 2, no keyframe dropped, no age drop,
enqueued 86 − 1 = 85 dequeued) and still produced 124/65 `ad6f71a5`; the lanes barely engage when dual keeps up with
session1 at 1×. `gate.out` is the script's verdict; the script itself is `p1t3_gate.sh`. The raw artifacts of these five
runs (`*.json`, `*.events.jsonl`, ~25k lines) are deliberately NOT committed: nothing compares against them (the references
stay `B1.*` and `task2-admit-before-decode/S.*`), and the script regenerates them in ~7 min. From this record on, raw
artifacts are gitignored under `eval/baselines/`; only a new reference anchor is force-added.

## Task 4 reproduction (pose mailbox, 2026-09-10) — `task4-pose-mailbox/`

Same harness, branch `feature/pose-mailbox` (main `1f22934` + the mailbox). **M** = packaged default (lossless, sensor
clock): 124/65 @53 `ad6f71a5b89c8506` = this anchor; dequeue (86) and receiver (240) sequences identical to B1; per-line
`(decision, reason, frame_seq, depth_valid_frac)` identical to `task2-admit-before-decode/S.events.jsonl`. Final
`/stats.robot_pose` `xyz / quaternion_xyzw / timestamp / frame_epoch` equal the task-2 record (session1's last received
frame, seq 405, epoch 0); `sensor_ts_ns` = that frame's stamp; `pose_clock: sender`; **one writer:** `writes_accepted`
240 = every tracking-normal receiver line, `sensor_ts_regressions` 0, `rejected_writes` 0. **PASS.** Script + verdict only
(raw artifacts gitignored).

## Task 5 reproduction (headless metrics, 2026-09-11) — `task5-headless-metrics/`

Same harness, branch `feature/headless-metrics` (main `ae60117` + the analytics ticker), **no visualization client
anywhere**. **M** = packaged default (lossless, sensor clock): 124/65 @53 `ad6f71a5b89c8506` = this anchor; dequeue (86)
and receiver (240) sequences identical to B1; per-line identical to the task-2 record; `robot_pose` predicates of task 4
unchanged (240 / 0 / 0). **Headless metrics (new):** the raw JSON now carries `rollup` and `healthz`; the rollup owner
ticked 90 times with `late_ticks` 0, `stale_rollups` 0, `stalled` false and `last_tick_age_s` 0.5 at read time;
`latency_hourly` has 90 buckets = ticks = `segmentation_hourly` (35 with frames, none flagged `stale_interval`, `elapsed_s`
max 1.015) whose per-bucket sums equal the lifetime counters exactly — frames 53 = 53,
received 240 = 240, gate_rejections 33 = 33, throttle_skips 154 = 154, queue_drops + superseded + age_drops 0; the
aggregate reads `input_hz` 5.0 / `effective_ratio` 0.232 (B1 and the task-2 record, made before the ticker, read 0.0 /
1163.6); the last bucket's `wm_total` / `wm_confirmed` 124 / 65 equal `/stats`; `segmentation_hourly` sums to 53;
`/healthz.ingest` = `{policy: lossless, depth: {fifo: 0}, lane_full: false, closed: false, closed_puts: 0, admitted 9 + 77
= 86 dequeued, no lane drops, max_depth_seen 2, blocked_s 0.0}` with no `frame_flow` key (replay: watchdog off). **PASS.**
Gate run 1 on the same tree failed one predicate (throttle 148 vs 154: the ticker's start had re-anchored the drop
cursors, so six skips recorded before it started were in no bucket); the fix (time cursor only) passed in run 2; the
recorded run 3 is after the code review (`stalled` / tick-age / bucket-count predicates added, `closed` informational).
Script + verdict only (raw artifacts gitignored).
