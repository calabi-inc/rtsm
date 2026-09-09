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
