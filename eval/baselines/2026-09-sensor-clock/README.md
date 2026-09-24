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

## Task 6 reproduction (config key moves, 2026-09-11) — `task6-config-keys/`

Same harness, branch `feature/config-ingest-keys` (main `1fcd4d9` + the moves), three headless `dual` replays.
**M** (packaged defaults, 1×): 124/65 @53 `ad6f71a5b89c8506` = this anchor; dequeue (86) and receiver (240) sequences
identical to B1 (reasons included — `non_kf_grace_s: 0.0` and the moves change nothing); every task-5 predicate holds
(pose 240/0/0, rollup 90 ticks = 90 buckets per buffer / 0 late / 0 stale / not stalled, Σ frames 53 / received 240 /
gate_rejections 33 / throttle 154 / drops 0 / seg 53, last bucket WM 124/65, `/healthz.ingest` idle + drained with
9 + 77 admitted); the runner's logged
config `SHA-256` equals the CPU-side fingerprint of the same profiles on the harness-patched base (validates the
fingerprint replication). **L** (a profile setting the OLD paths `io.websocket.keyframe_every_n: 30` /
`io.websocket.nonkf_min_interval_s: 0.5`, 1×): identical to M; the run log carries both `rtsm.cfg` deprecation lines;
its fingerprint equals the CPU-side fingerprint with and without the legacy profile (old path → same resolved dict).
**T — the falsifying run** (`io.websocket.nonkf_min_interval_s: 1.0` through the OLD path, 5×): receiver enqueued 48
(9 keyframes + 39 non-keyframes) / throttled 192, dequeue keyframes 9 — exactly the counts derived offline from B1's
receiver lines with the attempt-based throttle rule (which reproduces 86/154 at 0.5 s line for line); deprecation
line present; fingerprint equal via the old and the new path; multiset 97/45 `670dca82b330db26` (fewer admitted
frames, as expected). A runner site still reading `io.websocket` (or dropping the kwargs) would have reproduced
86/154 here. **PASS** (recorded run = after the code review). `gate.out` is the script's verbatim output: the six
`DeprecationWarning` lines are the gate's own CPU-side `load_config` calls echoing the shim on stderr. Script + verdict
only (raw artifacts gitignored).

## G1-C — the wedge proof (2026-09-18) — `g1c-wedge/` — **FAIL on the production configuration, PASS with the viz TSDF path off; harness valid**

Branch `feature/g1c-wedge-proof` (tree = main `26c2508` + this directory; **no product code changes**). The plan's G1-C
(execution plan v2, line 145) runs the WHOLE `python -m rtsm` process — packaged defaults: grounded_sam2, visualization
ON (TSDF fusion on), analytics ON, watchdog ON, diagnostics OFF — with a **separate sender process** streaming
`recordings/session1/messages.bin` raw bytes through the live websocket receiver (hello/hello_ack per loop, fresh
`session_id` per loop → frame-epoch bump, **uncompressed** like the phone), a **separate poller** standing in for the
agent (`/stats` 10 Hz / 0.6 s, `/search/semantic` 0.5 Hz / 3.0 s, `/healthz` + RSS 1 Hz, `Connection: close` per call
like `rtsm_client.py`) and a **headless dashboard client** on `:8002/ws` (browser-default compression). Scripts:
`g1c_gate.py` (driver, predicates, `--reeval` from saved artifacts), `g1c_sender.py`, `g1c_poller.py`,
`g1c_vizclient.py`, `nobrowser/sitecustomize.py` (neutralises the runner's browser auto-open; P10 asserts exactly one
dashboard client ever connected). `gate.out` = the GPU run's verbatim output (original P5/P6 constants);
`reeval.out` = the same seven runs re-derived offline after two harness calibrations (below); `summary.json` (force-added)
= the re-evaluated numbers. Raw `*.jsonl`, `rtsm_*.log`, `finals.json` are local only (the script regenerates them).

**Design review before code** (three lenses — falsifiability, repo feasibility, realism vs E1 — two independent refuters per
high/critical finding; 32 findings, 13 verified, 9 survived): the sender's default permessage-deflate negotiation
(it turned the transport ceiling into zlib time — see the table), the non-existent `:8083` viz port (integrated mode),
the TSDF frame ring inside the RSS budget, uvicorn's 32-message read-ahead queue hiding a steady pose lag (→ the
end-to-end lag predicate P11), an absolute pose-rate floor that a proportional collapse would pass (→ poll-normalised P2),
"F only proves the harness can read a queue counter" (→ calibration run C), inline TSDF work on the receive coroutine (→
attribution runs R1V/R1N), P8 aliasing (→ watchdog transition lines), P10 gaps (six log patterns). Two findings were
refuted (reconnect-per-loop racing the parked tail; F never filling the legacy queue) and one deferred (a laggy-dashboard
phase).

### Transport ceiling (CPU-only, before any GPU run; 4.30 MB NV12 frames of session1, localhost, sender "as fast as accepted")

| path | permessage-deflate negotiated | uncompressed |
|---|---|---|
| raw asyncio TCP loopback (no websocket layer) | — | ~490 msg/s (2 GB/s) |
| `websockets` 16.0 client → `websockets.serve` (new impl), no parse | 7.7 Hz | — |
| **production `WebSocketReceiver`** (uvicorn 0.43 `ws=auto` → `websockets.legacy`) + header/pose/depth parse, queue refuses before RGB decode | **4.67 Hz (221 ms/frame)** | **84.5 Hz (12 ms/frame)** |
| same receiver with uvicorn `ws="websockets-sansio"` | 7.67 Hz | 96.6 Hz |

Uncompressed, the receiver's transport+parse path is 15× the phone's cadence (5.4 Hz, median gap 186 ms) and is not a
bottleneck. **With permessage-deflate negotiated it drops to 4.67 Hz — the E1 memo's "4.5 Hz input".** The receiver
OFFERS the extension (uvicorn `ws_per_message_deflate=True` default; `rtsm/io/websocket.py` does not set it) and this
repo does not record whether Calabi Lens accepts it (`meta.json` keeps hello/ack only). **Open question, founder (Lens
source or one packet capture):** if Lens negotiates it, `ws_per_message_deflate=False` in the receiver's `uvicorn.Config`
is a one-line, 18× receiver-side change and a co-factor of the E1 wedge; if it does not, the transport is exonerated.
The harness streams uncompressed (the conservative reading).

### Runs and verdicts (`reeval.out`; every number below is from the re-evaluation, gate.out holds the originals)

| run | policy | speed | throttle | s | role | verdict |
|---|---|---|---|---|---|---|
| C | latest | 1× | 0.5 | 60 | calibration: rtsm suspended 3.0 s at +25 s | **CALIBRATED** — P1 (6 timeouts), P2 (window 36 < 45.9), P3 (gap 3.17 s, 7 stale), P5 (stall 2.60 s), P9 (`late_ticks` 1), P11 (lag 2.98 s) all tripped |
| F | **legacy** | 1× | 0.2 | 150 | falsifier | **WEDGE-VISIBLE** — `ingest_q` max 173 (final 137), RSS 4.2 → 6.2 GB, +575 MB/min in the last minute; pose lag p99 0.128 s (the receive-time mailbox is independent of the queue, as designed) |
| R1 | latest | 1× (phone cadence) | 0.5 | 300 | HARD — the E1 condition | **FAIL**: P1, P3 |
| RB | latest | 1× | 0.2 | 150 | HARD — pipeline overload, bounded lanes | **FAIL**: P1, P3 |
| RS | latest | 3× (saturated) | 0.5 | 150 | HARD — achieved 15.79 Hz of 16.27 pushed | **FAIL**: P1, P2 (82 < 85), P3 |
| R1V | latest | 1× | 0.5 | 150 | attribution: `visualization.tsdf.enable=false`, client attached | **PASS** (all 14) |
| R1N | latest | 1× | 0.5 | 150 | attribution: `--no-viz` | **PASS** (all 13) |

What holds in every `latest` run, i.e. **the P1 ingest work did what it was built to do**: lanes bounded (`ingest_q` ≤ 3,
`max_depth_seen` 3, `age_dropped` 0, 95–288 non-KF supersessions per run), `writes_accepted` == frames sent with 0
regressions / 0 rejects across 4–10 fresh-epoch loops, sender→observed pose lag p99 0.11–0.13 s (max 0.18–0.41 s),
transport keeping the phone's schedule to within 17 ms (`send()` stall p99 ≤ 25 ms, 0 failures, 0 extensions), no
watchdog degradation, rollup owner alive with 0 late ticks, RSS plateau (+57 … +182 MB after the 60 s footprint settle,
ring-credited), and at 3× pacing the whole process still ingests 15.8 Hz.

What fails, only with the viz TSDF path on (R1 / RB / RS vs R1V / R1N):

| symptom | TSDF on (R1) | TSDF off (R1V) | no viz (R1N) |
|---|---|---|---|
| `/stats` p99 / max / samples > 0.6 s (n≈1.4–2.8k) | 0.487 s / 0.618 s / 2 | 0.031 / 0.050 / 0 | 0.031 / 0.186 / 0 |
| max gap between pose changes / stale samples | 0.818 s / 81 of 2771 (2.9 %) | 0.400 s / 0 | 0.400 s / 0 |
| dashboard stream | **silent from +46 s to the end (300 s), socket still open** | 1 MB/s for the whole run | — |
| RSS plateau | ~4.7 GB | ~3.75 GB | ~3.8 GB |

**Mechanism (measured, not inferred):** in R1, 48 of the 51 pose gaps > 0.5 s start within 0.11 s of a keyframe send, and
the 54 slow `/stats` samples (> 0.3 s) sit at the same instants (+16, 22, 27, 33, 38, 44, 50, 56, 61 … every ~5.5 s = the
keyframe cadence) for all 300 s — also after the dashboard client was gone, so it is not the send path. Every admitted
keyframe runs `vis_server.handle_frame_packet` synchronously inside the websocket receive coroutine (rtsm/io/websocket.py
`_on_keyframe`, run.py:351): resize → `TSDFIntegrator.integrate` (33–265 ms real scene, 0.8 s synthetic; **releases the
GIL** but blocks the receive loop) → `should_extract` is true at every keyframe (`extract_interval_s` 2.0 < KF cadence) →
a new unguarded `tsdf-extract` thread → `extract_point_cloud` under the same lock — and **Open3D's `extract_point_cloud`
HOLDS THE GIL**: with a busy Python thread as the probe, other threads run at 2–4 % of their idle rate for the whole
extraction, which takes 1.07 s at 1.4 M points, 1.65 s at 2.1 M, 2.59 s at 3.2 M (synthetic room; the real R1 cloud was
~0.6 M points by +45 s because the 44 s loop re-scans one room). The whole interpreter — API thread, receive loop, pipeline
— freezes for that long every 2–5 s, and the time grows with the explored volume. In E1 (a larger room, 10+ minutes of
new volume, a browser dashboard) that is a mechanism for "progressive degradation to pose 1 Hz and HTTP dead", separate
from and additive to the legacy queue (memo `ingest-drop-policy-2026-09.md`). The dashboard's silent death: the extracted
cloud is broadcast whole every time (3 → 10 MB per message inside 45 s in R1; 21–48 MB in the synthetic run), the API
uvicorn deflates it on its event loop for a browser-compressed client, `_try_send_bytes` gives up after 5 s and
`_broadcast_bytes` discards the client without a close frame or a log line; `client_count` drops to 0 and every push stops.

**Verdict:** G1-C FAILS on the production configuration; the ingest path passes every predicate; the failing predicates
are attributable, reproducible and fixable in the visualization TSDF path. Proposed P1 task 7 (founder decision):
(1) never call TSDF work from the receive coroutine — a bounded worker (one in flight, latest-wins) like the ingest lanes;
(2) extraction must not hold the GIL for seconds: a separate process for the viz volume, or an incremental / rate- and
size-bounded extraction, or `visualization.tsdf.enable: false` as the packaged default with the E1 protocol updated
(4b already tells the operator to check the pose feed); (3) log a discarded dashboard client and send a close frame;
(4) answer the permessage-deflate question above and set `ws_per_message_deflate=False` on the receiver if Lens accepts
it. Re-run G1-C after task 7; R1V is the expected shape of a pass.

**Harness calibrations after the GPU run (both visible in gate.out vs reeval.out):** (a) P5 "delivered load" was evaluated
per integer second; a 5.4 Hz stream alternates 5/6 frames per second and the recording's own 0.35 s gaps make single
seconds of 4 frames normal, so it flagged 1–9 seconds in every run including R1V/R1N; it is now the sender's schedule slip
(max < 1.0 s; measured 15–17 ms in every paced run) plus achieved/target ≥ 0.95. (b) P6's baseline moved from +20 s to
+60 s: the process footprint (TSDF volume + CUDA host allocations) settles by ~60 s (R1 curve: 2.2 GB ready → 4.34 GB at
30 s → 4.58 at 60 s → 4.72 at 300 s), so the +20 s baseline measured warm-up, not growth; both views are printed.
Neither calibration changes a verdict's direction on the failing predicates (P1/P3), and F still fails P6 by +1.1 GB.

**Deviations from the plan text:** "3–5× recorded pace" → the phone's own cadence (1×) for the E1 condition plus a 3×
saturation run in which the achieved ingest Hz is the measurement (a paced sender is backpressured to whatever the receiver
accepts; "load" is only honest as the delivered rate); `/search/semantic` bound 3.0 s (the agent's real `RTSMClient`
timeout; the 0.6 s the plan cites is the ESP32 bridge's) with 0.6 s applied to `/stats` as written; age at dequeue asserted
through `age_dropped == 0` under `max_frame_age_s` 2.0 (diagnostics trace OFF = production) plus P11, the end-to-end lag
the plan did not have. Windows/3.12 note: `time.monotonic()` ticks at 15.6 ms on this interpreter; every harness
measurement uses `perf_counter`, cross-process alignment uses `time.time()`.

### P1 task 7 re-run (2026-09-18) — packaged default headless → **HARD GATE PASS** — `g1c-wedge/task7-rerun/` + `task7-rerun-viz/`

Founder decision 2026-09-18: the core's packaged default is **viz off, TSDF off** (`visualization.enable: false`,
`visualization.tsdf.enable: false`; `rtsm --viz` turns the dashboard on; `rtsm demo` keeps it on). Branch
`feature/p1-task7-viz-default-off` (main `8c2c0b1` + the defaults, a `--viz` flag mutually exclusive with `--no-viz`,
the API root no longer serving a dashboard page when headless, docs, tests; no ingest code touched). The harness
reads the viz/TSDF state of every run from the runner's own log lines instead of assuming it, and the dashboard states
became the non-gating runs R1Z (`--viz`) and R1T (`--viz` + TSDF on = the pre-task-7 packaged config). Two harness
fixes on the way: `--out` is resolved to an absolute path (the sender runs with the repo root as cwd), and the
calibration run no longer requires a `stale` sample (a headless process refills the mailbox before the 10 Hz poller can
catch `age_s > 0.5`; the 2.66 s gap is the symptom). Same matrix otherwise; gate.out per directory is verbatim.

| run | config | verdict | `/stats` p99 / max | max pose gap / stale | pose lag p99 / max | RSS adj. growth / plateau | lanes |
|---|---|---|---|---|---|---|---|
| C | headless, rtsm suspended 3 s | **CALIBRATED** (P1 4 timeouts, P2 41 < 46.4, P3 gap 2.66 s, P5 stall 2.45 s, P9 `late_ticks` 1, P11 lag 2.86 s) | — | — | — | — | — |
| R1 | headless, phone cadence, 300 s | **PASS** (13/13) | 0.031 / 0.16 s | 0.40 s / 0 of 2980 | 0.111 / 0.136 s | +89 MB / 3.86 GB | q ≤ 3, 0 age drops, 100 supersessions |
| RB | headless, throttle 0.2 s | **PASS** | 0.030 / 0.22 s | 0.40 s / 0 | 0.105 / 0.131 s | +57 MB / 3.85 GB | q ≤ 3, 214 supersessions |
| RS | headless, 3× pacing | **PASS** — ingests the full 16.29 of 16.27 Hz pushed (no longer saturated) | 0.032 / 0.22 s | 0.30 s / 0 | 0.080 / 0.105 s | +77 MB / 3.84 GB | q ≤ 3, 176 supersessions |
| F | headless, LEGACY queue, throttle 0.2 s | **WEDGE-VISIBLE** — `ingest_q` 67 (23 at the end), +245 MB/min in the last minute | 0.030 / 0.21 s | 0.40 s / 0 | 0.114 / 0.123 s | +394 MB / 4.26 GB | growing |
| R1Z | `--viz`, TSDF off, client attached (retry dir) | **PASS** (14/14) | 0.032 / 0.06 s | 0.40 s / 0 | 0.106 / 0.130 s | +40 MB / 3.84 GB | q ≤ 3 |
| R1T | `--viz` + TSDF on, client attached (retry dir) | **FAIL** P1 (p99 0.347 s, max 0.566 s), P3 (max gap 0.565 s) — the TSDF mechanism, unchanged | 0.347 / 0.566 s | 0.565 s / 0 | 0.15 / 0.24 s | +116 MB / 4.71 GB | q ≤ 3 |

`task7-rerun/gate.out` holds the full matrix; its R1Z/R1T rows there FAIL only P10 ("2 dashboard clients"): the
built-in browser pane left open from the founder's eyeballing session auto-reconnected to every viz-enabled process
(main.ts reconnects every 2 s), so those two were repeated with the pane closed into `task7-rerun-viz/` — the table's
R1Z/R1T rows are from the retry (its own "HARD GATE" line reads FAIL only because that invocation contains no gating
run). **G1-C is closed as PASS on the packaged configuration.** The dashboard with TSDF off passes as well; the fused
map stays opt-in with its measured residual, and the worker/child-process designs in
`plans/permanent-plan/tsdf-viz-fix-2026-09.md` are deferred until a long dashboard-on session is actually needed.
The 2026-09-18 FAIL record above is unchanged.

## P2 stage A reproduction (pose ledger, 2026-09-21) — `p2-ledgers/stage-a/` — **G2-A HARD GATE PASS**

Branch `feature/p2-pose-ledger` (main `fa30945` + the pose ledger): `PoseEvent` kind `pose` in the diagnostic event log
(`schema_version` 3, ledger schema 1, behind `diagnostics.ledgers`), written by the websocket / replay parser BEFORE the
tracking-state filter drops a frame and, for frames that pass it, right after the depth decode (before the keyframe rule,
the throttle and the admission), and once per `rtabmap.tracking_pose` on ZeroMQ; `rtsm/evaluation/ledger.py` reader with
`pose_health` and the `python -m rtsm.evaluation.ledger summarize` CLI; `diagnostics.ledgers` / `ledger_format` validated
before the GPU check in both runners; `[eval]` extra (pyarrow) for `ledger_format: parquet`. `p2a_gate.sh` = two headless
dual replays of session1 on the packaged replay defaults (lossless + sensor clock), diagnostics on, ledgers ON (A_on) /
OFF (A_off); `gate.out` is the verbatim script output.

| run | multiset | dequeue / receiver vs B1 | `pose` lines | predicates |
|---|---|---|---|---|
| A_on | 124/65 @53 `ad6f71a5b89c8506` | identical (86) / identical (240) | 240 | 1, 3–7 PASS |
| A_off | 124/65 @53 `ad6f71a5b89c8506` | identical (86) / identical (240) | none (`ledgers.enabled false`) | 1–2 PASS |

Logic predicates on A_on: (4) `depth_valid_frac` equal on all 240 pose/receiver pairs, none null (the pre-filter
statistic, taken at the same point); (5) the LAST pose line equals the mailbox (`/stats.robot_pose`: xyz and quaternion
to 0.0, stamp 683373591451416, epoch 0); (6) the pose lines' `t_sensor_ns` sequence equals the receiver lines' and every
`conf_hist` sums to 49 152 (the raw 256×192 confidence map, taken before the resize); (3) 240 pose lines == 240 receiver
lines == `writes_accepted` 240, `rx_seq` join 240/240, all `tracking_state normal`, all `mailbox_write`, 0 regressions.

**Session1 pose-health reference (`pose_health`, group `replay/0`):** 240 frames, stream span 40.52 s → `sensor_hz`
5.90; intervals 165 × 200 ms + 73 × 100 ms + 1 × 217 ms (p50 200.0 / p95 200.0 / max 216.7 ms; jitter 0.002 ms);
0 gaps (> 2 × median), 0 tracking-limited episodes, 0 discontinuities (0.5 m + 1 m/s · dt), `depth_valid_frac` 1.0,
`conf2_frac` mean 0.747 / p10 0.657, 0 pose errors, `writes_expected` 240.

**The four session1 time spans, reconciled (founder question 2026-09-21):** sensor stamps 40.52 s == sender wall
stamps 40.51 s (the phone's two clocks agree to 0.04 s) → the CAPTURE cadence is 5.9 Hz, and the 73 intervals of
100 ms sit at the start of the session (positions 0–9 consecutive, then sparser): the phone captures at 10 Hz and
settles to 5 Hz as the pipe fills. Recorded ARRIVAL span 44.06 s (p50 186 ms, i.e. 5.4 Hz delivered whatever the
stamp cadence): arrival-minus-sensor lag ramps steadily from 0 to +3.54 s (+1.0 s in the first 30 frames, then
≈ +0.25 s per 30 frames) with 14 catch-up bursts summing to −3.17 s — the transport delivers slower than the phone
stamps, so a queue builds sender-side; that is what the analytics' `input_hz` 5.0 measured in P1 task 5 and what
the new `pose_health.delivery_lag` reports (live runs; see below). Replay ARRIVAL span 46.14 s = the recording's
44.06 s + 2.08 s of replayer pacing drift (8.7 ms per frame: the replayer schedules each sleep after the previous
frame's parse, uncompensated — harmless for the sensor-clock anchor, which is speed-independent by G1-B, but "real
time" replay is 4.7 % slow; parked, master plan §8b). Harness wall ≈ 46.1 s + 25 s `DRAIN_WAIT` + startup ≈ 75.8 s,
which is the "240 frames, 75.8 s" `scripts/benchmark_datasheet.py`'s repro blurb has carried since b016660 — the
blurb now states the spans. Under replay `delivery_lag.end_s` reads 5.62 s (3.54 live + 2.08 replay drift).

Informational: `t_total` mean 312.9 ms (on) vs 360.1 ms (off) — run-to-run GPU variance on identical processing (the
≤ 5 % overhead predicate is G2-C's, over 3 × 3 runs); `events.jsonl` 245 012 B (on) vs 125 754 B (off) → 497 B per
pose line. CPU suite: 804 passed, 1 failed — `examples/rc_car_agent/tests/test_server.py::
test_baseline_no_match_resumes_search_e2e`, a wall-clock e2e of the fake car that fails 3/3 on the untouched main tree
on this box today as well (pre-existing, unrelated; the agent imports nothing this change touches).

## P2 stage B reproduction (observation ledger, 2026-09-22) — `p2-ledgers/stage-b/` — **G2-B HARD GATE PASS**

Branch `feature/p2-observation-ledger` (main `a0962c7` + the observation ledger): `ObservationEvent` kind `obs` (ledger
schema 1), one line per candidate the associator looked at, written on the pipeline thread right after association;
`Associator.update_with_candidates(on_observation=...)` reports every candidate's exit (matched / created /
spawn_capped / no_p_cam / no_embedding / create_failed) with the raw world point, the camera-frame centroid, the winning
match's residuals and the gate audit counters; `WorkingMemory.view_bin_id`; `ledger.observation_summary`; docs.
`p2b_gate.sh` = two headless dual replays of session1 on the packaged replay defaults, diagnostics on, ledgers ON (B_on) /
OFF (B_off); `gate.out` is the verbatim output of the PASSING run.

| run | multiset | dequeue / receiver vs B1 | `obs` lines | predicates |
|---|---|---|---|---|
| B_on | 124/65 @53 `ad6f71a5b89c8506` | identical (86) / identical (240) | 695 (+ 240 `pose`) | 1–6 PASS |
| B_off | 124/65 @53 `ad6f71a5b89c8506` | identical (86) / identical (240) | none | 1–2 PASS |

Counts (3): 408 matched == Σ frame.n_matched, 279 created == Σ frame.n_created, 8 no_p_cam; on every one of the 53
processed frames #obs == `scoring.n_selected`. Logic (4): all 403 scored matches satisfy cos ≥ 0.90, dist ≤ 0.50 m,
px_err ≤ 60; all 93 spawns that had gate survivors have max_cos < 0.90. Logic (5): 0 pose-correction messages in the
replay; for all 124 final objects `xyz_world` lies inside the bbox of the object's own raw `created` + `matched`
`p_world` (the EMA is a convex combination), and every hits == 1 object equals its create point to 1e-5. Logic (6):
for all 124 objects the number of distinct `view_bin` values over its observations equals the WM's `view_bins` count
(the harness's `/objects` list carries the count; the key set is available via `include_vectors=true`).

**Finding — the associator's fallback matches without scoring (5 of 408 matches on session1).** The first gate run
FAILED predicate 1 with 123/65 (`c2589c7d71f8f2a5`) on BOTH B_on and B_off: my first version of the hook had turned
the `else:` after the `fallback_all_when_empty` block into `if cand_ids:`, which made the fallback ids go through the
gates + scoring. In the shipped code they never do: when the index returns nothing and the WM has < 20 objects the
fallback fills `cand_ids` and the branch ends, so `best_id` keeps the PREVIOUS candidate's value — the candidate is
"matched" to that object (with the previous candidate's residuals fed to the WM) or spawned, with no check either
way. Scoring them is arguably the intended behaviour, but it changes the anchor (124 → 123 objects), so stage B
restores the original flow exactly (second run: anchor identical, 6/6) and the ledger marks such lines
`matched_without_scoring: true` with no residuals: 5 of 408 on session1. Pinned by
`test_fallback_path_matches_without_scoring_and_the_ledger_says_so`; parked in master plan §8b — fix with a new anchor.

Recorded: `|p_cam.z − mask.depth_p50| < 0.5 m` for 100 % of 687 lines; residuals of the scored matches cos p50 0.946 /
p95 0.980, dist p50 0.036 / p95 0.138 m, px_err p50 11.3 px; range p50 2.60 / p95 3.59 m; matched per object p50 2 /
max 15; view bins used {11: 491, 12: 196} — 2 of the 24 bins (a forward-looking walk); 279 objects created, 124 alive
at the end (155 protos expired — a P3 "duplicate spawn" input); `timing_ms.ledger` p50 1.02 / p95 1.30 ms per processed
frame; `t_total` mean 217.7 ms (on) vs 217.0 ms (off), +0.3 %; 1 339 B per obs line, events.jsonl 1.18 MB (on) vs
127 KB (off) for the 46 s replay.

## P2 stage C reproduction (view ledger, frame outcomes, Parquet, schema freeze; 2026-09-23) — `p2-ledgers/stage-c/` — **G2-C = G2 HARD GATE PASS 8/8 — P2 EXIT GATE MET**

Branch `feature/p2-view-ledger` (stacked on stage B, PR #43): `ViewEvent` kind `view` — one line per processed frame,
written BEFORE association, listing every live WM object whose stored position projects inside the RGB image with
`u`/`v`, `expected_depth` (camera z of the stored position) and `observed_depth` (NaN-aware 3×3 median of the frame's
depth map at that pixel, mapped RGB→depth with the mask stage's rule); `rtsm/core/frustum.py` = the associator's
`_project_px` vectorised, pinned to it point by point on random poses by `tests/evaluation/test_frustum.py` (the
CLAUDE.md pose-math rule) BEFORE any pipeline wiring; `ledger.frame_outcomes` / `outcome_histogram` (one outcome per
sensor frame across receiver / lane / dequeue lines); the view/obs join in `observation_summary`; `to_parquet` verified
end to end; `obs.label_topk` changed from `[[label, score]]` pairs to `[{label, score}]` records because Arrow cannot
type mixed-type pairs (found by the gate on the real file, not by the unit tests, which were skipped without pyarrow;
pyarrow is now installed on the dev box and a test converts every kind on realistic rows). **Ledger schema 1
(`pose`, `obs`, `view`) is FROZEN with this stage**: later changes add fields or kinds and bump `ledgers.schema`.
`p2c_gate.sh` = six headless dual replays of session1 (C1–C3 on / off, interleaved); `gate.out` is verbatim.

| run | multiset | dequeue / receiver vs B1 | kinds | `t_total` mean |
|---|---|---|---|---|
| C1/C2/C3_on | 124/65 @53 `ad6f71a5b89c8506` ×3 | identical ×3 | pose 240, obs 695, view 53 (+ trace) | 204.2 / 202.1 / 204.6 ms |
| C1/C2/C3_off | 124/65 @53 `ad6f71a5b89c8506` ×3 | identical ×3 | trace only | 204.6 / 204.2 / 203.5 ms |

Predicates: (1) anchor + sequences identical on all six; (2) off runs carry no ledger kind, on runs have `#view == #frame
== 53` with a 1:1 stamp join, pose 240, obs 695; (3) **logic — the projection agrees with the associator on real data:**
402 of 403 scored matches have their object in that frame's view list (99.75 %); the one exception matched at
`px_err` 55.4 px of the 60 px reprojection gate, i.e. the stored position sat just outside the image edge (a 16-hit
object); 0 of 279 created objects were already in a view list; (4) **logic — depth:** of the 390 matched-in-view pairs
with an observed depth, 388 agree within 0.30 m (99.5 %; |err| p50 0.027 m, p95 0.134 m); the two beyond are one object
seen at expected 2.75–2.79 m vs observed 2.44–2.49 m, a stored centroid ≈ 0.3 m behind the visible surface — the
along-ray / partial-view bias P4's attribution is built to separate; (5) **overhead (the G2 number):** `t_total` mean
203.6 ms (on) vs 204.1 ms (off), ratio 0.998 — the ledgers' cost is inside run-to-run noise; `timing_ms.view` p50 1.30 /
p95 2.03 ms and `timing_ms.ledger` p50 0.98 / p95 1.26 ms per processed frame; (6) `frame_outcomes` == the P1 counters
(processed 53, gate_rejected:skip 29, gate_rejected:near_recent_keyframe 4, throttled 154; 240 frames, none left
enqueued); (7) Parquet: all six kinds written with row counts equal to the JSONL (1 551 842 B → 264 228 B, 5.9×); (8)
meta schema_version 3, ledgers schema 1, every view line `v1_occlusion_agnostic`, `LEDGER_KINDS == (pose, obs, view)`.

**Recorded (P3 inputs, occlusion-agnostic model):** in-frustum objects per processed frame p50 32 / max 63 out of a
live memory p50 106 / max 126; over the 53 frames 398 in-frustum object-views were matched and 1 235 were not (24 % —
the raw "detection rate over in-frustum views" before visibility from the depth pair is applied); 5 834 B per view line;
`events.jsonl` 1.55 MB (on) vs 129 KB (off) for the 46 s replay ≈ 34 KB/s with every ledger on at this cadence.
CPU suite: 841 passed, 1 failed (the RC-car fake-car e2e, pre-existing and flaky: it fails on main and passed on the
stage-B run of the same tree).
