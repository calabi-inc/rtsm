# Pre-reconcile baselines — 2026-09-07 (Gate 4.5 P0, task 1)

Captured on the RTX 5090 dev box (Windows 11, Python 3.12.10, torch 2.11.0+cu128, driver 616.64) with
`scripts/benchmark_datasheet.py <backend>` — one `python -m rtsm --replay recordings/session1` per run, each branch's
own committed `rtsm/cfg/rtsm.yaml` (the harness patches only `segmentation.backend` + `visualization.enable` and restores
the file in `finally`). Three repeats per (branch, backend), interleaved dual → grounded_sam2. The persisted FAISS index
was moved aside before the batch and restored after; each run started from an empty `model_store/faiss`.

| Branch dir | Commit | Tag |
|---|---|---|
| `main-5528abf/` | `5528abf` origin/main after PRs #23 #24 #25 #26 #27 (PR #28 still open, logging-only) | `pre-reconcile-2026-09` |
| `demo2-f7a0880/` | `f7a0880` tip of `feature/demo2-rc-car-agent` (E1 dataset sealed at `49ec2d9`, paper/** frozen at this tip) | — |
| `june-2026-reference/` | the April/June JSONs that lived in `reports/` (gitignored) before this batch overwrote them | — |

Per-branch `floor.json` + `README.md` hold the per-run numbers. Object identity is compared as the multiset of
`(label_primary, xyz_world rounded to 1 mm, hits, confirmed)` over the `/objects` page (first 100 objects — a harness cap).

## Measured floors (the G0 gate values)

| branch | backend | objects (3 runs) | confirmed | frames processed | t_total mean | verdict |
|---|---|---|---|---|---|---|
| main | dual | 107 / 107 / 107 | 70 / 70 / 70 | 53 / 53 / 53 | 0.27–0.29 s | **bitwise identical multiset — floor 0** |
| main | grounded_sam2 | 139 / 133 / 133 | 81 / 81 / 81 | 53 / 53 / 53 | 1.05–1.08 s | **objects spread 6**, confirmed 0; runs 2–3 identical, run 1 differs |
| demo2 | dual | 115 / 116 / 115 | 66 / 68 / 66 | 53 / 54 / 53 | 0.29–0.32 s | runs 1 and 3 identical; run 2 processed **one extra frame** → +1 object / +2 confirmed |
| demo2 | grounded_sam2 | 25 / 25 / 25 | 18 / 18 / 18 | 53 / 54 / 54 | 0.36–0.38 s | counts floor 0; multiset differs run 1 vs 2–3 (mm-level xyz). NB: demo2's committed yaml carried the E1 perception settings (box 0.30 + five-class vocabulary), so this is session1 seen through those settings — not an E1 result |

**G0 targets derived from this:** merged main with default config must reproduce `dual` 107/70 with an **identical**
multiset, and `grounded_sam2` within 133–139 / 81; merged main with `--profile e1-demo2` must reproduce
`grounded_sam2` 25/18 (the profile carries demo2's `box_threshold 0.30` + 5-class E1 vocabulary — that vocabulary,
not the threshold, is why demo2's grounded_sam2 finds 25 objects vs main's ~135). This is a config-survival check on the
session1 replay; E1 itself is evaluated from the trial logger (`examples/rc_car_agent/paper/**`, frozen), which none of these runs touch.

## Findings worth carrying forward

1. **Frame-admission jitter is real and it is the wall-clock throttle.** demo2 processed 53 frames on runs 1/3 and 54
   on run 2 (grounded_sam2: 53/54/54) — one throttle-boundary flip — and that single frame moved dual by +1 object /
   +2 confirmed. Given the frame set, the pipeline is deterministic. This is exactly P1's G1-A/G1-B split: the
   sensor-time throttle is what makes replay reproducible.
2. **grounded_sam2 on main is not run-to-run deterministic** (139 vs 133 objects at identical confirmed counts). Any
   fixed "±2" tolerance was never meaningful for it; the measured spread is 6.
3. **`dual` differs between the branches on identical config: main 107/70 vs demo2 115/66.** The reviewers' claim
   that dual is "config-clean across branches" holds for the yaml but the *code* diverges (demo2's association /
   pipeline changes). The reconcile must explain which side the merged tree reproduces — this is the first thing G0
   will surface.
4. **Hourly-bucket columns are not comparable across runs.** Tier-2 rollup starts only when the browser tab attaches
   (bucket counts 39 / 29 / 12 on main dual, 12 / 12 / 6 on demo2 dual), so `gate_rejections`, `queue_depth_max`,
   `throttle_skips` sums are partial. Trust `working_memory.*`, the `/objects` multiset, `frame_count`, and
   `latency.t_total` only. (P1 task 5 moves the rollup to a headless timer.)
5. **June → today on main dual: 111/74 → 107/70, 54 → 53 frames.** `pose_conversion_failures = 0` and
   `frame_rejections = 0`, so neither PR #23 nor PR #27 explains it; it is a main-branch change since June or the
   same throttle-boundary effect as (1). Not investigated further — today's floors are the anchor, June is reference.
6. Harness note: `benchmark_datasheet.py` treats every CLI argument as a backend name; passing `--help` runs a
   (failing) replay and rewrites `rtsm/cfg/rtsm.yaml` — a stray `rtsm.yaml.datasheet_bak.<pid>` from Sep 6 (a killed
   run) was found and the working yaml restored from HEAD before this batch (backup preserved in `%TEMP%`).
