# Post-reconcile anchor, HEADLESS (`--no-viz`) — tree `430256b` (2026-09-08)

Same recipe as `../2026-09-post-reconcile/` except the benchmark harness now runs headless by default (`visualization.enable=false` + `--no-viz`; `--viz` restores the old behaviour). This directory is the **forward reference** for P1's G1-A wall-parity gate; `../2026-09-post-reconcile/` remains the record of the G0 gate as passed (viz on, comparable to the pre-reconcile and demo2 baselines, which were all measured with viz on).

## Why the numbers differ from the viz-on anchor

With no viz server the pipeline step is faster (no per-frame JPEG encode / broadcast), so the wall-clock non-keyframe throttle and ingest-gate TTL admit a slightly different set of non-keyframes. The effect on session1: the **confirmed objects are identical** to the viz-on anchor; the difference is a handful of extra unconfirmed protos with 1–2 hits. This is the same wall-clock coupling behind the 53/54-frame flip, and it is what P1's sensor-time clock (execution plan P1 task 1) removes. Tier-2 `latency_hourly` columns and `input_hz` are empty/0 headless (the rollup lives in the viz push loop until P1 task 5).

| job | run | objects/confirmed | frames | full sha | confirmed-only sha | confirmed identical to a same-frame-count viz-on run? | objects only here / only there | t_total mean | pcf | gate lines |
|---|---|---|---|---|---|---|---|---|---|---|
| dual | 1 | 121/66 | 53 | `92e0a8f1206d77da` | `667439a458bf4296` | no same-frame-count viz-on sample |  | 0.2084 | 0 | 0 |
| dual | 2 | 121/66 | 53 | `92e0a8f1206d77da` | `667439a458bf4296` | no same-frame-count viz-on sample |  | 0.1926 | 0 | 0 |
| dual | 3 | 121/66 | 53 | `92e0a8f1206d77da` | `667439a458bf4296` | no same-frame-count viz-on sample |  | 0.1906 | 0 | 0 |
| grounded_sam2 | 1 | 134/72 | 53 | `316bcdc8028a09b9` | `973f580b69fa679a` | YES (vs viz-on run1.json) | +7 / -0 (unconf: 7) | 0.84 | 0 | 0 |
| grounded_sam2 | 2 | 134/72 | 53 | `316bcdc8028a09b9` | `973f580b69fa679a` | YES (vs viz-on run1.json) | +7 / -0 (unconf: 7) | 0.8648 | 0 | 0 |
| grounded_sam2 | 3 | 134/72 | 53 | `316bcdc8028a09b9` | `973f580b69fa679a` | YES (vs viz-on run1.json) | +7 / -0 (unconf: 7) | 0.8247 | 0 | 0 |
| grounded_sam2-e1settings | 1 | 25/18 | 54 | `a556dbdcad9fef0a` | `853aa22a4da1442a` | YES (vs viz-on run1.json) | +0 / -0 (unconf: 0) | 0.2323 | 0 | 0 |
| grounded_sam2-e1settings | 2 | 25/18 | 54 | `a556dbdcad9fef0a` | `853aa22a4da1442a` | YES (vs viz-on run1.json) | +0 / -0 (unconf: 0) | 0.2435 | 0 | 0 |
| grounded_sam2-e1settings | 3 | 25/18 | 54 | `a556dbdcad9fef0a` | `853aa22a4da1442a` | YES (vs viz-on run1.json) | +0 / -0 (unconf: 0) | 0.2269 | 0 | 0 |

## Same-frame-count check (dual, 54 frames)

- headless check run (the first `--no-viz` verification, 54 frames): 122 objects / 68 confirmed, full sha `fc6d9aea5df23786`; vs viz-on run1.json (116/68): confirmed-only multiset identical = True; objects only headless +6 (unconfirmed: 6), only viz-on -0

## Headless floor (use these for G1-A)

- **dual:** objects 121–121, confirmed 66–66; full multiset identical across runs: True; confirmed-only multiset identical across runs: True; per run: 121/66@53, 121/66@53, 121/66@53
- **grounded_sam2:** objects 134–134, confirmed 72–72; full multiset identical across runs: True; confirmed-only multiset identical across runs: True; per run: 134/72@53, 134/72@53, 134/72@53
- **grounded_sam2-e1settings:** objects 25–25, confirmed 18–18; full multiset identical across runs: True; confirmed-only multiset identical across runs: True; per run: 25/18@54, 25/18@54, 25/18@54

```
{"name": "_start", "head": "430256b", "branch": "chore/reconcile-demo2-2026-09", "time": "2026-09-08T19:48:19"}
{"name": "_end", "time": "2026-09-08T20:04:32"}
```
