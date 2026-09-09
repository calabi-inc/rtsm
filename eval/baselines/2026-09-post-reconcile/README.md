# Post-reconcile anchor — merged tree `3fb0fc2` (G0 record, 2026-09-08)

Same harness and recipe as `../2026-09-pre-reconcile/` (`scripts/benchmark_datasheet.py`, replay of `recordings/session1`, empty FAISS store per run, multiset = `(label_primary, xyz_world@1mm, hits, confirmed)` over the first-100 `/objects` page, sha256[:16]). Three interleaved repeats of each of: `dual` (default config), `grounded_sam2` (default config), `grounded_sam2 --profile examples/rc_car_agent/e1-demo2.profile.yaml` (the E1 perception SETTINGS replayed on session1 — not an E1 evaluation; E1 is scored from the trial logger in the frozen paper tree).

- packaged `rtsm/cfg/rtsm.yaml` sha256[:16] = `c3f8851670931607`
- `examples/rc_car_agent/e1-demo2.profile.yaml` sha256[:16] = `fcca868bb52d9752`
- `objects_full` (the whole `/objects` map, limit 500) is recorded alongside the page sha from this anchor on; only the page sha is comparable to the pre-reconcile shas.

| job | run | objects/confirmed | frames | page sha | matches | full sha (count) | upserts | pose_conv_fail | gate log lines | t_total mean |
|---|---|---|---|---|---|---|---|---|---|---|
| dual | 1 | 116/68 | 54 | `72fd5c7f0475da90` | POST-FLIP 116/68 @54 | `f1900cae5289bd9a` (116) | 249 | 0 | 0 | 0.2261 |
| grounded_sam2 | 1 | 127/72 | 53 | `6b5a2ca503fe3af5` | new anchor | `7ff6d8935e275c17` (127) | 269 | 0 | 0 | 0.9466 |
| grounded_sam2-e1settings | 1 | 25/18 | 54 | `a556dbdcad9fef0a` | demo2's session1 multiset with the E1 perception settings | `a556dbdcad9fef0a` (25) | 77 | 0 | 0 | 0.257 |
| dual | 2 | 116/68 | 54 | `72fd5c7f0475da90` | POST-FLIP 116/68 @54 | `f1900cae5289bd9a` (116) | 243 | 0 | 0 | 0.2391 |
| grounded_sam2 | 2 | 122/72 | 53 | `5ec1337f412da431` | new anchor | `7568b03f05810dd1` (122) | 281 | 0 | 0 | 0.959 |
| grounded_sam2-e1settings | 2 | 25/18 | 54 | `a556dbdcad9fef0a` | demo2's session1 multiset with the E1 perception settings | `a556dbdcad9fef0a` (25) | 78 | 0 | 0 | 0.2573 |
| dual | 3 | 116/68 | 54 | `72fd5c7f0475da90` | POST-FLIP 116/68 @54 | `f1900cae5289bd9a` (116) | 249 | 0 | 0 | 0.2385 |
| grounded_sam2 | 3 | 122/72 | 53 | `5ec1337f412da431` | new anchor | `7568b03f05810dd1` (122) | 284 | 0 | 0 | 0.9875 |
| grounded_sam2-e1settings | 3 | 25/18 | 54 | `a556dbdcad9fef0a` | demo2's session1 multiset with the E1 perception settings | `a556dbdcad9fef0a` (25) | 70 | 0 | 0 | 0.2551 |

## G0 verdict

- **dual:** run1 116/68@54 POST-FLIP 116/68 @54; run2 116/68@54 POST-FLIP 116/68 @54; run3 116/68@54 POST-FLIP 116/68 @54
- **grounded_sam2:** run1 127/72@53 new anchor; run2 122/72@53 new anchor; run3 122/72@53 new anchor
- **grounded_sam2-e1settings:** run1 25/18@54 demo2's session1 multiset with the E1 perception settings; run2 25/18@54 demo2's session1 multiset with the E1 perception settings; run3 25/18@54 demo2's session1 multiset with the E1 perception settings

Rules: `dual` must be POST-FLIP for every run (the BGR/RGB fix is landing on purpose; a PRE-FLIP sha means it was lost, any other sha is a reconcile regression). `grounded_sam2` default is a NEW anchor (main's 133–139/81 was measured pre-flip). `grounded_sam2-e1` is compared against demo2's 25/18 (counts; the multiset was already mm-jittery run to run at f7a0880). `pose_conversion_failures` must be 0 and no frame-quality-gate skip lines may appear in the run log (PR #27's 0-rejections baseline). Expected, pre-declared non-G0 deltas vs the pre-reconcile main baselines: `upserts_total` (the `ltm_min_view_bins` starvation fix now upserts single-bin objects).

```
{"name": "_start", "head": "3fb0fc2", "branch": "chore/reconcile-demo2-2026-09", "time": "2026-09-08T19:06:41"}
{"name": "_end", "time": "2026-09-08T19:23:45"}
```
