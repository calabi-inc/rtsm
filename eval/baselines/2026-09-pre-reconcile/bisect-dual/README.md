# dual attribution sweep — 2026-09-08 (P0 task 1b)

One `scripts/benchmark_datasheet.py dual` run per commit (as checked out at that commit, its own `rtsm.yaml`, empty FAISS store), one retry whenever a run processed 54 frames instead of 53 or errored. Multiset = the `/objects` recipe from `../README.md`. Driver: `sweep_dual.py` (session scratchpad).

## Every result is one of four multisets

| multiset | name | objects/confirmed | frames | occurrences |
|---|---|---|---|---|
| `935bc8eb254c4dfe` | B54 (base/June: 111/74) | 111/74 | 54 | 15 |
| `b71b98ca1fc2bf0d` | A53 (main tip: 107/70) | 107/70 | 53 | 10 |
| `72fd5c7f0475da90` | D54 (demo2 tip: 116/68) | 116/68 | 54 | 10 |
| `1994e0fe5dd6167c` | C53 (demo2 tip: 115/66) | 115/66 | 53 | 3 |

## Per commit

| # | commit | subject | attempt | frames | objects/confirmed | multiset | note |
|---|---|---|---|---|---|---|---|
| 00-main-HEAD-pr28 | `73aa8e2` | Merge pull request #28 from calabi-inc/fix/empty-except | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| 01-base-pr20 | `96e71eb` | Merge pull request #20 from calabi-inc/fix/robot-pose-receive-time | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 01-base-pr20 | `96e71eb` | Merge pull request #20 from calabi-inc/fix/robot-pose-receive-time | 2 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 02-demo2-mergept | `a9770e2` | Merge main (PR #20 robot_pose at receive time) into feature/demo2-rc-c | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| 03 | `9385483` | chore(run): add __main__ guard so `python -m rtsm.run` executes main() | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| 04 | `c75e787` | feat(core): frame_epoch — pose-frame discontinuity marker for agents | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 04 | `c75e787` | feat(core): frame_epoch — pose-frame discontinuity marker for agents | 2 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| 05 | `c5e56e0` | feat(examples): Phase G + H software — calibration, baseline search, E | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| 06 | `386b2e6` | Harden static dir picker: validate assets before serving a frontend | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 06 | `386b2e6` | Harden static dir picker: validate assets before serving a frontend | 2 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 07 | `0ca4b6a` | Fix ghost objects in semantic search after /reset | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 07 | `0ca4b6a` | Fix ghost objects in semantic search after /reset | 2 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| 08 | `120fe7c` | fix(core): BGR/RGB channel swap poisoned the whole perception stack | 1 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 08 | `120fe7c` | fix(core): BGR/RGB channel swap poisoned the whole perception stack | 2 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 09 | `ab2999c` | fix(core): half-dead retrieval boots + receive-time depth clearance | 1 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 09 | `ab2999c` | fix(core): half-dead retrieval boots + receive-time depth clearance | 2 | 53 | 115/66 | C53 (demo2 tip: 115/66) |  |
| 10 | `f7ab20b` | refactor(core): clearance sensing lives in the io layer, where it runs | 1 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 10 | `f7ab20b` | refactor(core): clearance sensing lives in the io layer, where it runs | 2 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 11 | `2e41dc3` | feat(rtsm+rc-car): grounded-label propagation, /search/label, union re | 1 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 11 | `2e41dc3` | feat(rtsm+rc-car): grounded-label propagation, /search/label, union re | 2 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 12 | `ddc1de4` | feat(rc-car+viz): one sweep per decision point; viz search = label+sem | 1 | 53 | 115/66 | C53 (demo2 tip: 115/66) |  |
| 13 | `1504b39` | feat(rtsm+rc-car): judgment-grade snapshots, GDINO 0.30, signed target | 1 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 13 | `1504b39` | feat(rtsm+rc-car): judgment-grade snapshots, GDINO 0.30, signed target | 2 | 53 | 115/66 | C53 (demo2 tip: 115/66) |  |
| 14 | `88a61fe` | chore: perception-package config (GDINO 0.30, judgment-crop knobs) + p | 1 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| 14 | `88a61fe` | chore: perception-package config (GDINO 0.30, judgment-crop knobs) + p | 2 | 54 | 116/68 | D54 (demo2 tip: 116/68) |  |
| m1-pr21 | `bede613` | Merge pull request #21 from calabi-inc/security/postcss-sourcemappingu | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m1-pr21 | `bede613` | Merge pull request #21 from calabi-inc/security/postcss-sourcemappingu | 2 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m2-staticdir | `c9914fc` | Harden static dir picker: validate assets before serving a frontend | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m2-staticdir | `c9914fc` | Harden static dir picker: validate assets before serving a frontend | 2 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m3-pr22-watchdog | `bc33c41` | Merge pull request #22 from calabi-inc/feature/frame-flow-watchdog | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m3-pr22-watchdog | `bc33c41` | Merge pull request #22 from calabi-inc/feature/frame-flow-watchdog | 2 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m4-pr25-tuning | `efe4bc8` | Merge pull request #25 from calabi-inc/feature/config-tuning | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m4-pr25-tuning | `efe4bc8` | Merge pull request #25 from calabi-inc/feature/config-tuning | 2 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| m5-pr26-deadkeys | `aebc6e3` | Merge pull request #26 from calabi-inc/chore/config-dead-keys | 1 | 54 | 111/74 | B54 (base/June: 111/74) |  |
| m5-pr26-deadkeys | `aebc6e3` | Merge pull request #26 from calabi-inc/chore/config-dead-keys | 2 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| m6-pr27-framegate | `918ebbe` | Merge pull request #27 from calabi-inc/feature/frame-quality-gate | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| m7-pr24-persist | `1aaa041` | Merge pull request #24 from calabi-inc/fix/silent-reset-persist-failur | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |
| m8-pr23-posedrop | `5528abf` | Merge pull request #23 from calabi-inc/fix/pose-conversion-drop-frame | 1 | 53 | 107/70 | A53 (main tip: 107/70) |  |

## Reading

1. **The 53-vs-54-frame flip is run-to-run timing noise, not code.** The same commit produced both frame counts (c75e787, ab2999c, 1504b39, efe4bc8 …), and the multiset is a pure function of the frame count on each side of the BGR fix. This is the wall-clock non-keyframe throttle boundary the drop-policy memo predicted; P1's sensor-time throttle removes it.
2. **The main ↔ demo2 `dual` divergence is exactly one commit: `120fe7c` (BGR/RGB channel swap fix).** Every demo2 commit before it reproduces main's multiset for the same frame count; every commit from it onward reproduces demo2's tip multiset for the same frame count. Clearance, frame_epoch, label propagation, judgment crops, GDINO 0.30 and the event log do not move `dual` on session1.
3. **No main-side PR (#21–#28) changed `dual` output.** Each lands on the base multiset (54 frames) or the main-tip multiset (53 frames); there is no third value on that side. June's 111/74 → today's 107/70 was the 54→53 flip, not a regression.
4. **Consequence for G0:** the merged tree carries the BGR fix on purpose, so the merged default-config `dual` anchor is **demo2's pair — 115/66 (`1994e0fe5dd6167c`) on 53 frames or 116/68 (`72fd5c7f0475da90`) on 54 frames** — not main's 107/70. A merged run that reproduces main's multiset would mean the BGR fix was lost in the merge. The gate accepts either variant keyed on `frame_count`.

```
{"label": "_start", "branch": "main", "head": "73aa8e2", "time": "2026-09-08T17:06:43"}
{"label": "_note", "msg": "base != main (111/74 vs 107/70) -> sweeping main first-parent side"}
{"label": "_end", "head": "73aa8e2", "branch": "main", "time": "2026-09-08T18:16:04"}
```
