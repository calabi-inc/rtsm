# rtsm eval report — `rgbd_dataset_freiburg1_desk.bag`

> **Floor not established: 1 run(s).** Every number below needs at least 3 same-input runs to carry a floor (`rtsm eval … --repeats 3`). Read the values as one sample.

## Run

| | |
|---|---|
| input | `recordings/external/rgbd_dataset_freiburg1_desk.bag` (bag) |
| mode / cadence | `as_deployed` — **deployed (as_deployed)** |
| keyframe rule | every 30 admitted frames |
| non-keyframe throttle | 0.5 s on the sensor clock |
| sweep gate | enforce |
| ingest policy | lossless |
| repeats | 1 — fingerprints identical |
| config fingerprint | `e57e33b0b91da0c91579eab027b3c5feb0193bfd48f2b1b9ed2dcf4cee2e5908` |
| commit / rtsm / python | `8278fadfa445` / None / 3.12.10 |
| cluster radius | 0.5 m (the associator's distance gate); sensitivity at 0.25, 0.5, 1 m |
| wall time per run | 33.3 s |

The floor of a number is its spread over the 1 same-input runs: a difference between two bags or two versions smaller than the floor means nothing.

## Frames and admission

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| sensor frames seen | 300 | insufficient (n=1) |
| admitted (enqueued) | 31 | insufficient (n=1) |
| throttled | 269 | insufficient (n=1) |
| throttled fraction | 89.7 % | insufficient (n=1) |
| dropped: tracking not normal | 0 | insufficient (n=1) |
| processed | 22 | insufficient (n=1) |
| keyframes processed | 11 | insufficient (n=1) |
| sweep gate rejected | 9 | insufficient (n=1) |
| frame-quality rejected | 0 | insufficient (n=1) |
| gap between processed frames p50 (s) | 0.500 | insufficient (n=1) |
| gap between processed frames p95 (s) | 0.632 | insufficient (n=1) |
| longest unprocessed stretch (s) | 0.700 | insufficient (n=1) |
| processed span (s) | 10.300 | insufficient (n=1) |

Dequeue outcomes (run 1): `gate_rejected:near_recent_keyframe` 9, `processed:hard_max_s` 11, `processed:keyframe` 11.

## Pose stream

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| pose lines | 300 | insufficient (n=1) |
| sensor rate (Hz) | 29.029 | insufficient (n=1) |
| span (s) | 10.300 | insufficient (n=1) |
| stamp jitter (ms) | 0.141 | insufficient (n=1) |
| gaps | 9 | insufficient (n=1) |
| tracking-limited episodes | 0 | insufficient (n=1) |
| tracking-limited fraction | 0.0 % | insufficient (n=1) |
| discontinuities (> 0.5 m + 1 m/s·dt) | 0 | insufficient (n=1) |
| pose parse errors | 0 | insufficient (n=1) |
| depth valid fraction p50 | 74.3 % | insufficient (n=1) |
| confidence-2 fraction p50 | – | – |

## Memory

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects at the end | 168 | insufficient (n=1) |
| confirmed | 29 | insufficient (n=1) |
| objects created over the run | 175 | insufficient (n=1) |
| transient (created, not in the final memory) | 7 | insufficient (n=1) |
| survivors with a single hit | 109 | insufficient (n=1) |
| matches made without scoring (associator fallback) | 1 | insufficient (n=1) |

Fingerprint (run 1): `f765afea21b9dbd8`. Hits histogram: 1: 109, 2: 30, 3-5: 25, 6-10: 3, 11+: 1. View bins per object: 1: 152, 2: 16.

## Spatial clusters (proxy for physical objects)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters at 0.5 m | 17 | insufficient (n=1) |
| … with at least one surviving object | 17 | insufficient (n=1) |
| … with more than one object | 14 | insufficient (n=1) |
| duplicate objects over the run (ids − clusters) | 158 | insufficient (n=1) |
| duplicate objects in the final memory | 151 | insufficient (n=1) |
| clusters at 0.25 m (sensitivity) | 40 | insufficient (n=1) |
| clusters at 0.5 m (sensitivity) | 17 | insufficient (n=1) |
| clusters at 1 m (sensitivity) | 7 | insufficient (n=1) |

A cluster groups objects whose median raw positions lie within the radius (leader clustering in creation order, no chaining). Two real objects closer than the radius fall into one cluster; a duplicate spawn farther than the radius is not seen. The sensitivity rows bound that.

## Detection over in-frustum views

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| views (cluster in the frustum on a processed frame) | 154 | insufficient (n=1) |
| re-identified (a member matched) | 66 | insufficient (n=1) |
| duplicated (a member created instead) | 24 | insufficient (n=1) |
| missed | 64 | insufficient (n=1) |
| detection rate (re-identified + duplicated) | 58.4 % | insufficient (n=1) |
| re-identification rate | 42.9 % | insufficient (n=1) |

By expected range (object level, run 1):

| range (m) | views | re-identified | rate |
|---|---|---|---|
| 0.50-1.00 | 412 | 62 | 0.150 |
| 1.00-1.50 | 472 | 48 | 0.102 |
| 1.50-2.00 | 112 | 8 | 0.071 |
| 2.00-2.50 | 62 | 2 | 0.032 |
| 2.50-3.00 | 12 | 0 | 0.000 |

The frustum model is occlusion-agnostic (`v1_occlusion_agnostic`): an object behind another counts as a view, so every rate here is a lower bound on what the perception could see.

## Label disagreement

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 2 labelled observations | 59 | insufficient (n=1) |
| disagreement mean (1 − modal share) | 32.5 % | insufficient (n=1) |
| disagreement p50 | 33.3 % | insufficient (n=1) |
| disagreement p95 | 57.4 % | insufficient (n=1) |
| objects whose observations disagree | 72.9 % | insufficient (n=1) |
| clusters whose surviving objects carry different primary labels | 13 | insufficient (n=1) |

## Position scatter (raw observations vs the object median)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 3 observations | 29 | insufficient (n=1) |
| observations | 126 | insufficient (n=1) |
| along-ray RMS (m) | 0.039 | insufficient (n=1) |
| lateral RMS (m) | 0.051 | insufficient (n=1) |
| along / lateral | 0.761 | insufficient (n=1) |
| |along| vs range: slope (m per m) | 0.022 | insufficient (n=1) |
| |along| vs range: intercept (m) | -0.008 | insufficient (n=1) |
| |along| vs range: R² | 0.025 | insufficient (n=1) |
| lateral vs range: slope (m per m) | 0.031 | insufficient (n=1) |

| range (m) | n | along RMS (m) | lateral RMS (m) |
|---|---|---|---|
| 0.50-1.00 | 44 | 0.018 | 0.012 |
| 1.00-1.50 | 75 | 0.046 | 0.062 |
| 1.50-2.00 | 5 | 0.036 | 0.019 |
| 2.00-2.50 | 2 | 0.077 | 0.132 |

## Duplicate spawns

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects created within the radius of a live object | 166 | insufficient (n=1) |
| … as a fraction of created objects | 94.9 % | insufficient (n=1) |
| … while the original was in the frustum | 125 | insufficient (n=1) |
| reason: original not returned by the index | 5 | insufficient (n=1) |
| reason: nearby but failed the distance / z / reprojection gates | 39 | insufficient (n=1) |
| reason: passed the gates, cosine below 0.9 | 122 | insufficient (n=1) |
| reason: other | 0 | insufficient (n=1) |

Alive = a surviving object, or one observed less than 10 s earlier (the proto TTL). The reason classes read the created line's audit fields (`n_nearby`, `n_gate_survivors`, `max_cos`).

## Revisits

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters seen in ≥ 2 visits | 0 | insufficient (n=1) |
| revisits | 0 | insufficient (n=1) |
| re-identified on return | 0 | insufficient (n=1) |
| duplicated on return | 0 | insufficient (n=1) |
| missed on return | 0 | insufficient (n=1) |
| re-identification lower bound on revisits | – | – |

A visit ends when the cluster leaves the frustum for ≥ 5 s of sensor time. Missed and duplicated revisits count as failures, so the bound is a lower bound (occlusion is a miss here).

## Worst moments (run 1)

| t (s) | frame | kind | score | detail |
|---|---|---|---|---|
| 1.5679 | 64 | frame | 27.000 | missed 1, duplicate_spawns 13, in_frustum 5, reidentified 2 |
| 2.5683 | 94 | frame | 25.000 | missed 3, duplicate_spawns 11, in_frustum 6, reidentified 2 |
| 4.6682 | 157 | frame | 24.000 | missed 4, duplicate_spawns 10, in_frustum 9, reidentified 3 |
| 4.9679 | 166 | frame | 24.000 | missed 4, duplicate_spawns 10, in_frustum 9, reidentified 4 |
| 6.7 | 218 | frame | 24.000 | missed 4, duplicate_spawns 10, in_frustum 10, reidentified 3 |
| 5.6999 | 188 | frame | 23.000 | missed 5, duplicate_spawns 9, in_frustum 10, reidentified 3 |
| 1.9681 | 76 | frame | 21.000 | missed 3, duplicate_spawns 9, in_frustum 6, reidentified 3 |
| 0.0 | 17 | frame | 20.000 | duplicate_spawns 10 |
| 0.968 | 46 | frame | 20.000 | duplicate_spawns 10, in_frustum 2, reidentified 2 |
| 5.1999 | 173 | frame | 20.000 | missed 2, duplicate_spawns 9, in_frustum 10, reidentified 3 |

Score = missed clusters + 2 × duplicate spawns (+ 1 when the frame-quality gate rejected the frame); pose gaps / discontinuities / tracking-limited episodes score by their size. `t` is sensor time since the first pose line.

## Largest clusters (run 1)

| cluster | objects (surviving) | observations | first–last seen (s) | re-id rate | visits | top labels | surviving labels |
|---|---|---|---|---|---|---|---|
| c0005 | 48 (48) | 82 | 0.968–8.2999 | 0.556 | 1 | pillow (6), sheet (5), cup (4) | 3D glasses, beeper, bowl, cash, converter, cup, document, edge, fax, hammer, jack, matchbox, mirror, mouse, office desk, pall, paper, pencil sharpener, pet toy, photo frame, pillow, pot, ratchet, remove, router, sheet, shoe, snack bag, soap, staple, stapler, stationery, table, tea bag, tool |
| c0008 | 37 (37) | 55 | 1.5679–10.3001 | 0.611 | 1 | album (12), beeper (5), tape (4) | DVD, album, app icon, array, beeper, beetle, bottle, cigar box, converter, crab, fax, film format, gift card, glass, lottery, matchbox, picture, pillow, publication, remove, sheet, stationery, tape, wasp |
| c0006 | 18 (18) | 47 | 0.968–10.3001 | 0.722 | 1 | shoe (5), laptop (5), laptop keyboard (5) | beeper, computer, computer chair, crt screen, knife, laptop, laptop keyboard, matchbox, monitor, screen, screenshot, shoe, television, thermos |
| c0007 | 23 (23) | 37 | 0.968–10.3001 | 0.556 | 1 | mouse (4), office desk (4), hardware (3) | 3D glasses, assemble, audio, bureau, chair, desktop, edge, hardware, laptop, mousepad, office cubicle, office desk, playroom, sheet, storage box, towel, wall, workplace |
| c0000 | 12 (10) | 22 | 0.0–5.1999 | 0.471 | 1 | keyboard (6), mouse (4), sheet (3) | bureau, document, film format, keyboard, mouse, remove, sheet, shoe |
| c0009 | 10 (10) | 21 | 4.9679–8.8679 | 0.500 | 1 | laptop (5), screenshot (4), laptop keyboard (3) | bowl, clutch, keyboard, laptop, laptop keyboard, paper, screenshot, sheet, shoe |
| c0001 | 6 (5) | 11 | 0.0–6.068 | 0.188 | 1 | bowl (4), monitor (2), crt screen (1) | bed, chair, computer, monitor, pet bowl |
| c0002 | 5 (3) | 6 | 0.0–5.6999 | 0.100 | 1 | calculator (1), system (1), remove (1) | book, notepad, remove |
| c0013 | 2 (2) | 4 | 7.2679–8.2999 | 1.000 | 1 | lemon (2), flower (1), calculator (1) | calculator, lemon |
| c0004 | 3 (2) | 3 | 0.0–5.1999 | 0.000 | 1 | sewing machine (1), DVD (1), hardware (1) | DVD, hardware |
| c0011 | 2 (2) | 3 | 5.6999–6.068 | 1.000 | 1 | pillow (1), crt screen (1), sheet (1) | crt screen, pillow |
| c0012 | 2 (2) | 3 | 5.6999–6.068 | 1.000 | 1 | keyboard (1), pillow (1), bowl (1) | bowl, keyboard |
| c0003 | 2 (1) | 2 | 0.0–0.532 | 0.000 | 1 | converter (1), shoe (1) | shoe |
| c0010 | 1 (1) | 2 | 5.1999–6.7 | 0.333 | 1 | soap (1), pillow (1) | soap |
| c0014 | 2 (2) | 2 | 7.2679–9.8681 | 0.000 | 1 | converter (1), lamp (1) | converter, lamp |

## Method notes

- Everything above is computed from the run's ledgers (`events.jsonl`: pose / obs / view lines, the frame-flow trace) and the final memory in `summary.json`. No ground truth, no labels required; label numbers use the detector's top-1 label per observation.
- Objects = every id the associator matched or created; a **transient** object was created and is not in the final memory (a proto that expired, or one the memory evicted).
- Raw observations (`p_world` as the associator computed it) are used everywhere; the memory's smoothed positions appear only in the fingerprint.
- Per-object, per-cluster, per-frame and per-duplicate records are in `metrics.json` (`objects`, `clusters`, `frames`, `duplicate_spawns`, `worst_moments`); each object and cluster carries the sensor stamps of the frames it was seen and missed on.
- Parameters: cluster radius 0.5 m, revisit gap 5 s, range bin 0.5 m, scatter needs ≥ 3 observations, proto TTL 10 s, cosine threshold 0.9.
