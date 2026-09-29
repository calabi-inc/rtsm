# rtsm eval report — `session1`

## Run

| | |
|---|---|
| input | `recordings/session1` (replay) |
| mode / cadence | `as_deployed` — **deployed (as_deployed)** |
| keyframe rule | every 30 admitted frames |
| non-keyframe throttle | 0.5 s on the sensor clock |
| sweep gate | enforce |
| ingest policy | lossless |
| repeats | 3 — fingerprints identical |
| config fingerprint | `e57e33b0b91da0c91579eab027b3c5feb0193bfd48f2b1b9ed2dcf4cee2e5908` |
| commit / rtsm / python | `8278fadfa445` / None / 3.12.10 |
| cluster radius | 0.5 m (the associator's distance gate); sensitivity at 0.25, 0.5, 1 m |
| wall time per run | 14.2 s, 13.5 s, 14.8 s |

The floor of a number is its spread over the 3 same-input runs: a difference between two bags or two versions smaller than the floor means nothing.

## Frames and admission

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| sensor frames seen | 240 | 0.000 |
| admitted (enqueued) | 86 | 0.000 |
| throttled | 154 | 0.000 |
| throttled fraction | 64.2 % | 0.0 % |
| dropped: tracking not normal | 0 | 0.000 |
| processed | 53 | 0.000 |
| keyframes processed | 9 | 0.000 |
| sweep gate rejected | 33 | 0.000 |
| frame-quality rejected | 0 | 0.000 |
| gap between processed frames p50 (s) | 0.600 | 0.000 |
| gap between processed frames p95 (s) | 2.045 | 0.000 |
| longest unprocessed stretch (s) | 2.300 | 0.000 |
| processed span (s) | 40.519 | 0.000 |

Dequeue outcomes (run 1): `gate_rejected:near_recent_keyframe` 4, `gate_rejected:skip` 29, `processed:hard_max_s` 30, `processed:keyframe` 9, `processed:parallax` 13, `processed:ttl+novelty` 1.

## Pose stream

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| pose lines | 240 | 0.000 |
| sensor rate (Hz) | 5.898 | 0.000 |
| span (s) | 40.519 | 0.000 |
| stamp jitter (ms) | 0.002 | 0.000 |
| gaps | 0 | 0.000 |
| tracking-limited episodes | 0 | 0.000 |
| tracking-limited fraction | 0.0 % | 0.0 % |
| discontinuities (> 0.5 m + 1 m/s·dt) | 0 | 0.000 |
| pose parse errors | 0 | 0.000 |
| depth valid fraction p50 | 100.0 % | 0.0 % |
| confidence-2 fraction p50 | 76.5 % | 0.0 % |

## Memory

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects at the end | 124 | 0.000 |
| confirmed | 65 | 0.000 |
| objects created over the run | 279 | 0.000 |
| transient (created, not in the final memory) | 155 | 0.000 |
| survivors with a single hit | 43 | 0.000 |
| matches made without scoring (associator fallback) | 5 | 0.000 |

Fingerprint (run 1): `ad6f71a5b89c8506`. Hits histogram: 1: 43, 2: 16, 3-5: 30, 6-10: 27, 11+: 8. View bins per object: 1: 111, 2: 13.

## Spatial clusters (proxy for physical objects)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters at 0.5 m | 49 | 0.000 |
| … with at least one surviving object | 40 | 0.000 |
| … with more than one object | 43 | 0.000 |
| duplicate objects over the run (ids − clusters) | 230 | 0.000 |
| duplicate objects in the final memory | 84 | 0.000 |
| clusters at 0.25 m (sensitivity) | 98 | 0.000 |
| clusters at 0.5 m (sensitivity) | 49 | 0.000 |
| clusters at 1 m (sensitivity) | 19 | 0.000 |

A cluster groups objects whose median raw positions lie within the radius (leader clustering in creation order, no chaining). Two real objects closer than the radius fall into one cluster; a duplicate spawn farther than the radius is not seen. The sensitivity rows bound that.

Across the 3 runs: 49 of run 1's 49 clusters are present in every run, 0 only in some (cluster counts per run: [49, 49, 49]).

## Detection over in-frustum views

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| views (cluster in the frustum on a processed frame) | 470 | 0.000 |
| re-identified (a member matched) | 258 | 0.000 |
| duplicated (a member created instead) | 77 | 0.000 |
| missed | 135 | 0.000 |
| detection rate (re-identified + duplicated) | 71.3 % | 0.0 % |
| re-identification rate | 54.9 % | 0.0 % |

By expected range (object level, run 1):

| range (m) | views | re-identified | rate |
|---|---|---|---|
| 1.00-1.50 | 34 | 2 | 0.059 |
| 1.50-2.00 | 267 | 69 | 0.258 |
| 2.00-2.50 | 685 | 119 | 0.174 |
| 2.50-3.00 | 457 | 152 | 0.333 |
| 3.00-3.50 | 139 | 47 | 0.338 |
| 3.50-4.00 | 25 | 6 | 0.240 |
| 4.00-4.50 | 14 | 3 | 0.214 |
| 4.50-5.00 | 6 | 0 | 0.000 |
| 5.00-5.50 | 4 | 0 | 0.000 |
| 5.50-6.00 | 2 | 0 | 0.000 |

The frustum model is occlusion-agnostic (`v1_occlusion_agnostic`): an object behind another counts as a view, so every rate here is a lower bound on what the perception could see.

## Label disagreement

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 2 labelled observations | 117 | 0.000 |
| disagreement mean (1 − modal share) | 36.4 % | 0.0 % |
| disagreement p50 | 50.0 % | 0.0 % |
| disagreement p95 | 68.9 % | 0.0 % |
| objects whose observations disagree | 74.4 % | 0.0 % |
| clusters whose surviving objects carry different primary labels | 27 | 0.000 |

## Position scatter (raw observations vs the object median)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 3 observations | 65 | 0.000 |
| observations | 421 | 0.000 |
| along-ray RMS (m) | 0.083 | 0.000 |
| lateral RMS (m) | 0.141 | 0.000 |
| along / lateral | 0.590 | 0.000 |
| |along| vs range: slope (m per m) | -0.006 | 0.000 |
| |along| vs range: intercept (m) | 0.047 | 0.000 |
| |along| vs range: R² | 0.002 | 0.000 |
| lateral vs range: slope (m per m) | -0.021 | 0.000 |

| range (m) | n | along RMS (m) | lateral RMS (m) |
|---|---|---|---|
| 1.00-1.50 | 1 | 0.027 | 0.013 |
| 1.50-2.00 | 58 | 0.121 | 0.191 |
| 2.00-2.50 | 102 | 0.111 | 0.155 |
| 2.50-3.00 | 178 | 0.056 | 0.143 |
| 3.00-3.50 | 58 | 0.052 | 0.044 |
| 3.50-4.00 | 18 | 0.069 | 0.025 |
| 4.00-4.50 | 6 | 0.029 | 0.025 |

## Duplicate spawns

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects created within the radius of a live object | 241 | 0.000 |
| … as a fraction of created objects | 86.4 % | 0.0 % |
| … while the original was in the frustum | 193 | 0.000 |
| reason: original not returned by the index | 9 | 0.000 |
| reason: nearby but failed the distance / z / reprojection gates | 139 | 0.000 |
| reason: passed the gates, cosine below 0.9 | 93 | 0.000 |
| reason: other | 0 | 0.000 |

Alive = a surviving object, or one observed less than 10 s earlier (the proto TTL). The reason classes read the created line's audit fields (`n_nearby`, `n_gate_survivors`, `max_cos`).

## Revisits

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters seen in ≥ 2 visits | 12 | 0.000 |
| revisits | 12 | 0.000 |
| re-identified on return | 10 | 0.000 |
| duplicated on return | 1 | 0.000 |
| missed on return | 1 | 0.000 |
| re-identification lower bound on revisits | 83.3 % | 0.0 % |

A visit ends when the cluster leaves the frustum for ≥ 5 s of sensor time. Missed and duplicated revisits count as failures, so the bound is a lower bound (occlusion is a miss here).

## Worst moments (run 1)

| t (s) | frame | kind | score | detail |
|---|---|---|---|---|
| 29.7021 | 297 | frame | 25.000 | missed 5, duplicate_spawns 10, in_frustum 8, reidentified 2 |
| 24.9018 | 249 | frame | 23.000 | missed 3, duplicate_spawns 10, in_frustum 8, reidentified 2 |
| 12.0009 | 120 | frame | 22.000 | missed 2, duplicate_spawns 10, in_frustum 7, reidentified 4 |
| 26.0019 | 260 | frame | 22.000 | missed 4, duplicate_spawns 9, in_frustum 9, reidentified 4 |
| 35.8025 | 358 | frame | 21.000 | missed 5, duplicate_spawns 8, in_frustum 13, reidentified 5 |
| 14.0011 | 140 | frame | 20.000 | duplicate_spawns 10, in_frustum 7, reidentified 3 |
| 31.8022 | 318 | frame | 20.000 | missed 6, duplicate_spawns 7, in_frustum 13, reidentified 5 |
| 14.9011 | 149 | frame | 17.000 | missed 1, duplicate_spawns 8, in_frustum 9, reidentified 6 |
| 11.0009 | 110 | frame | 16.000 | missed 2, duplicate_spawns 7, in_frustum 10, reidentified 4 |
| 23.8017 | 238 | frame | 16.000 | duplicate_spawns 8, in_frustum 5, reidentified 2 |

Score = missed clusters + 2 × duplicate spawns (+ 1 when the frame-quality gate rejected the frame); pose gaps / discontinuities / tracking-limited episodes score by their size. `t` is sensor time since the first pose line.

## Largest clusters (run 1)

| cluster | objects (surviving) | observations | first–last seen (s) | re-id rate | visits | top labels | surviving labels |
|---|---|---|---|---|---|---|---|
| c0007 | 16 (8) | 53 | 4.3004–40.5194 | 0.737 | 2 | laptop (13), amplifier (4), knife (4) | brake, fume hood, knife, laptop, storage box, television |
| c0005 | 8 (6) | 44 | 0.0–37.5026 | 0.917 | 2 | folder (11), paper (5), fax (5) | book, card box, fax, keycard, legend, light switch |
| c0018 | 18 (5) | 43 | 12.0009–19.8015 | 0.846 | 1 | assemble (9), whiteboard (5), folder (3) | amplifier, assemble, bag, file cabinet, mattress |
| c0001 | 11 (8) | 37 | 0.0–40.5194 | 0.846 | 2 | pet toy (23), pillow (5), tinsel (3) | doll, pet toy, pillow, pitcher plant |
| c0041 | 6 (6) | 32 | 29.7021–35.8025 | 1.000 | 1 | fan (11), portable air conditioner (8), heater (4) | fan, foam roller, heater, window |
| c0020 | 16 (3) | 28 | 12.501–19.8015 | 0.900 | 1 | paper towel (4), wall (4), soap (3) | paper towel, router, socket |
| c0027 | 9 (3) | 28 | 18.0014–23.8017 | 0.875 | 1 | blanket (6), diaper bag (5), pack (4) | diaper bag, pack, shoe |
| c0037 | 13 (4) | 28 | 24.9018–32.8023 | 0.750 | 1 | remote (6), laptop (4), speaker (3) | camera lens, domino, remote, speaker |
| c0000 | 5 (3) | 26 | 0.0–40.5194 | 0.857 | 2 | pet toy (16), flower (3), hamster (1) | carnation, hamster, pet toy |
| c0031 | 11 (2) | 23 | 20.5015–27.602 | 0.368 | 1 | shoe (4), bag (3), bottle (3) | electronic, tube |
| c0030 | 18 (5) | 22 | 19.8015–34.4024 | 0.190 | 1 | knife (7), shoe (3), phone (2) | converter, hammer, knife, wrench |
| c0015 | 5 (3) | 19 | 10.4008–17.0013 | 0.727 | 1 | drawer (11), whiteboard (4), edge (2) | drawer, whiteboard |
| c0008 | 8 (3) | 17 | 4.3004–40.5194 | 0.727 | 2 | lamp (8), pot (3), pillow (1) | lamp, pot |
| c0021 | 6 (1) | 17 | 12.501–22.6017 | 0.786 | 1 | printer (4), mattress (2), flip (1) | printer |
| c0042 | 5 (5) | 16 | 29.7021–35.8025 | 0.833 | 1 | speaker (2), binder (2), pen (2) | dehumidifier, fax, speaker, system |

## Method notes

- Everything above is computed from the run's ledgers (`events.jsonl`: pose / obs / view lines, the frame-flow trace) and the final memory in `summary.json`. No ground truth, no labels required; label numbers use the detector's top-1 label per observation.
- Objects = every id the associator matched or created; a **transient** object was created and is not in the final memory (a proto that expired, or one the memory evicted).
- Raw observations (`p_world` as the associator computed it) are used everywhere; the memory's smoothed positions appear only in the fingerprint.
- Per-object, per-cluster, per-frame and per-duplicate records are in `metrics.json` (`objects`, `clusters`, `frames`, `duplicate_spawns`, `worst_moments`); each object and cluster carries the sensor stamps of the frames it was seen and missed on.
- Parameters: cluster radius 0.5 m, revisit gap 5 s, range bin 0.5 m, scatter needs ≥ 3 observations, proto TTL 10 s, cosine threshold 0.9.
