# rtsm eval report — `session1`

> **Floor not established: 1 run(s).** Every number below needs at least 3 same-input runs to carry a floor (`rtsm eval … --repeats 3`). Read the values as one sample.

## Run

| | |
|---|---|
| input | `recordings/session1` (replay) |
| mode / cadence | `every_frame` — **exhaustive (every_frame)** |
| keyframe rule | every 1.0 s of sensor time |
| non-keyframe throttle | 0.0 s on the sensor clock |
| sweep gate | shadow (shadow: every processed frame also reported restricted to the frames the deployed gate would have admitted — *masked*) |
| ingest policy | lossless |
| repeats | 1 — fingerprints identical |
| config fingerprint | `e57e33b0b91da0c91579eab027b3c5feb0193bfd48f2b1b9ed2dcf4cee2e5908` |
| commit / rtsm / python | `8278fadfa445` / None / 3.12.10 |
| cluster radius | 0.5 m (the associator's distance gate); sensitivity at 0.25, 0.5, 1 m |
| wall time per run | 55.9 s |

The floor of a number is its spread over the 1 same-input runs: a difference between two bags or two versions smaller than the floor means nothing.

## Frames and admission

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| sensor frames seen | 240 | insufficient (n=1) |
| admitted (enqueued) | 240 | insufficient (n=1) |
| throttled | 0 | insufficient (n=1) |
| throttled fraction | 0.0 % | insufficient (n=1) |
| dropped: tracking not normal | 0 | insufficient (n=1) |
| processed | 240 | insufficient (n=1) |
| keyframes processed | 39 | insufficient (n=1) |
| sweep gate rejected | 0 | insufficient (n=1) |
| frame-quality rejected | 0 | insufficient (n=1) |
| gap between processed frames p50 (s) | 0.200 | insufficient (n=1) |
| gap between processed frames p95 (s) | 0.200 | insufficient (n=1) |
| longest unprocessed stretch (s) | 0.217 | insufficient (n=1) |
| processed span (s) | 40.519 | insufficient (n=1) |
| frames the deployed gate would have rejected | 168 | insufficient (n=1) |
| … fraction of processed | 70.0 % | insufficient (n=1) |

Dequeue outcomes (run 1): `processed:hard_max_s` 33, `processed:keyframe` 39, `processed:near_recent_keyframe` 12, `processed:skip` 156.

## Pose stream

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| pose lines | 240 | insufficient (n=1) |
| sensor rate (Hz) | 5.898 | insufficient (n=1) |
| span (s) | 40.519 | insufficient (n=1) |
| stamp jitter (ms) | 0.002 | insufficient (n=1) |
| gaps | 0 | insufficient (n=1) |
| tracking-limited episodes | 0 | insufficient (n=1) |
| tracking-limited fraction | 0.0 % | insufficient (n=1) |
| discontinuities (> 0.5 m + 1 m/s·dt) | 0 | insufficient (n=1) |
| pose parse errors | 0 | insufficient (n=1) |
| depth valid fraction p50 | 100.0 % | insufficient (n=1) |
| confidence-2 fraction p50 | 76.5 % | insufficient (n=1) |

## Memory

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects at the end | 306 | insufficient (n=1) |
| confirmed | 185 | insufficient (n=1) |
| objects created over the run | 654 | insufficient (n=1) |
| transient (created, not in the final memory) | 348 | insufficient (n=1) |
| survivors with a single hit | 110 | insufficient (n=1) |
| matches made without scoring (associator fallback) | 0 | insufficient (n=1) |

Fingerprint (run 1): `908b36b90b152b33`. Hits histogram: 1: 110, 2: 11, 3-5: 68, 6-10: 38, 11+: 79. View bins per object: 1: 278, 2: 28.

## Spatial clusters (proxy for physical objects)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters at 0.5 m | 65 | insufficient (n=1) |
| … with at least one surviving object | 53 | insufficient (n=1) |
| … with more than one object | 53 | insufficient (n=1) |
| duplicate objects over the run (ids − clusters) | 589 | insufficient (n=1) |
| duplicate objects in the final memory | 253 | insufficient (n=1) |
| clusters at 0.25 m (sensitivity) | 136 | insufficient (n=1) |
| clusters at 0.5 m (sensitivity) | 65 | insufficient (n=1) |
| clusters at 1 m (sensitivity) | 24 | insufficient (n=1) |

A cluster groups objects whose median raw positions lie within the radius (leader clustering in creation order, no chaining). Two real objects closer than the radius fall into one cluster; a duplicate spawn farther than the radius is not seen. The sensitivity rows bound that.

## Detection over in-frustum views

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| views (cluster in the frustum on a processed frame) | 2916 | insufficient (n=1) |
| re-identified (a member matched) | 1437 | insufficient (n=1) |
| duplicated (a member created instead) | 259 | insufficient (n=1) |
| missed | 1220 | insufficient (n=1) |
| detection rate (re-identified + duplicated) | 58.2 % | insufficient (n=1) |
| re-identification rate | 49.3 % | insufficient (n=1) |
| masked: views | 898 | insufficient (n=1) |
| masked: re-identification rate | 47.4 % | insufficient (n=1) |
| masked: detection rate | 55.8 % | insufficient (n=1) |

By expected range (object level, run 1):

| range (m) | views | re-identified | rate | masked views | masked rate |
|---|---|---|---|---|---|
| 0.50-1.00 | 1 | 0 | 0.000 | 1 | 0.000 |
| 1.00-1.50 | 553 | 70 | 0.127 | 174 | 0.121 |
| 1.50-2.00 | 3110 | 431 | 0.139 | 890 | 0.138 |
| 2.00-2.50 | 6525 | 766 | 0.117 | 2082 | 0.111 |
| 2.50-3.00 | 5324 | 850 | 0.160 | 1714 | 0.149 |
| 3.00-3.50 | 1315 | 263 | 0.200 | 376 | 0.215 |
| 3.50-4.00 | 201 | 33 | 0.164 | 68 | 0.206 |
| 4.00-4.50 | 155 | 19 | 0.123 | 62 | 0.113 |
| 4.50-5.00 | 74 | 0 | 0.000 | 22 | 0.000 |
| 5.00-5.50 | 19 | 0 | 0.000 | 8 | 0.000 |
| 5.50-6.00 | 9 | 0 | 0.000 | 3 | 0.000 |

The frustum model is occlusion-agnostic (`v1_occlusion_agnostic`): an object behind another counts as a view, so every rate here is a lower bound on what the perception could see.

## Label disagreement

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 2 labelled observations | 247 | insufficient (n=1) |
| disagreement mean (1 − modal share) | 35.4 % | insufficient (n=1) |
| disagreement p50 | 41.5 % | insufficient (n=1) |
| disagreement p95 | 75.0 % | insufficient (n=1) |
| objects whose observations disagree | 74.5 % | insufficient (n=1) |
| clusters whose surviving objects carry different primary labels | 40 | insufficient (n=1) |

## Position scatter (raw observations vs the object median)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 3 observations | 185 | insufficient (n=1) |
| observations | 2570 | insufficient (n=1) |
| along-ray RMS (m) | 0.037 | insufficient (n=1) |
| lateral RMS (m) | 0.034 | insufficient (n=1) |
| along / lateral | 1.081 | insufficient (n=1) |
| |along| vs range: slope (m per m) | 0.010 | insufficient (n=1) |
| |along| vs range: intercept (m) | -0.002 | insufficient (n=1) |
| |along| vs range: R² | 0.032 | insufficient (n=1) |
| lateral vs range: slope (m per m) | 0.006 | insufficient (n=1) |

| range (m) | n | along RMS (m) | lateral RMS (m) |
|---|---|---|---|
| 1.00-1.50 | 12 | 0.071 | 0.035 |
| 1.50-2.00 | 388 | 0.028 | 0.024 |
| 2.00-2.50 | 640 | 0.039 | 0.033 |
| 2.50-3.00 | 1078 | 0.031 | 0.039 |
| 3.00-3.50 | 318 | 0.053 | 0.029 |
| 3.50-4.00 | 106 | 0.047 | 0.027 |
| 4.00-4.50 | 28 | 0.039 | 0.057 |

## Duplicate spawns

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects created within the radius of a live object | 617 | insufficient (n=1) |
| … as a fraction of created objects | 94.3 % | insufficient (n=1) |
| … while the original was in the frustum | 563 | insufficient (n=1) |
| reason: original not returned by the index | 3 | insufficient (n=1) |
| reason: nearby but failed the distance / z / reprojection gates | 388 | insufficient (n=1) |
| reason: passed the gates, cosine below 0.9 | 226 | insufficient (n=1) |
| reason: other | 0 | insufficient (n=1) |

Alive = a surviving object, or one observed less than 10 s earlier (the proto TTL). The reason classes read the created line's audit fields (`n_nearby`, `n_gate_survivors`, `max_cos`).

## Revisits

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters seen in ≥ 2 visits | 18 | insufficient (n=1) |
| revisits | 18 | insufficient (n=1) |
| re-identified on return | 16 | insufficient (n=1) |
| duplicated on return | 0 | insufficient (n=1) |
| missed on return | 2 | insufficient (n=1) |
| re-identification lower bound on revisits | 88.9 % | insufficient (n=1) |

A visit ends when the cluster leaves the frustum for ≥ 5 s of sensor time. Missed and duplicated revisits count as failures, so the bound is a lower bound (occlusion is a miss here).

## Worst moments (run 1)

| t (s) | frame | kind | score | detail |
|---|---|---|---|---|
| 24.7018 | 247 | frame | 22.000 | missed 6, duplicate_spawns 8, in_frustum 10, reidentified 4, gate_shadow skip |
| 31.8022 | 318 | frame | 22.000 | missed 12, duplicate_spawns 5, in_frustum 19, reidentified 6, gate_shadow skip |
| 11.5009 | 115 | frame | 21.000 | missed 5, duplicate_spawns 8, in_frustum 12, reidentified 4, gate_shadow skip |
| 11.3009 | 113 | frame | 20.000 | missed 4, duplicate_spawns 8, in_frustum 11, reidentified 5, gate_shadow skip |
| 33.9024 | 339 | frame | 20.000 | missed 10, duplicate_spawns 5, in_frustum 17, reidentified 5 |
| 34.0024 | 340 | frame | 20.000 | missed 10, duplicate_spawns 5, in_frustum 17, reidentified 6, gate_shadow skip |
| 23.8017 | 238 | frame | 19.000 | missed 3, duplicate_spawns 8, in_frustum 8, reidentified 2, gate_shadow skip |
| 31.9022 | 319 | frame | 19.000 | missed 11, duplicate_spawns 4, in_frustum 19, reidentified 7, gate_shadow skip |
| 34.4024 | 344 | frame | 19.000 | missed 9, duplicate_spawns 5, in_frustum 17, reidentified 4 |
| 35.3024 | 353 | frame | 19.000 | missed 9, duplicate_spawns 5, in_frustum 17, reidentified 7, gate_shadow skip |

Score = missed clusters + 2 × duplicate spawns (+ 1 when the frame-quality gate rejected the frame); pose gaps / discontinuities / tracking-limited episodes score by their size. `t` is sensor time since the first pose line.

## Largest clusters (run 1)

| cluster | objects (surviving) | observations | first–last seen (s) | re-id rate | visits | top labels | surviving labels |
|---|---|---|---|---|---|---|---|
| c0008 | 52 (34) | 324 | 2.6002–40.5194 | 0.742 | 2 | laptop (61), amplifier (25), sheet (17) | amplifier, cat bed, converter, crt screen, doormat, envelope, file cabinet, flip, fume hood, handbag, hassock, jack, knife, laptop, night light, pan, quilting, safe, sew, sheet, speaker, storage box |
| c0001 | 16 (14) | 250 | 0.0–40.5194 | 0.919 | 2 | pet toy (118), pillow (29), flower (27) | bread, flower, hassock, pet toy, pillow, pitcher plant, rag doll, tinsel |
| c0005 | 32 (16) | 184 | 0.0–37.5026 | 0.757 | 2 | folder (45), fax (22), book (21) | book, fax, folder, gift card, keycard, legend, note |
| c0026 | 30 (12) | 158 | 11.5009–20.0015 | 0.981 | 1 | paper towel (17), wall (11), beeper (10) | CD, beeper, bottle, edge, hardware, jack, knife, paper towel, tape, wall |
| c0025 | 24 (8) | 141 | 11.3009–22.6017 | 0.897 | 1 | printer (13), folder (12), whiteboard (9) | amplifier, fax, flip, folder, hardware, night light, pencil, storage box |
| c0000 | 10 (3) | 134 | 0.0–40.5194 | 0.895 | 2 | pet toy (104), flower (10), daisy (4) | carnation, figurine, hamster |
| c0028 | 25 (11) | 133 | 12.501–20.3015 | 0.902 | 1 | assemble (41), whiteboard (16), light switch (15) | assemble, figurine, file cabinet, keycard, matchbox, mattress, mirror, remove, socket |
| c0058 | 12 (12) | 123 | 28.502–36.3025 | 0.938 | 1 | portable air conditioner (37), fan (31), air conditioner (22) | air conditioner, fan, foam roller, heater, knife, window |
| c0052 | 16 (8) | 105 | 24.7018–32.8023 | 0.820 | 1 | laptop (21), remote (19), speaker (10) | camera lens, keyboard, laptop, remote, remove, zoom lens |
| c0036 | 20 (7) | 102 | 17.8013–24.1018 | 0.892 | 1 | blanket (23), diaper bag (20), pack (14) | blanket, diaper bag, knife, pack, sheet, shoe |
| c0012 | 24 (11) | 99 | 4.0003–40.5194 | 0.797 | 2 | lamp (36), mirror (10), pillow (8) | christmas ball, hammer, lamp, mirror, nightstand, pet toy, punch bag, webcam |
| c0042 | 18 (5) | 79 | 20.5015–29.0021 | 0.426 | 1 | bottle (19), shoe (12), bag (10) | contain, hoverboard, tool, tube |
| c0024 | 5 (3) | 73 | 10.3008–17.8013 | 0.848 | 1 | drawer (40), whiteboard (13), knife (6) | drawer, knife, whiteboard |
| c0039 | 31 (8) | 66 | 18.8014–34.4024 | 0.386 | 1 | knife (18), shoe (8), phone (4) | converter, eyeliner, knife, recliner, tool, wrench |
| c0003 | 47 (24) | 63 | 0.0–40.1194 | 0.231 | 2 | ray (16), dagger (7), sheet (6) | blanket, blue artist, catfish, dagger, flip, floor window, ray, sew, squeeze, sword, tube |

## Method notes

- Everything above is computed from the run's ledgers (`events.jsonl`: pose / obs / view lines, the frame-flow trace) and the final memory in `summary.json`. No ground truth, no labels required; label numbers use the detector's top-1 label per observation.
- Objects = every id the associator matched or created; a **transient** object was created and is not in the final memory (a proto that expired, or one the memory evicted).
- Raw observations (`p_world` as the associator computed it) are used everywhere; the memory's smoothed positions appear only in the fingerprint.
- Per-object, per-cluster, per-frame and per-duplicate records are in `metrics.json` (`objects`, `clusters`, `frames`, `duplicate_spawns`, `worst_moments`); each object and cluster carries the sensor stamps of the frames it was seen and missed on.
- Parameters: cluster radius 0.5 m, revisit gap 5 s, range bin 0.5 m, scatter needs ≥ 3 observations, proto TTL 10 s, cosine threshold 0.9.
