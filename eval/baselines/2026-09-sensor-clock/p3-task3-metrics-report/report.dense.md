# rtsm eval report — `session1`

> **Floor not established: 1 run(s).** Every number below needs at least 3 same-input runs to carry a floor (`rtsm eval … --repeats 3`). Read the values as one sample.

## Run

| | |
|---|---|
| input | `recordings/session1` (replay) |
| mode / cadence | `dense` — **representative (dense)** |
| keyframe rule | every 1.0 s of sensor time |
| non-keyframe throttle | 0.2 s on the sensor clock |
| sweep gate | shadow (shadow: every processed frame also reported restricted to the frames the deployed gate would have admitted — *masked*) |
| ingest policy | lossless |
| repeats | 1 — fingerprints identical |
| config fingerprint | `e57e33b0b91da0c91579eab027b3c5feb0193bfd48f2b1b9ed2dcf4cee2e5908` |
| commit / rtsm / python | `8278fadfa445` / None / 3.12.10 |
| cluster radius | 0.5 m (the associator's distance gate); sensitivity at 0.25, 0.5, 1 m |
| wall time per run | 47.0 s |

The floor of a number is its spread over the 1 same-input runs: a difference between two bags or two versions smaller than the floor means nothing.

## Frames and admission

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| sensor frames seen | 240 | insufficient (n=1) |
| admitted (enqueued) | 195 | insufficient (n=1) |
| throttled | 45 | insufficient (n=1) |
| throttled fraction | 18.8 % | insufficient (n=1) |
| dropped: tracking not normal | 0 | insufficient (n=1) |
| processed | 195 | insufficient (n=1) |
| keyframes processed | 39 | insufficient (n=1) |
| sweep gate rejected | 0 | insufficient (n=1) |
| frame-quality rejected | 0 | insufficient (n=1) |
| gap between processed frames p50 (s) | 0.200 | insufficient (n=1) |
| gap between processed frames p95 (s) | 0.300 | insufficient (n=1) |
| longest unprocessed stretch (s) | 0.300 | insufficient (n=1) |
| processed span (s) | 40.519 | insufficient (n=1) |
| frames the deployed gate would have rejected | 126 | insufficient (n=1) |
| … fraction of processed | 64.6 % | insufficient (n=1) |

Dequeue outcomes (run 1): `processed:hard_max_s` 30, `processed:keyframe` 39, `processed:near_recent_keyframe` 12, `processed:skip` 114.

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
| objects at the end | 249 | insufficient (n=1) |
| confirmed | 154 | insufficient (n=1) |
| objects created over the run | 564 | insufficient (n=1) |
| transient (created, not in the final memory) | 315 | insufficient (n=1) |
| survivors with a single hit | 88 | insufficient (n=1) |
| matches made without scoring (associator fallback) | 0 | insufficient (n=1) |

Fingerprint (run 1): `f9a81c3936d3e0db`. Hits histogram: 1: 88, 2: 7, 3-5: 51, 6-10: 36, 11+: 67. View bins per object: 1: 226, 2: 23.

## Spatial clusters (proxy for physical objects)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters at 0.5 m | 64 | insufficient (n=1) |
| … with at least one surviving object | 52 | insufficient (n=1) |
| … with more than one object | 53 | insufficient (n=1) |
| duplicate objects over the run (ids − clusters) | 500 | insufficient (n=1) |
| duplicate objects in the final memory | 197 | insufficient (n=1) |
| clusters at 0.25 m (sensitivity) | 130 | insufficient (n=1) |
| clusters at 0.5 m (sensitivity) | 64 | insufficient (n=1) |
| clusters at 1 m (sensitivity) | 22 | insufficient (n=1) |

A cluster groups objects whose median raw positions lie within the radius (leader clustering in creation order, no chaining). Two real objects closer than the radius fall into one cluster; a duplicate spawn farther than the radius is not seen. The sensitivity rows bound that.

## Detection over in-frustum views

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| views (cluster in the frustum on a processed frame) | 2294 | insufficient (n=1) |
| re-identified (a member matched) | 1136 | insufficient (n=1) |
| duplicated (a member created instead) | 221 | insufficient (n=1) |
| missed | 937 | insufficient (n=1) |
| detection rate (re-identified + duplicated) | 59.2 % | insufficient (n=1) |
| re-identification rate | 49.5 % | insufficient (n=1) |
| masked: views | 832 | insufficient (n=1) |
| masked: re-identification rate | 48.3 % | insufficient (n=1) |
| masked: detection rate | 57.1 % | insufficient (n=1) |

By expected range (object level, run 1):

| range (m) | views | re-identified | rate | masked views | masked rate |
|---|---|---|---|---|---|
| 0.50-1.00 | 1 | 0 | 0.000 | 1 | 0.000 |
| 1.00-1.50 | 401 | 55 | 0.137 | 145 | 0.131 |
| 1.50-2.00 | 2257 | 356 | 0.158 | 760 | 0.158 |
| 2.00-2.50 | 4700 | 592 | 0.126 | 1761 | 0.121 |
| 2.50-3.00 | 3547 | 673 | 0.190 | 1347 | 0.183 |
| 3.00-3.50 | 947 | 204 | 0.215 | 321 | 0.234 |
| 3.50-4.00 | 170 | 27 | 0.159 | 62 | 0.210 |
| 4.00-4.50 | 113 | 13 | 0.115 | 56 | 0.125 |
| 4.50-5.00 | 46 | 0 | 0.000 | 18 | 0.000 |

The frustum model is occlusion-agnostic (`v1_occlusion_agnostic`): an object behind another counts as a view, so every rate here is a lower bound on what the perception could see.

## Label disagreement

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 2 labelled observations | 203 | insufficient (n=1) |
| disagreement mean (1 − modal share) | 37.5 % | insufficient (n=1) |
| disagreement p50 | 43.8 % | insufficient (n=1) |
| disagreement p95 | 77.6 % | insufficient (n=1) |
| objects whose observations disagree | 77.8 % | insufficient (n=1) |
| clusters whose surviving objects carry different primary labels | 37 | insufficient (n=1) |

## Position scatter (raw observations vs the object median)

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects with ≥ 3 observations | 154 | insufficient (n=1) |
| observations | 2032 | insufficient (n=1) |
| along-ray RMS (m) | 0.038 | insufficient (n=1) |
| lateral RMS (m) | 0.035 | insufficient (n=1) |
| along / lateral | 1.079 | insufficient (n=1) |
| |along| vs range: slope (m per m) | 0.010 | insufficient (n=1) |
| |along| vs range: intercept (m) | -0.001 | insufficient (n=1) |
| |along| vs range: R² | 0.029 | insufficient (n=1) |
| lateral vs range: slope (m per m) | 0.006 | insufficient (n=1) |

| range (m) | n | along RMS (m) | lateral RMS (m) |
|---|---|---|---|
| 1.00-1.50 | 10 | 0.075 | 0.034 |
| 1.50-2.00 | 317 | 0.028 | 0.024 |
| 2.00-2.50 | 500 | 0.041 | 0.035 |
| 2.50-3.00 | 855 | 0.030 | 0.038 |
| 3.00-3.50 | 246 | 0.053 | 0.030 |
| 3.50-4.00 | 83 | 0.053 | 0.044 |
| 4.00-4.50 | 21 | 0.036 | 0.053 |

## Duplicate spawns

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| objects created within the radius of a live object | 525 | insufficient (n=1) |
| … as a fraction of created objects | 93.1 % | insufficient (n=1) |
| … while the original was in the frustum | 469 | insufficient (n=1) |
| reason: original not returned by the index | 4 | insufficient (n=1) |
| reason: nearby but failed the distance / z / reprojection gates | 336 | insufficient (n=1) |
| reason: passed the gates, cosine below 0.9 | 185 | insufficient (n=1) |
| reason: other | 0 | insufficient (n=1) |

Alive = a surviving object, or one observed less than 10 s earlier (the proto TTL). The reason classes read the created line's audit fields (`n_nearby`, `n_gate_survivors`, `max_cos`).

## Revisits

| metric | value (run 1) | floor (spread over runs) |
|---|---|---|
| clusters seen in ≥ 2 visits | 17 | insufficient (n=1) |
| revisits | 17 | insufficient (n=1) |
| re-identified on return | 15 | insufficient (n=1) |
| duplicated on return | 1 | insufficient (n=1) |
| missed on return | 1 | insufficient (n=1) |
| re-identification lower bound on revisits | 88.2 % | insufficient (n=1) |

A visit ends when the cluster leaves the frustum for ≥ 5 s of sensor time. Missed and duplicated revisits count as failures, so the bound is a lower bound (occlusion is a miss here).

## Worst moments (run 1)

| t (s) | frame | kind | score | detail |
|---|---|---|---|---|
| 24.7018 | 247 | frame | 22.000 | missed 6, duplicate_spawns 8, in_frustum 10, reidentified 4, gate_shadow skip |
| 31.8022 | 318 | frame | 22.000 | missed 12, duplicate_spawns 5, in_frustum 19, reidentified 6, gate_shadow skip |
| 11.3009 | 113 | frame | 20.000 | missed 4, duplicate_spawns 8, in_frustum 11, reidentified 5, gate_shadow skip |
| 33.9024 | 339 | frame | 20.000 | missed 10, duplicate_spawns 5, in_frustum 17, reidentified 5 |
| 11.5009 | 115 | frame | 19.000 | missed 5, duplicate_spawns 7, in_frustum 12, reidentified 4, gate_shadow skip |
| 23.8017 | 238 | frame | 19.000 | missed 3, duplicate_spawns 8, in_frustum 8, reidentified 2, gate_shadow skip |
| 24.9018 | 249 | frame | 19.000 | missed 5, duplicate_spawns 7, in_frustum 10, reidentified 3, gate_shadow skip |
| 34.4024 | 344 | frame | 19.000 | missed 9, duplicate_spawns 5, in_frustum 17, reidentified 4 |
| 35.3024 | 353 | frame | 19.000 | missed 9, duplicate_spawns 5, in_frustum 17, reidentified 7, gate_shadow skip |
| 20.2015 | 202 | frame | 18.000 | missed 8, duplicate_spawns 5, in_frustum 15, reidentified 5, gate_shadow skip |

Score = missed clusters + 2 × duplicate spawns (+ 1 when the frame-quality gate rejected the frame); pose gaps / discontinuities / tracking-limited episodes score by their size. `t` is sensor time since the first pose line.

## Largest clusters (run 1)

| cluster | objects (surviving) | observations | first–last seen (s) | re-id rate | visits | top labels | surviving labels |
|---|---|---|---|---|---|---|---|
| c0008 | 49 (31) | 269 | 2.6002–40.5194 | 0.736 | 2 | laptop (48), amplifier (22), sheet (15) | amplifier, cat bed, converter, crt screen, doormat, edge, envelope, file cabinet, flip, fume hood, handbag, hassock, jack, laptop, night light, quilting, safe, sew, sheet, speaker, storage box |
| c0001 | 16 (13) | 184 | 0.0–40.5194 | 0.899 | 2 | pet toy (90), pillow (27), flower (13) | christmas decoration, flower, hassock, pet toy, pillow, pitcher plant, rag doll |
| c0005 | 22 (8) | 150 | 0.0–37.5026 | 0.808 | 2 | folder (38), fax (18), book (14) | book, folder, gift card, keycard, legend, note, passport |
| c0026 | 23 (9) | 120 | 11.5009–20.0015 | 0.974 | 1 | paper towel (14), wall (8), beeper (7) | CD, beeper, edge, hardware, jack, knife, paper towel, tape, wall |
| c0000 | 9 (4) | 119 | 0.0–40.5194 | 0.912 | 2 | pet toy (81), flower (17), daisy (5) | carnation, daisy, figurine, flower |
| c0025 | 25 (6) | 109 | 11.3009–22.5017 | 0.849 | 1 | printer (11), folder (10), heater (9) | amplifier, fax, folder, hardware, night light, remote |
| c0028 | 23 (9) | 100 | 12.501–20.2015 | 0.895 | 1 | assemble (29), whiteboard (12), light switch (10) | assemble, figurine, file cabinet, keycard, matchbox, mirror, shoe |
| c0057 | 8 (8) | 99 | 28.502–36.3025 | 0.925 | 1 | portable air conditioner (30), fan (26), air conditioner (17) | air conditioner, fan, foam roller, window |
| c0051 | 18 (7) | 90 | 24.7018–32.8023 | 0.810 | 1 | laptop (18), remote (16), speaker (10) | camera lens, clip art, laptop, remote, tape, zoom lens |
| c0035 | 16 (6) | 85 | 17.8013–24.1018 | 0.903 | 1 | blanket (19), diaper bag (17), pack (13) | blanket, diaper bag, knife, pack, shoe |
| c0012 | 20 (9) | 79 | 4.0003–40.5194 | 0.808 | 2 | lamp (28), pillow (7), mirror (7) | hammer, lamp, nightstand, pillow, punch bag, webcam |
| c0041 | 17 (5) | 70 | 20.5015–29.5021 | 0.431 | 1 | bottle (16), shoe (9), bag (8) | hoverboard, tea bag, tool, tube |
| c0038 | 32 (6) | 56 | 18.8014–34.4024 | 0.307 | 1 | knife (16), shoe (7), phone (4) | converter, knife, tool, wrench |
| c0024 | 6 (4) | 53 | 10.3008–17.8013 | 0.824 | 1 | drawer (26), whiteboard (10), knife (5) | drawer, knife, whiteboard |
| c0053 | 7 (6) | 53 | 26.7019–34.2024 | 0.895 | 1 | ipad (18), laptop (13), television (5) | ipad, laptop, smartphone |

## Method notes

- Everything above is computed from the run's ledgers (`events.jsonl`: pose / obs / view lines, the frame-flow trace) and the final memory in `summary.json`. No ground truth, no labels required; label numbers use the detector's top-1 label per observation.
- Objects = every id the associator matched or created; a **transient** object was created and is not in the final memory (a proto that expired, or one the memory evicted).
- Raw observations (`p_world` as the associator computed it) are used everywhere; the memory's smoothed positions appear only in the fingerprint.
- Per-object, per-cluster, per-frame and per-duplicate records are in `metrics.json` (`objects`, `clusters`, `frames`, `duplicate_spawns`, `worst_moments`); each object and cluster carries the sensor stamps of the frames it was seen and missed on.
- Parameters: cluster radius 0.5 m, revisit gap 5 s, range bin 0.5 m, scatter needs ≥ 3 observations, proto TTL 10 s, cosine threshold 0.9.
