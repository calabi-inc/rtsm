# TUM fr3 / long_office_household — a handheld loop around two office desks

**The recording.** `rgbd_dataset_freiburg3_long_office_household.bag` from the TUM RGB-D benchmark: a handheld Kinect (640×480, 30 Hz) carried for 87 s around two office desks covered with household objects (books, bottles, cups, a keyboard, dolls), coming back to where it started; ground-truth poses from a motion-capture system published as a four-hop TF chain. 2012 data. Licence CC BY 4.0; citation: Sturm, Engelhard, Endres, Burgard, Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012. Fetch it from `https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.bag` (1.7 GB).

**The command** (released defaults, our `grounded_sam2` detector; one desktop GPU; three same-input repeats so every number carries its floor):

```bash
rtsm eval rgbd_dataset_freiburg3_long_office_household.bag --repeats 3
```

## What the reader decided

The bag needs nothing from you: RGB `/camera/rgb/image_color` (`bgr8`, 2 585 messages), depth `/camera/depth/image` (`32FC1`, NaN for invalid), the RGB `CameraInfo`, poses from `/tf` through `world → kinect → openni_camera → openni_rgb_frame → openni_rgb_optical_frame` (the mocap hop moves at 100 Hz, the calibration hops are republished constants), depth registered to the RGB (the two `CameraInfo` K matrices agree to 0.00 %). 2 488 of the 2 585 RGB frames found a depth frame within the pairing tolerance (the worst pair is 8 ms apart); the other 97 are the bag's own drops. 31 121 messages in all; the probe and the refusal checks run before any model loads.

## The numbers (run 1 of 3; the floor is the spread over the three runs)

Every floor on this page is 0: three runs gave the same fingerprint (`d9e820e281a3fdf4`) and the same ledgers.

| | |
|---|---|
| frames seen / admitted / processed | 2 488 / 254 / 194 (the deployed cadence: a keyframe every 30 frames, non-keyframes throttled to 0.5 s; 2 234 throttled, the sweep gate rejected 60) |
| processed cadence | a processed frame every 0.47 s (p50), 0.82 s (p95); longest unprocessed stretch 1.07 s over the 87 s span |
| pose stream | 28.5 Hz over 87.1 s, jitter 0.42 ms, 0 discontinuities; 53 gaps in the paired stream (the 97 RGB frames without a depth partner) |
| depth valid | 83.6 % of pixels (p50) — the Kinect's holes |
| objects at the end / confirmed | 277 / 237; 746 created over the run, 469 of them transient (expired protos); 30 survivors with a single hit |
| spatial clusters at 0.50 m | 49 (103 at 0.25 m, 24 at 1.0 m); 49 of 49 present in every run; 35 with a surviving object, 28 holding more than one; 242 duplicate objects in the final memory |
| **re-identification over in-frustum views** | **31.4 %** of 3 211 views (detection incl. duplicates 36.6 %; 2 037 views missed) |
| by range (object level) | 0.5–1.0 m 15.3 %, 1.0–1.5 m 14.7 %, 1.5–2.0 m 9.5 %, 2.0–2.5 m 5.6 %, 2.5–3.0 m 2.8 %, 3.0–3.5 m 0.7 % |
| duplicate spawns | 707 = **94.8 % of created objects**; 666 while the original was in the frustum; **647 failed the cosine gate (0.90)**, 59 the distance / reprojection gates, 1 not returned by the index |
| label disagreement | mean 15.0 %, p50 0 %, p95 50 % over 333 objects with ≥ 2 labelled observations (42.6 % of them disagree at least once); 20 clusters whose surviving objects carry different primary labels |
| position scatter | along-ray RMS 0.032 m, lateral 0.031 m; no range dependence worth the name (slope 0.008 m per m, R² 0.06) |
| **revisits** | 25 clusters seen in two or more visits, 41 returns; **22 re-identified on return (53.7 %)**, 1 duplicated, 18 missed |
| wall time | 206–216 s per run after the model load |

The largest cluster (on the bookshelf, 77 ids over the run, 31 surviving) was seen from the first to the last second over three visits and re-identified in 61 % of its views; its surviving objects carry eight different primary labels (book, box, desk, doll, person, shelf, stuffed animal, table). Full per-cluster, per-object and per-frame records are in the run's `metrics.json`.

## Reading it

- **The appearance gate is the binding constraint, as on [fr1](tum-fr1-desk.md).** 647 of the 707 duplicate spawns passed the geometric gates and failed on the CLIP cosine threshold. Positions are consistent (3 cm scatter either way) and every cluster is stable across runs; the memory sees the same places and fails to recognise the same things. Among duplicates, "the original was in the frustum" holds 94 % of the time, so these are not occlusion artefacts.
- **The loop is where the memory earns something.** The camera returns to the first desk and the shelf; of 41 returns to a place the memory already holds, 22 land on an existing object. That is a lower bound (occlusion counts as a miss, and a return that matched *any* object of a 77-id cluster counts once), and it is the number that was 0 of 4 on the 20-second fr1 sweep: longer sequences with real revisits are what this report is for.
- **Range is the second constraint.** Re-identification is 15 % under 1.5 m and 3 % at 2.5–3 m. Most views in this office are at 1.5–3 m, where a 640×480 Kinect crop of a cup is a few dozen pixels. The absolute rates describe that sensor as much as the memory.
- **A 0.5 m cluster is a place, not an object.** A shelf of books is one cluster at 0.5 m and several at 0.25 m (49 vs 103 clusters); the sensitivity rows bound how much of the "duplicate" count is neighbours, and the label list on the shelf cluster (book, doll, shelf, box …) says that cluster holds several real things.
- **Labels leak from the vocabulary.** `person` and `forklift` appear among the surviving labels of two office clusters. They come from the detector's open vocabulary, not from the memory; the report attributes them to `grounded_sam2` with the public indoor vocabulary, and a detector of your own replaces them ([Your own detector](../guides/eval.md#your-own-detector)).
- **The poses are not the problem.** Mocap ground truth, 0 discontinuities, 0.42 ms jitter: everything on this page is perception and association.

What this page is not: a benchmark of the 2012 Kinect, a claim about other detectors, or a tuned result — these are the packaged defaults at commit `6a61a61`, the code released as 0.2.0. The same command on your own bag gives the same table about your data.
