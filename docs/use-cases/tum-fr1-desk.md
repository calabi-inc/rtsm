# TUM fr1 / desk — a short handheld sweep over a desk

**The recording.** `rgbd_dataset_freiburg1_desk.bag` from the TUM RGB-D benchmark: a handheld Kinect (640×480, 30 Hz) swept over an office desk for 20 s, ground-truth poses from a motion-capture system published as a four-hop TF chain. 2012 data. Licence CC BY 4.0; citation: Sturm, Engelhard, Endres, Burgard, Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012. Fetch it from `https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.bag` (390 MB).

**The command** (released defaults, our `grounded_sam2` detector; one desktop GPU; three same-input repeats so every number carries its floor):

```bash
rtsm eval rgbd_dataset_freiburg1_desk.bag --repeats 3
```

## What the reader decided

The bag needs nothing from you: RGB `/camera/rgb/image_color` (613 messages), depth `/camera/depth/image` (`32FC1`, NaN for invalid), the RGB `CameraInfo`, poses from `/tf` through `world → kinect → openni_camera → openni_rgb_frame → openni_rgb_optical_frame` (the mocap hop moves, the calibration hops are republished constants), depth registered to the RGB (the two `CameraInfo` K matrices agree). 586 of the 613 RGB frames found a depth frame within the 20 ms pairing tolerance; the rest are the bag's own drops.

## The numbers (run 1 of 3; the floor is the spread over the three runs)

Every floor on this page is 0: three runs gave the same fingerprint (`09380c457a4ba091`) and the same ledgers.

| | |
|---|---|
| frames seen / admitted / processed | 586 / 59 / 45 (the deployed cadence: a keyframe every 30 frames, non-keyframes throttled to 0.5 s; the sweep gate rejected 14) |
| pose stream | 29.5 Hz over 19.8 s, jitter 0.17 ms, 0 discontinuities; 9 gaps in the paired stream (27 RGB frames of the bag have no depth partner within the tolerance) |
| depth valid | 74.8 % of pixels (p50) — the Kinect's holes |
| objects at the end / confirmed | 217 / 49; 366 created over the run, 149 of them transient (expired protos) |
| spatial clusters at 0.50 m | 24 (54 at 0.25 m, 8 at 1.0 m); 24 of 24 present in every run |
| **re-identification over in-frustum views** | **24.7 %** of 401 views (detection incl. duplicates 36.9 %) |
| by range | 0.5–1.0 m 9.7 %, 1.0–1.5 m 7.4 %, beyond 1.5 m under 2 % (object level) |
| duplicate spawns | 349 = **95.4 % of created objects**; 306 while the original was in the frustum; **296 failed the cosine gate (0.90)**, 51 the distance / reprojection gates, 2 not returned by the index |
| label disagreement | mean 14.2 %, p50 0 %, p95 50 % over 92 objects with ≥ 2 labelled observations; 13 clusters whose surviving objects carry different primary labels |
| position scatter | along-ray RMS 0.024 m, lateral 0.018 m; no range dependence worth the name (slope 0.014 m per m, R² 0.07) |
| revisits | 4 clusters seen in two visits; **0 re-identified on return** |
| wall time | 48–50 s per run after the model load |

## Reading it

- **The appearance gate is the binding constraint on this sensor.** 296 of the 349 duplicate spawns passed the geometric gates and failed on the CLIP cosine threshold: the same desk object seen twice from a 640×480 Kinect crop does not embed within 0.90 of itself. Positions are consistent (2 cm scatter) and the clusters are stable across runs; the memory sees the same places, it does not recognise the same things. On a 2026 phone camera (our session1, 1920×1440) the same code re-identifies 55 % of views with 86 % duplicates — better, and still the dominant failure. That is the number the report exists to put in front of you.
- **Revisits are the hard case.** Four objects left the frustum and came back; none was recognised. On a 20-second sweep the revisits are seconds apart, so this is appearance, not drift. The 87-second [fr3 loop](tum-fr3-long-office.md), with 41 real returns, re-identifies 22 of them.
- **Range matters more than it should.** Re-identification falls from 10 % under a metre to about 1 % past 1.5 m: at 640×480, an object at 2 m is a few dozen pixels of crop.
- **The poses are not the problem.** Mocap ground truth, 0 discontinuities, 0.17 ms jitter: everything on this page is perception and association.

What this page is not: a benchmark of the 2012 Kinect, a claim about other detectors (the report attributes every number to `grounded_sam2` with the public indoor vocabulary), or a tuned result — these are the packaged defaults. The same command on your own bag gives the same table about your data; a detections topic of your own detector gives it about that detector ([Your own detector](../guides/eval.md#your-own-detector)).
