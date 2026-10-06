# Evaluating a Bag (`rtsm eval`)

`rtsm eval` runs the pipeline headlessly on a recording — a ROS 1 `.bag`, a rosbag2 directory, a bare `.mcap` or a Lens recording — and writes everything a report needs: the frame-flow trace, the pose / observation / view ledgers, the final memory, and the settings that produced them. Then it writes the report: `metrics.json` (every per-object, per-cluster and per-frame result as data) and `report.md` (the number sheet). No server, no dashboard, no shared state: each run gets its own directory and its own vector store, and the packaged configuration is never written.

```bash
pip install "rtsm[eval]"
rtsm eval recordings/session1_bag                       # our own layout, deployed settings, one run
rtsm eval recordings/session1 --repeats 3               # a Lens recording, three same-input runs -> every number carries its floor
rtsm eval rgbd_dataset_freiburg3_long_office_household.bag --mode dense --out eval_output/fr3
rtsm eval my_rosbag2 --mode every_frame --max-frames 500 --set io.bag.topics.depth=/camera/aligned_depth_to_color/image_raw
rtsm report eval_output/fr3                             # regenerate metrics.json + report.md from the run directories (no GPU)
```

## What a run does

1. Resolves the settings: the **sensor clock**, the **lossless lane** (nothing is dropped, the producer waits), the keyframe rule and the non-keyframe throttle for the chosen mode, the gate mode, and records the config fingerprint, the commit and the versions in `resolved.json`.
2. Probes a bag first (`bag_probe.json`): topics, pose source, registration. A bag v1 cannot read is refused with the reason before any model loads (see [Reading Bags](bags.md)).
3. Loads the models once, then for each repeat builds a fresh runtime — memory, index, gate, sweep cache, queue, pipeline, event log — through the same factory the live runner uses (`rtsm/engine.py`).
4. Feeds the input through the bag source (or the replay source) as fast as the queue accepts, runs the pipeline until the source is done and the queue is empty, force-flushes every object to the run's vector store, and writes `summary.json`.
5. `repeats.json` collects the per-run fingerprints and counts; `identical_fingerprints` says whether N runs agreed.
6. Writes the report over the run directories (`metrics.json` + `report.md`, below). `--no-report` skips it; `rtsm report <dir>` writes it later.

## Modes (cadences)

| | `as_deployed` (default) | `dense` | `every_frame` |
|---|---|---|---|
| cadence | **deployed** | **representative** | **exhaustive** |
| keyframes | every `ingest.keyframe_every_n` admitted frames (the source's own keyframes when it flags them) | every `eval.keyframe_interval_s` of sensor time (1.0 s) | as `dense` |
| non-keyframe throttle | `ingest.nonkf_min_interval_s` | `1 / eval.process_rate_hz` (5 Hz → 0.2 s) | **none** — every frame the input contains is admitted |
| sweep gate | **enforced** | **shadow**: the decision is logged as `gate_shadow` on the dequeue line, the frame is processed anyway | shadow |
| what it answers | what the deployed system would have remembered — the determinism anchors are defined on it | what the perception could have seen at a steady cadence | whether a detector failure hides behind the throttle: the only frames not processed are the ones the frame-quality gate rejects or whose tracking is not normal, both logged with a reason |
| cost | 53 frames on session1 | 195 frames on session1 | every frame: a 5-minute 30 Hz bag is ~9 000 frames, about 30 min per repeat on a desktop GPU with the dual backend |

`as_deployed` is exactly what `python -m rtsm --replay` / `--bag` does, so a run on `recordings/session1` reproduces the G1-B anchor (124 objects / 65 confirmed at 53 processed frames, fingerprint `ad6f71a5b89c8506`). Under `dense` and `every_frame` every view-based number in the report is also given **masked** — restricted to the frames the deployed gate would have admitted — so the observation metrics can be read at either cadence. The keyframe rule under `every_frame` stays the interval rule on purpose: the memory's position update trusts keyframes far more than other frames, so making every frame a keyframe (or none) would change what the memory does, not only how often it looks. The report always states the cadence that produced its numbers.

## The run directory

```
eval_output/<input>-<mode>-<stamp>/
  resolved.json      the resolved settings, cadence, config fingerprint, commit (+ dirty flag and diff digest), versions, the model files the config names (size, sha256, or exists: false) and the hub model ids
  bag_probe.json     (bags) topics, pose source, registration
  repeats.json       fingerprints / counts / wall time per run, identical_fingerprints
  metrics.json       every metric per run + the aggregate with floors; per-object / per-cluster / per-frame records
  report.md          the number sheet
  run_1/
    events.jsonl     the frame-flow trace + the pose / obs / view ledgers (schema 3, ledgers schema 1)
    summary.json     counts (receiver decisions, dequeue outcomes, gate shadows), the memory (every object in the /objects shape + the fingerprint), latency and segmentation aggregates, the source's statistics
    faiss/           this run's vector store
    crops/           (--save-crops) the memory's per-object JPEG snapshots, <id>/<k>.jpg + index.json
  run_2/ …
```

`summary.json.memory.fingerprint` is the sha256 of the sorted multiset of `(label_primary, xyz rounded to 3 decimals, hits, confirmed)` — the same definition the determinism gates use (the session1 anchor above).

## The report

Every number is computed per run from the ledgers alone (`rtsm/evaluation/metrics.py`), then aggregated over the N same-input repeats (`rtsm/evaluation/report.py`). The **floor** of a number is its spread (max − min) across the runs: what the system shows on identical input, below which a difference between two bags or two versions means nothing. Fewer than three runs cannot establish a floor and the report says so at the top. The report is a pure function of the run directories — no wall-clock stamps — so `rtsm report` reproduces it byte for byte. No ground truth and no labels are needed; the label numbers use each observation's top-1 scored label (the detector's label first; under `segmentation.labels.prompt_free_primary: classifier` a prompt-free model's name stays out and the CLIP vocabulary classifier's label leads).

| section | what it measures | how | caveat |
|---|---|---|---|
| Frames and admission | how many frames the input had, how many the receiver admitted, throttled or dropped, how many the pipeline processed or rejected, the sensor-time gap between processed frames | the `receiver` / `dequeue` lines | under shadow: how many processed frames the deployed gate would have rejected |
| Pose stream | rate, span, jitter, gaps, tracking-limited episodes, discontinuities (a step larger than 0.5 m + 1 m/s · dt), depth-valid and confidence-2 fractions | `pose_health` over the `pose` ledger | the discontinuity rule is a detector, nothing acts on it |
| Memory | objects and confirmed at the end, objects **created** over the run, **transient** objects (created but not in the final memory: an expired proto or an evicted object), survivors with a single hit, matches the associator made without scoring | the `obs` ledger joined to `summary.json.memory` | the fingerprint covers the survivors only |
| Spatial clusters | groups of objects whose median raw positions lie within the cluster radius — the proxy for one physical object; duplicates = objects − clusters, over the run and in the final memory; counts at ½×, 1× and 2× the radius | leader clustering in creation order: a track joins the nearest existing leader within the radius, else starts a cluster; deterministic, no chaining | two real objects closer than the radius fall into one cluster; a duplicate farther than the radius is not seen; the default radius is the associator's own distance gate (`assoc.gate_dist_base_m`) so a duplicate inside it is one the associator could have matched by position |
| Detection over in-frustum views | for every processed frame whose `view` line lists a cluster member: **re-identified** (a member matched), **duplicated** (a member created instead — a segment was found there, the memory did not recognise it), **missed**; the detection rate (re-identified + duplicated) and the re-identification rate; by expected range | the `view` ledger joined to the `obs` ledger on the frame stamp | the frustum is occlusion-agnostic (`v1_occlusion_agnostic`): an object behind another counts as a view, so every rate is a lower bound |
| Label disagreement | per object `1 − modal share` of its top-1 labels, the distribution over objects with ≥ 2 observations, clusters whose surviving objects carry different primary labels | `label_topk[0]` per obs line (the detector's label first, or the vocabulary classifier's under `prompt_free_primary: classifier`; `detector_label` keeps the detector's own name) | vocabulary noise counts as disagreement |
| Position scatter | per object with ≥ 3 observations, the residual of each raw observation to the object's median, split into the component **along** the camera-to-object ray (depth) and the **lateral** remainder; pooled RMS, per range bin, and a least-squares line of \|along\| against range | `p_world` and `cam_t_wc` per obs line | the median is the reference, not a ground-truth position |
| Duplicate spawns | a `created` line within the cluster radius of an earlier object that was **alive** at that stamp (a survivor, or observed less than `object.proto_ttl_s` earlier); how many happened while the original was in the frustum; the reason class from the created line's audit fields: `not_in_index` (the proximity index returned nothing), `gated` (nearby but no candidate passed the distance / z / reprojection gates), `low_similarity` (a candidate passed the gates, the best cosine was below `assoc.cos_min`), `other` | the `obs` ledger | alive is approximated from the ledgers (no removal line exists) |
| Revisits | the presence frames of a cluster split into visits at gaps ≥ `eval.metrics.revisit_gap_s` of sensor time; every visit after the first is a revisit: **re-identified** (a member matched during it), **duplicated** (a member created and none matched), **missed**; the re-identification **lower bound** = re-identified / revisits | the `view` + `obs` ledgers | missed and duplicated revisits both count as failures — an occluded object is a miss here |
| Worst moments | processed frames scored by misses + 2 × duplicate spawns (+ 1 when the frame-quality gate rejected the frame), plus pose gaps, discontinuities and tracking-limited episodes, each with the sensor stamp and the seconds since the first pose line | all of the above | scrub to the stamp in your bag viewer |
| Largest clusters | the clusters with the most observations: members, survivors, first/last seen, re-identification rate, visits, top labels, surviving labels | | |

`metrics.json` holds, per run, `scalars` (every number above, flat), `clusters`, `objects`, `frames`, `duplicate_spawns`, `worst_moments` (each object and cluster carries the stamps of the frames it was seen and missed on — the object-anchored moment retrieval), and under `aggregate` the floors plus the clusters of run 1 matched across the other runs (`clusters_across_runs`: how many are present in every run). Comparing against a *stored* baseline — another bag, another model version, another site — is not part of the free report.

Parameters live under `eval.metrics` in the configuration: `cluster_radius_m` (null = `assoc.gate_dist_base_m`), `cluster_radii_m` (the sensitivity radii), `revisit_gap_s`, `range_bin_m`, `min_obs_scatter`, `worst_n`, `moments_cap`. Override them like any other key (`--set eval.metrics.cluster_radius_m=0.3`), for `rtsm eval` and `rtsm report` alike.

## Your own detector

If the bag already carries a detector's output as a `vision_msgs/msg/Detection2DArray` (or `Detection3DArray`) topic, RTSM can run on *those* detections instead of its own: the memory, the ledgers and the report are then about your detector.

```bash
rtsm eval my_bag --set segmentation.backend=external                       # the detections topic is auto-discovered
rtsm eval my_bag --set segmentation.backend=external --set io.bag.topics.detections=/perception/detections
```

How it works: the bag reader pairs each RGB frame with the detections message nearest in stamp (within `io.bag.pair_tolerance_s`; detectors publish after the image, so a frame waits up to a second for its message), the `external` backend turns the boxes into instance masks (the pixels within `segmentation.external.depth_band_m` of the box's median depth, or the box itself without depth), labels are the top hypothesis of each detection, CLIP embeddings are computed by RTSM from the crops exactly as for its own detectors, and everything downstream is unchanged. The `vision_msgs` definitions ship with RTSM (both the ROS 2 and the ROS 1 layouts), so a bag without embedded message definitions still reads. 3-D boxes are projected into the image through the bag's TF at the stamp and the camera intrinsics.

Scores: a detector that reports no confidence (the 2026 VLM detectors, for instance) is a first-class case. Its labels are stored with `segmentation.external.unscored_label_prior`, the observation ledger records `score: null` for them, and the report header says `scores: absent` so no label-confidence number is read as measured. The header always names the detector the numbers came from: our backend, or the topic and message type, with how many frames had detections and how many messages matched no frame.

Not supported in v1: masks from the detector (boxes only; a mask-refiner slot exists but only `none` is implemented), tracks (ids are carried, not used), proprietary formats (write an adapter against `rtsm.io.detections.DetectionsAdapter` and register it). `scripts/export_detections.py` writes a bag with RTSM's own detections as such a topic, which is how the path is gated and a demonstration of the format.

## Options

`--mode`, `--repeats`, `--out`, `--label`, `--max-frames` (bag inputs), `--max-wall-s` (abort a run and record it), `--no-report`, `--save-crops` (write each object's JPEG snapshots into `run_N/crops/`, for report renderers), plus the usual `--config` / `--profile` / `--set` overrides. Defaults live in the `eval:` block of the configuration.

## For tooling built on the run directory

Other tools, including Calabi's paid compare, build on three things that are versioned and changelogged: the run-directory layout above, the ledger schema (1) and the `metrics_schema` in `metrics.json`, and the modules `rtsm.evaluation.metrics` (`MetricParams`, `compute_metrics`, `object_tracks`, `leader_clusters`), `rtsm.evaluation.report` (`run_dirs`, `run_metrics`, `aggregate`, `render_markdown`, `write_report`) and `rtsm.evaluation.ledger` (`read_events`, `by_kind`). A computation a tool needs that is missing here belongs in these modules, not in the tool.
