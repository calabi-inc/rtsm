# Evaluating a Bag (`rtsm eval`)

`rtsm eval` runs the pipeline headlessly on a recording — a ROS 1 `.bag`, a rosbag2 directory, a bare `.mcap` or a Lens recording — and writes everything a report needs: the frame-flow trace, the pose / observation / view ledgers, the final memory, and the settings that produced them. No server, no dashboard, no shared state: each run gets its own directory and its own vector store, and the packaged configuration is never written.

```bash
pip install "rtsm[eval]"
rtsm eval recordings/session1_bag                       # our own layout, deployed settings
rtsm eval recordings/session1 --repeats 3               # a Lens recording, three same-input runs
rtsm eval rgbd_dataset_freiburg3_long_office_household.bag --mode dense --out eval_output/fr3
rtsm eval my_rosbag2 --set io.bag.topics.depth=/camera/aligned_depth_to_color/image_raw --max-frames 500
```

## What a run does

1. Resolves the settings: the **sensor clock**, the **lossless lane** (nothing is dropped, the producer waits), the keyframe rule and the non-keyframe throttle for the chosen mode, the gate mode, and records the config fingerprint, the commit and the versions in `resolved.json`.
2. Probes a bag first (`bag_probe.json`): topics, pose source, registration. A bag v1 cannot read is refused with the reason before any model loads (see [Reading Bags](bags.md)).
3. Loads the models once, then for each repeat builds a fresh runtime — memory, index, gate, sweep cache, queue, pipeline, event log — through the same factory the live runner uses (`rtsm/engine.py`).
4. Feeds the input through the bag source (or the replay source) as fast as the queue accepts, runs the pipeline until the source is done and the queue is empty, force-flushes every object to the run's vector store, and writes `summary.json`.
5. `repeats.json` collects the per-run fingerprints and counts; `identical_fingerprints` says whether N runs agreed (the report turns this into the self-consistency floor).

## Modes

| | `as_deployed` (default) | `dense` |
|---|---|---|
| keyframes | every `ingest.keyframe_every_n` admitted frames (the source's own keyframes when it flags them) | every `eval.keyframe_interval_s` of sensor time (1.0 s) |
| non-keyframe throttle | `ingest.nonkf_min_interval_s` | `1 / eval.process_rate_hz` (5 Hz → 0.2 s) |
| sweep gate | **enforced** | **shadow**: the decision is logged as `gate_shadow` on the dequeue line, the frame is processed anyway |
| what it answers | what the deployed system would have remembered — the determinism anchors are defined on it | what the perception could have seen at a steady cadence; observation metrics are masked afterwards by what the gate would have admitted |

`as_deployed` is exactly what `python -m rtsm --replay` / `--bag` does, so a run on `recordings/session1` reproduces the G1-B anchor (124 objects / 65 confirmed at 53 processed frames, fingerprint `ad6f71a5b89c8506`).

## The run directory

```
eval_output/<input>-<mode>-<stamp>/
  resolved.json      the resolved settings, config fingerprint, commit, versions
  bag_probe.json     (bags) topics, pose source, registration
  repeats.json       fingerprints / counts / wall time per run, identical_fingerprints
  run_1/
    events.jsonl     the frame-flow trace + the pose / obs / view ledgers (schema 3, ledgers schema 1)
    summary.json     counts (receiver decisions, dequeue outcomes, gate shadows), the memory (every object in the /objects shape + the fingerprint), latency and segmentation aggregates, the source's statistics
    faiss/           this run's vector store
  run_2/ …
```

`summary.json.memory.fingerprint` is the sha256 of the sorted multiset of `(label_primary, xyz rounded to 3 decimals, hits, confirmed)` — the same definition every gate record in `eval/baselines/` uses.

## Options

`--mode`, `--repeats`, `--out`, `--label`, `--max-frames` (bag inputs), `--max-wall-s` (abort a run and record it), plus the usual `--config` / `--profile` / `--set` overrides. Defaults live in the `eval:` block of the configuration.
