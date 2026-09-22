# Diagnostics & Ledgers

RTSM can write one append-only JSONL file per run describing what happened to every frame. It is off by default and costs nothing when off. Two things live in that file:

- the **frame-flow trace** (`receiver`, `dequeue`, `frame` lines): where each frame went and why — the record the determinism gates compare between runs;
- the **ledgers** (`pose` and `obs` today; `view` follows): the raw per-frame and per-object facts `rtsm eval` reads, kept before the working memory smooths them away.

```yaml
diagnostics:
  enabled: false        # master switch. false = zero overhead, no file created
  track_drops: false    # per-dropped-mask detail on the frame line (~10 µs per drop)
  event_log_path: null  # null = eval_output/<YYYYMMDD_HHMMSS>/events.jsonl
                        # "my_dir/"          = auto-named file inside that directory
                        # "my_dir/run01.jsonl" = exact path (overwritten with a warning)
  ledgers: false        # write the ledger lines as well (eval runs)
  ledger_format: jsonl  # jsonl | parquet (converted when the file closes; pip install "rtsm[eval]")
```

Turn it on for one replay without editing the file:

```bash
python -m rtsm --replay recordings/session1 --set diagnostics.enabled=true --set diagnostics.ledgers=true
```

The path is logged at startup (`event_log: writing diagnostics to ...`). Read it back:

```bash
python -m rtsm.evaluation.ledger summarize eval_output/<run>/events.jsonl
```

## File layout

Every line is one JSON object with a `kind`. The first line is `kind: "meta"`:

| field | meaning |
|---|---|
| `schema_version` | 3 (2 = trace only; 3 adds the ledger kinds — additive) |
| `created_wall_utc_s`, `created_mono_s`, `pid` | when and where the file was opened |
| `ingest_clock`, `ingest_policy` | the resolved `ingest.clock` / `ingest.policy` of the run |
| `ledgers` | `{"enabled": false}` or `{"enabled": true, "schema": 1, "format": "jsonl"}` |

All timestamps named `timestamp` are `time.monotonic()` at the write; `*_ns` are integer nanoseconds on the sensor clock; `*_s` are float seconds; `*_frac` are fractions in [0, 1].

### Trace kinds (schema_version 2)

- `receiver` — one per receiver decision: `enqueued`, or `dropped` with a reason (`malformed`, `parse_error`, `tracking_state`, `throttle`, `duplicate_ts`, `no_camera_frame`, `queue_full`, `kf_lane_full`, and the lane-side `superseded` / `kf_dropped` / `age` / `closed` written with `source: "lanes"`). Carries `frame_seq`, `t_sensor_ns`, `is_keyframe`, `rx_seq`, `lane`, `depth_valid_frac`.
- `dequeue` — one per dequeued frame: `outcome` (`processed` | `gate_rejected` | `frame_rejected` | `dropped`) and the `reason`, `queue_wait_s`, `clock_s`.
- `frame` — one per processed frame: mask filter counts, scoring summary, `n_matched`, `n_created`, stage timings.

The full definitions, including the A/A comparator contract, are in the module docstring of `rtsm/evaluation/event_log.py`.

## Ledger schema 1 — `pose`

!!! note "In progress"
    Schema 1 is frozen when the last ledger kind (`view`) lands. Until then fields are only added, never renamed.

One line per **sensor frame the receiver saw**, at input rate. Websocket and replay write it at two points of the parser: in the tracking-state filter, *before* a non-normal frame is dropped, and for frames that pass the filter right after the depth decode — before the keyframe rule, the throttle and the queue admission. So throttled, refused and tracking-limited frames are all in it. ZeroMQ writes one line per `rtabmap.tracking_pose` message (never for `kf_pose`).

| field | type | meaning |
|---|---|---|
| `source` | str | `websocket`, `replay` or `zeromq` (same labels as the receiver lines) |
| `rx_seq` | int? | websocket/replay: the receiver's message count — the **join key** to this frame's `receiver` line; `null` on zeromq |
| `frame_seq` | int? | header `frame_id`; `null` on zeromq |
| `t_sensor_ns` | int? | header `timestamp_ns` (0 or missing → `null`); zeromq: the pose stamp — its join key |
| `t_wall_utc_s` | float | header `unix_timestamp`, or this process's `time.time()` |
| `pose_clock` | str | `sender` or `server`: where `t_wall_utc_s` came from |
| `epoch` | int | the frame epoch at the write (a new streaming session bumps it) |
| `tracking_state` | str | the header string verbatim (`normal`, `limited`, `not_available`, …); zeromq always `not_available` |
| `mailbox_write` | bool | the receiver handed this pose to the pose sink (the `/stats.robot_pose` mailbox) |
| `t_wc` | [3]? | translation in metres, after the camera-convention flip — the same values the mailbox and the pipeline see; `null` when the pose did not parse |
| `q_wc_xyzw` | [4]? | unit quaternion, same convention; `null` with `t_wc` |
| `pose_error` | str? | why the pose did not parse (only possible on frames the tracking filter drops; a normal frame still raises as before) |
| `depth_valid_frac` | float? | finite fraction of the decoded depth **before** the confidence filter — identical to the `receiver` line's value; `null` when depth was not decoded (dropped-by-tracking frames, zeromq) |
| `conf_hist` | [3]? | counts of confidence 0 / 1 / 2 over the **raw** confidence map (before it is resized to the depth); `null` without a map |

What it does not carry: the admission outcome (keyframe flag, lane, drop reason). Join to the `receiver` line — on `(source, rx_seq)` for websocket/replay, on `(source, t_sensor_ns)` for zeromq — exactly as `receiver` and `dequeue` lines already join on `t_sensor_ns`.

### Pose-stream health

`rtsm.evaluation.ledger.pose_health(rows)` (also printed by `summarize`) computes, per `(source, epoch)` and in total:

- `sensor_hz`, `dt_ms` percentiles, `jitter_ms` (median absolute deviation of the inter-frame interval) over the **pose stream** — the lines with a pose and a sensor stamp, ordered by stamp;
- `gaps`: intervals longer than `gap_factor` (2.0) × the median interval;
- `limited_episodes`: maximal runs of lines whose `tracking_state` is not `normal`, in file order, with duration and a state histogram (never counted on zeromq, which has no tracking state);
- `discontinuities`: a translation step larger than `disc_base_m + disc_rate_mps × dt` (0.5 m + 1.0 m/s × dt) between consecutive stream poses — the RC-car agent's rule on the full 3-D translation. A **detector**: nothing acts on it;
- `delivery_lag`: arrival time minus sensor time, relative to the first frame — growth means the transport delivers frames slower than the sensor stamps them (a queue building on the sender or in the socket); `n_catchups` counts bursts where the lag drops by more than 50 ms. Under replay the replayer's own pacing drift (about 9 ms per frame) is included, so read it on live runs;
- `depth_valid_frac` and `conf2_frac` (share of confidence-2 pixels) statistics;
- `writes_expected`: lines with `mailbox_write` — on a replay this equals `/stats.robot_pose.writes_accepted`.

## Ledger schema 1 — `obs`

One line per **candidate the associator looked at**, on every processed frame, written on the pipeline thread right after association. So on each frame the number of `obs` lines equals the `frame` line's `scoring.n_selected`. The line keeps the raw measurement the working memory then smooths away, and the outcome with enough context to audit it against the association gates.

| field | type | meaning |
|---|---|---|
| `frame_seq`, `t_sensor_ns`, `epoch`, `is_keyframe` | | the frame; `t_sensor_ns` joins the `frame` and `dequeue` lines |
| `lane`, `keyframe_origin`, `rx_seq` | | the packet's ingest bookkeeping (`null` under the legacy queue) |
| `cam_t_wc` [3], `cam_q_wc_xyzw` [4] | | the packet's camera pose, same convention as the `pose` ledger |
| `cand_idx` | int | mask index in the segmentation output (joins the scoring trace's `mask_idx`) |
| `outcome` | str | `matched`, `created`, `spawn_capped`, `no_p_cam`, `no_embedding`, `create_failed` |
| `object_id` | str? | the matched or created object |
| `p_world` [3]? | | **raw** world point, `T_wc @ p_cam` as the associator computed it, before any EMA |
| `p_cam` [3]?, `range_m` | | camera-frame centroid of the mask and its distance |
| `view_bin` | int? | the working memory's own bin for that direction |
| `cos_sim`, `dist_m`, `px_err` | float? | the winning match's residuals (`matched` only; `px_err` is 0 without intrinsics) |
| `n_nearby`, `n_gate_survivors`, `max_cos` | | audit counters: objects the index returned, how many passed the distance / z / reprojection gates, the best cosine seen among them (passed or not) |
| `matched_without_scoring` | bool | the associator's "scan all objects when the index returns nothing" fallback matched this candidate to the previous candidate's object without gating or scoring it — a known flaw the ledger exposes (no residuals on such lines) |
| `label_topk` | [[str, float]] | detection label first, then the vocabulary classifier's |
| `priority` | float | the scoring priority that selected this candidate |
| `mask` | dict | `area_px`, `bbox`, `coverage`, `border_fraction`, `depth_valid`, `depth_p50`, `depth_spread`, `planar_inlier_pct`, `planar_rms_m`, `centroid_px` (RGB pixel space) |

Invariants a reader can check: every `matched` line without `matched_without_scoring` has `cos_sim ≥ assoc.cos_min` and `dist_m ≤ assoc.gate_dist_base_m`; a `created` line with `n_gate_survivors > 0` has `max_cos < assoc.cos_min`; the working memory's position of any object is a convex combination of its `created` and `matched` `p_world` values, so it lies inside their bounding box (absent pose corrections).

`rtsm.evaluation.ledger.observation_summary(rows)` returns outcome counts, lines per frame, matched observations per object, residual and range statistics, and the view-bin coverage — the inputs to the `rtsm eval` metrics.

## Parquet

`ledger_format: parquet` converts the finished JSONL into one file per kind (`events.pose.parquet`, `events.receiver.parquet`, …) next to it when the run closes; the JSONL stays the source of truth. The same conversion is available offline:

```bash
python -m rtsm.evaluation.ledger parquet eval_output/<run>/events.jsonl --out parquet/
```

Both need `pyarrow` (`pip install "rtsm[eval]"`); a run configured for Parquet without it refuses to start.
