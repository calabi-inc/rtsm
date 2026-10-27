# Changelog

All notable changes to `rtsm`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) with the caveat that anything before 1.0 may change between minor versions when the changelog says so.

## [Unreleased]

### Changed
- Documentation site: the diagrams drawn with box-drawing characters were set in a fallback font (the self-hosted Geist Mono carried the latin subset only), so their columns did not line up; arrows and box-drawing glyphs now come from Geist and Geist Mono themselves (`docs/assets/fonts/*-symbols.woff`, loaded only by pages that use them).

## [0.3.1] - 2026-10-26

The ledgers release. Everything a perception team needs to tell *why* an id is missing, moved or gone is now written by the engine itself, additively: every candidate the mask filter or the top-K cut dropped is located on the frame line with its box, the observation box is given in RGB pixel space and can carry the mask as COCO RLE, the memory records its own expiries, index evictions and pose-correction moves, and every raw detection of the backend and of a recording's own detections topic sits on the frame before anything filtered it. `rtsm eval` gains a timing run (`--speed`, the deployed ingest policy, the compute it ran on in `resolved.json`, arrival stamps on every receiver line) and the bag source paces on an absolute schedule and says whether the pace held. The live `ros2` source takes a `vision_msgs` detections topic. Ledger schema 1 → 5 and trace `schema_version` 3 → 4 over the release, every step additive (new kinds and fields, nothing renamed or removed); the memory's behaviour is unchanged on every check (the TUM fr1 fingerprint held through the five changes). No `metrics_schema` change. The `io.websocket.*` aliases still ship (removal stays at 0.4.0).

### Added
- **Raw detections.** Ledger kind `det` (ledger schema 4 → 5, additive): one line per processed frame and source with every raw detection before the mask filter, the selection and association. `backend`: the segmentation backend's own output (idx = the index the `obs` line's `cand_idx` and the `frame` line's `dropped` rows use, label and score as reported, the box in RGB pixel space, the `dual` confirmation); `topic`: the recording's or the live stream's detections topic as paired to the frame, whatever backend ran, with the message type, the stamp offset and the adapter's drops. A model's detections and a customer's sit side by side on the same frame.
- **Live ROS 2 `detections` role**: a `vision_msgs/Detection2DArray` or `Detection3DArray` topic (discovered by the bag reader's rules or `io.ros2.topics.detections`) is paired to each frame by nearest stamp, waiting `io.ros2.detections_wait_s` (0.2 s) for a message that follows the image, converted by the same adapters as the bag path, recorded on the `det` ledger and, under `segmentation.backend: external`, run as the memory's input. A missing `vision_msgs` package skips the role with a warning; the rest of the stream runs.
- **Optional masks.** `diagnostics.save_masks` (`rtsm eval --save-masks`): each selected candidate's mask on its `obs` line as `mask.rle`, COCO compressed RLE at mask resolution (`{"size": [h, w], "counts": "<string>"}`, the form pycocotools and FiftyOne read), encoded and decoded by the new `rtsm.utils.rle` in pure numpy. Off by default; about 0.3–2 KB per observation when on. Ledger schema 3 → 4 (additive). `resolved.json` records `save_masks`.
- **Memory events.** Ledger kind `mem` (ledger schema 2 → 3, additive): one line per change the working memory makes to an object on its own, with the object's state before it: `expire` (a proto past its TTL), `evict` (the proximity index's cell over its cap dropped the id from the index; the object stays in memory but no candidate can reach it) and `correct` (a pose correction moved the stored position: `p_before`, `p_after`, `shift_m`, `demoted`, the corrected frame id, `direct` or `fallback`). Carries `age_s` / `lifetime_s` on the ingest clock and the processed frame current at the event. Before, expiries were a debug log line, evictions were silent, and a correction left one session-level line with no per-object delta. `ProximityIndex.insert` / `update` now return the ids they evicted.
- `rtsm eval --speed` (`eval.speed`): pace the input at a multiple of its recorded rate (1.0 = real time by sensor stamps; unset = as fast as the queue accepts, as before). With `--set ingest.policy=latest` this is a timing run: the deployed lanes drop the frames the pipeline cannot keep up with, as on the robot; the run warns when a non-lossless policy is used without pacing. `resolved.json` and the ledger's meta line record `speed`.
- The bag source paces on an absolute schedule anchored at the first frame (the reader's own time no longer stretches the pace) and reports whether the pace was held (`stats()["pace"]`: `late_frames`, `max_lag_s`, `held`; in `summary.json` under `source`), with a warning at completion when it was not.
- `resolved.json`: `policy_configured`, and the compute the run used: `gpu` and `cuda` as torch reports them, `device_model` (the device tree, a Jetson's board name) and `power_mode` (the Jetson's nvpmodel status), `null` where absent.
- Receiver lines: `t_arrival_mono` / `t_arrival_wall_utc`, the transport's arrival stamps of the frame (before decode), on every line of the admit chain and on the lane-side drop lines; `IngestMeta` carries them on the packet.
- **Located drops.** The `frame` line carries `dropped`: one row per candidate that did not reach association, with the stage and reason (`filter`: `area` | `depth_valid` | `depth_spread`, under `diagnostics.track_drops`; `select`: `dedup` with `dup_of`, or `topk`), the box in RGB pixel space, the selection priority and the backend's own label and confidence for that mask; plus `rgb_hw` / `mask_hw`. With the frame's `obs` lines this accounts for every raw mask, so a reader can tell "no detection" from "detected, then filtered or cut". `rtsm eval` turns `track_drops` on. Trace `schema_version` 3 → 4 (additive).
- Observation ledger: the `mask` block gains `bbox_rgb` (the box in RGB pixel space; `bbox` stays in mask space) and `mask_hw`. Ledger schema 1 → 2 (additive).

### Changed
- `dual` backend: a FastSAM-only mask reports no detection confidence (`label_confidence` entry `null`) instead of `0.0`. The observation ledger already recorded `detector_score: null` for them (no label, so nothing was merged); the `scores` tensor is unchanged.
- Documentation site: the fonts (Geist, Geist Mono, Manrope) are served from the site itself under `docs/assets/fonts/` (SIL Open Font License) instead of from Google Fonts, so a visit to the docs sends no request to a third-party font service.

### Fixed
- `rtsm demo`: the completion summary counted the working memory after the exit freeze had moved it into the session record, so it always read 0; it now counts the frozen session.
- A websocket to `/ws` on a process without a visualization server (`--no-viz`) fell through to the static-files mount and logged an ASGI traceback per attempt; the handshake is refused (close code 1008).

## [0.3.0] - 2026-10-14

The release after 0.2.1: sessions (a SLAM boundary freezes the live session; a dashboard picker and `GET /sessions`), the `ros2` live source with an optional publisher, `rtsm eval --save-crops`, run provenance down to the weight files' hashes, validated weight files with `segmentation.weights_policy`, the prompt-free label switch, dashboard navigation, and the fixes from the pre-release review (all reproduced before they were fixed). No metric, report or ledger schema number changes; `resolved.json` gains fields. The `io.websocket.*` timing aliases stay until 0.4.0.

### Added
- `io.ros2.publish` (off by default): RTSM's objects as a periodic state snapshot on ROS 2 topics, `vision_msgs/Detection3DArray` and `visualization_msgs/MarkerArray`, independent of the ingest source; latched QoS, configurable topics, rate and frame; counters on `/stats.ros2_publish`. Standard message types only, no `ament` package.
- **Dashboard session picker** (`--viz`): the Session row lists the live session and every frozen one; a frozen session shows read-only with its clouds, objects, snapshots and a banner with Live / Pin / Merge into live / Delete; the live scene keeps counting in the HUD meanwhile and **Live** resyncs it; **End** freezes the live session by hand; sessions merged into the live one are nested in the picker and drawn dimmed in the live scene as prior content. WebSocket `session_update` (first message, and on every change) and `session_refined`; client commands `resync` and `viewing`. REST `GET /sessions/{key}/keyframes`, `/clouds` (consecutive `mesh_create` packets), `/objects/{oid}/snapshots[/{i}/image]`.
- `scripts/split_recording_session.py`: clone a recording so its later parts carry new session ids (frames hard-linked; one `--at` per split, so `--at 0.33 --at 0.66` makes three sessions); the replay source honours a `hello` recorded mid-stream, so one replay produces every part as its own session.
- **Validated weight files** (`rtsm/models/weights.py`): the FastSAM and YOLOE files every documented number was produced with, by sha256. A backend checks its file when created (`segmentation.weights_policy: warn` logs, `strict` refuses, `off` ignores); `resolved.json` records `validated` / `mismatch` per file and the report header shows it; `scripts/fetch_models.py` verifies after a download and `--verify` reports. Why: ultralytics fetches by file name and replaced `yoloe-26s-seg-pf.pt` upstream on 2026-09-17 with an export that labels an office `heat` and `razor blade`.
- **Sessions** (`rtsm/core/sessions.py`, `sessions:` in the config): a SLAM boundary (a new session id from the source, a receiver-inferred restart, a `reset` correction, `POST /reset`) freezes the live session instead of discarding it or piling the next map into it. A frozen session keeps its objects, keyframe clouds, pose history, vectors, crops and a timeline (start, reconnect, tracking gaps, refinements, relocalisation, merge, freeze), is persisted under `sessions.dir/<key>/` and listed again after a restart. A pose-correction batch from any lane is routed by keyframe id to the session that owns the keyframes, live or frozen; keys may be the pipeline's `ws_<seq>`, a source keyframe id through an alias learnt from `FramePacket.kf_id`, or `ts:<ns>` matched to a keyframe stamp. An empty batch is announced-only (a timeline event, nothing moves). A session ending with no keyframes and no objects is discarded; a frozen session a correction ties to the live map is marked merged (its objects stay its own); retention removes the oldest sessions past `keep_on_disk` / `max_age_days`, never the live chain, a pinned session or one served recently.
- REST: `GET /sessions`, `GET /sessions/{key}`, `POST /sessions/end`, `POST /sessions/{key}/merge` | `unmerge` | `pin` | `unpin`, `DELETE /sessions/{key}` (409 with a reason for the live session, a pinned one without `?force=true`, or a member of the live chain); `/healthz.sessions`.
- ZeroMQ lane: `rtabmap.map_correction` (the bridge's map→odom) is subscribed and composed onto every tracking pose at receive time (`io.apply_map_correction`, default true), since the bridge's `tracking_pose` is the odometry pose while `kf_pose` is already map-frame; the message's `map_id` becomes the session id `rtabmap-map-<id>`; each `map_correction` ends the previous burst of `kf_pose_update`; the RTAB-Map node id rides on the keyframe (`FramePacket.kf_id`). `/stats.source.map_correction`.
- Visualization server: `kf_pose_update` (per-keyframe re-posing from an adapter) is translated through the alias table and coalesced into one correction batch, so objects, the TSDF, the pose history and the session registry see it exactly as a Lens `pose_corrections` batch; before, it re-posed the cloud only.
- Lens `pose_corrections`: an empty `corrections` dict is forwarded as an announced-only correction; the message may carry `kind` (`loop_closure` | `relocalized_in_prior_map` | `reset` | `unknown`) and `map_ref`.
- Ledger kind `session` (`SessionEvent`): start / freeze / discard / refine / merge / unmerge / gc / delete.
- API docs: the ROS 2 ingest contract page (`docs/api/ros2.md`), linked from the API index, the REST `source` object and the Ingest Sources guide.
- **`ros2` live source** (`io.receiver: ros2`, `--ros2`) — a minimal rclpy subscriber on the same ingest front-end as the bag reader: topic roles by the bag reader's rules or `io.ros2.topics`, QoS matched to the publishers (`io.ros2.qos: auto`), TF composed at the image stamp, the registration check and the same refusals; a spin thread decoupled from the pairing/admission worker; `rtsm ros2 probe` prints the topics, the publishers' QoS and the TF / CameraInfo state before any model loads. Runs inside a sourced ROS 2 environment on Linux; refuses with a hint elsewhere. The live path reproduces the bag reader frame for frame on session1 (test).
- `rtsm eval --save-crops`: writes each object's JPEG snapshots into `run_N/crops/<id>/<k>.jpg` with an `index.json`, for report renderers; off by default.
- Provenance: `resolved.git_dirty` and `resolved.tree_digest` (a digest of `git diff HEAD` when the checkout has uncommitted tracked changes); the report's header marks a dirty commit.
- The report's method notes define *engine-confirmed*, *re-identified* (a reassociation) and the fingerprint.
- The eval guide names the run-directory layout, the schemas and the `rtsm.evaluation` modules as the surface other tools build on.
- Observation ledger: `detector_label` / `detector_score`, the backend's own per-mask label and raw confidence (additive).
- `segmentation.labels.prompt_free_primary` (`detector`, the default and the behaviour so far, or `classifier`): under a prompt-free detector (`dual`, `yoloe` with `-pf` weights) the model's own class name leads an object's scored labels by default; `classifier` keeps it out so the CLIP vocabulary classifier's label decides. A mitigation for weights that label badly: the `yoloe-26s-seg-pf.pt` that ultralytics 8.4.31 downloads from its v8.4.0 assets named shelves `heat` and boxes `razor blade` on TUM fr3 (82 % of tracks disagreeing on label), while the March export the dev box uses labels the same scene `bottle`, `cup`, `keyboard`. `SegmentationResult.label_source` records where a backend's labels come from. No default behaviour changes.
- Provenance: `resolved.model_files` (the weight files of the active backend with the file's size and sha256, or `exists: false`) and `resolved.hf_models` (every `*model_id`). The config fingerprint covers the path, not the file: two runs with one fingerprint and one commit differed because one checkout had no `model_store/` and ultralytics downloaded other `yoloe-26s-seg-pf.pt` weights, whose labels are unusable (`heat`, `array`, `Prague Castle` at 0.86 confidence).
- Dashboard navigation (`--viz`): a visible rotation-centre anchor with a drag gizmo, double-click placement and a show/hide button (P); a mouse-scheme switch (three.js default, Blender, Maya) kept in the browser; zoom to the cursor, inertia, orbit pivot at the depth under the cursor while the anchor is hidden, numpad view presets (1 / 3 / 7, Ctrl for the opposite side, 9 opposite, 2 / 4 / 6 / 8 orbit 15°), frame the selected object (F) or everything (Home), arrow-key panning, and a key-binding panel (H). Orthographic view is not provided.

### Changed
- `POST /reset` freezes the live session first and reports it under `session`; `{"discard": true}` keeps the previous behaviour (nothing kept). `sessions.enable: false` restores the previous behaviour entirely.
- On the ZeroMQ lane, poses from non-keyframe frames are now in the map frame once the bridge publishes `map_correction` (they stayed in the odometry frame before); `io.apply_map_correction: false` restores the old poses.
- The frame id the working memory and the viz registry share is `ws_<seq>` from the source's seq, else `rx_<n>` from the receiver's attempt count (ZeroMQ packets carry no seq; they used to collide on `ws_None`).
- `FramePacket` gains `session_id` and `kf_id`; `SourceContext` gains `on_session_change` and `on_corrections_flush`; `KeyframeRecord` gains `session_key`; `on_pose_corrections_batch` callbacks may take `kind`, `session_id`, `map_ref` keywords.
- The bag reader's world-frame rule is a shared function (`choose_world_frame`), used by bags and the live source alike.
- `/stats` carries a `source` object with the ingest source's own counters for sources that keep them (`bag`, `ros2`).
- The runner stops the ingest source on exit (`source.stop()`), so a live rclpy node is torn down before the interpreter finalises.


### Fixed
- `python -m rtsm.cli ...` ran nothing (no main guard); it now behaves like the `rtsm` console script.
- Sessions: the in-RAM cap evicted a frozen session's clouds, vectors and crops before the writer had persisted them, so the folder on disk was empty (deterministic with `keep_in_memory: 0`, and with the default whenever boundaries outran the writer); a record is never evicted until it is on disk. `DELETE /sessions/{key}` during an in-flight write no longer leaves a folder that reappears at the next start.
- `rtsm demo` lists the sessions on disk and runs retention at start, and waits for the writer at exit, as `python -m rtsm` does.
- Sessions: a keyframe of the next session admitted on the receiver thread before the pipeline decided the boundary was frozen into the ending session; keyframes now carry the packet's frame epoch and a freeze keeps newer-epoch keyframes live.
- ZeroMQ receiver: `stop()` now signals the poll loop and waits for it before closing the sockets; closing them from the main thread while the loop polled aborted the process inside libzmq on every Ctrl-C and skipped the rest of the shutdown. RTAB-Map map id 0 (a fresh database's first map) now names its session like every other map id.
- ROS 2 live source: the RGB / depth pairer and the TF buffer are bounded on a live stream (one stalled image stream no longer buffers the other without limit; TF keeps a 10 s horizon per hop and inserts out-of-order samples in place). A refusal destroys the node but leaves the rclpy context up for the publisher; the context is shut down at process exit.
- `segmentation.weights_policy: off` written bare in YAML (which PyYAML reads as the boolean False) is accepted as `off`.
- The eval report listed the FastSAM / YOLOE files as "missing at run start" for runs of backends that never load them; `resolved.model_files` now covers the active backend only.

### Deprecated
- The `io.websocket.keyframe_every_n` / `io.websocket.nonkf_min_interval_s` aliases announced for removal in 0.3.0 stay for this release; their removal moves to 0.4.0.

## [0.2.1] - 2026-10-04

### Added
- **Docker image** — `ghcr.io/calabi-inc/rtsm:<version>` / `:latest`, built from the tagged source by the release workflow: Python 3.12, PyTorch cu128, the `gpu`, `eval` and `mcp` extras, a non-root user, weights on a `/models` volume; `docker/docker-compose.yml` for the server; pull requests that touch the image build it without pushing (#55).

### Changed
- Documentation site redesigned in the calabi.com design language, with section landing pages, a factual landing page that states the known limitations, and the architecture diagram redrawn and brought up to date (#54, #56, #58). The favicon follows the browser's colour scheme (#57).
- The installation page says that a Linux container and a Windows install give slightly different numbers on the same bag, each deterministic on its own (#55).

### Fixed
- Every non-editable install logged `entry point 'replay' collides with a built-in source` at startup: the package advertises its own built-ins under the `rtsm.sources` group, and the registry now recognises them instead of warning (#55).
- The installation page described a Docker image and two Dockerfiles that did not exist (#55).

## [0.2.0] - 2026-10-03

The release after the April 0.1.1: a deterministic, observable ingest; offline evaluation on bags; the first pieces of an SDK. Numbers in pull-request references are on [github.com/calabi-inc/rtsm](https://github.com/calabi-inc/rtsm/pulls?q=is%3Apr+is%3Amerged).

### Added
- **`rtsm eval <bag-or-recording>`** — a headless evaluation run of the full pipeline on a bag or a Lens recording: sensor clock, lossless lane, an isolated vector store per run, N same-input repeats, deterministic termination (#48). Modes `as_deployed`, `dense` and `every_frame` (#48, #49).
- **The single-bag report** — `metrics.json` + `report.md` written after the repeats: spatial clusters, detection over in-frustum views, label disagreement, position scatter, duplicate spawns, revisits, worst moments, admission, pose-stream health, each number with its same-input floor over ≥ 3 runs; `rtsm report <dir>` regenerates it (#49).
- **Diagnostics event log** (`diagnostics.enabled`): the frame-flow trace (`receiver`, `dequeue`, `frame` lines) (#30) and the **ledgers** `pose`, `obs`, `view` (schema 1, frozen) with `python -m rtsm.evaluation.ledger` and an optional Parquet export (#42, #43, #44).
- **Bag readers** — ROS 1 `.bag`, rosbag2 (sqlite3 and MCAP) and bare MCAP through `rosbags`, with topic auto-discovery and overrides, TF composition at the image stamp, a registration check, and explicit refusals before any model loads; `python -m rtsm --bag` (#47). A Lens recording → rosbag2/MCAP converter, `scripts/recording_to_mcap.py` (#46).
- **The ingest front-end** — one shared admission chain for every source, versioned contracts (`rtsm/io/contracts.py`), a codec layer, and a source registry with the `rtsm.sources` entry-point group (#45).
- **The detections adapter** — a `vision_msgs` `Detection2DArray` / `Detection3DArray` topic in a bag becomes the pipeline's candidates (`segmentation.backend: external`); detectors without confidence are supported end to end; the report names the detector; `scripts/export_detections.py` (#52).
- **Python client** `rtsm.client` (`RtsmClient`) — the REST API's SDK, `requests` only, imports without the perception stack (#50); the server-side API contract test (#52).
- `rtsm config explain | show | validate` with documented tuning controls (#25); dead config keys removed (#26).
- A frame-quality gate before segmentation (dark / flat / depth-less frames) (#27).
- The frame-flow watchdog and `/healthz.frame_flow` (#22); the analytics ticker, `/healthz.ingest` and `/stats/analytics.rollup` (#37); frame rejections on the dashboard (#34).
- `ingest.policy: latest | lossless | legacy` — per-lane admission with bounded memory (#33); admit-before-decode (#32); the receive-time pose mailbox with frame epochs (#20, #36).
- The `[edge]` extra and `scripts/install_torch.py` for Jetson / ARM (#19).
- CI: the core CPU suite on Linux and Windows and a wheel install job on every pull request (#51). `rtsm version`.

### Changed
- **Headless by default.** The 3-D dashboard and TSDF fusion are off; `python -m rtsm --viz` turns the dashboard on (`rtsm demo` keeps it on). The viz path held the interpreter for seconds per keyframe under load (#41).
- **Ingest timing keys moved** from `io.websocket.*` to `ingest.*` (`keyframe_every_n`, `nonkf_min_interval_s`, `pair_window_*`); the old paths are aliased with a deprecation warning until 0.3.0 (#39).
- Replays run on the **sensor clock** by default (`ingest.clock: auto`), so a replay gives the same admitted sequence and memory at any speed (#31).
- Packaged defaults reconciled: the public `grounded_sam2` defaults ship; experiment tuning lives in profiles (#29).
- The `[mcp]` extra is pinned `mcp<2` (the 2.x package removed the server API the MCP modules use) (#51).
- Repository scope: `rtsm` carries the product and the SDK only (#50).

### Deprecated
- `io.websocket.keyframe_every_n`, `io.websocket.nonkf_min_interval_s` and the other timing keys under `io.websocket` — read through an alias with a warning; removed in 0.3.0 (#39).

### Removed
- `examples/` (the RC-car reference agent and the ESP32 firmware) — now [Vipipi/rtsm-rc-car-agent](https://github.com/Vipipi/rtsm-rc-car-agent); `AGENTS.md`, `reports/`, the evaluation tooling under `eval/` (#50).

### Fixed
- RGB frames reached the models in the wrong channel order; the pipeline now flips BGR→RGB exactly once at the model boundary (#29).
- A pose that failed to convert produced an identity pose; the frame is dropped instead (#23).
- A failed reset persisted silently (#24); an empty `except` swallowed errors (#28).
- The static frontend directory is validated before serving (2026-08-06).
- Demo frontend dependencies: `postcss` path traversal (#21), `vite` 6 → 8 (`esbuild` advisory).

## [0.1.1] - 2026-04-14
- Demo GIF and README; `force_all` vector flush in replay mode; expanded indoor vocabulary.

## [0.1.0] - 2026-04
- First public release: the RGB-D + pose pipeline, dual-confirmation segmentation, working memory, FAISS semantic search, REST API, MCP tools, the 3-D dashboard, record and replay.
