# Changelog

All notable changes to `rtsm`. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [Semantic Versioning](https://semver.org/) with the caveat that anything before 1.0 may change between minor versions when the changelog says so.

## [Unreleased]

### Added
- **`ros2` live source** (`io.receiver: ros2`, `--ros2`) — a minimal rclpy subscriber on the same ingest front-end as the bag reader: topic roles by the bag reader's rules or `io.ros2.topics`, QoS matched to the publishers (`io.ros2.qos: auto`), TF composed at the image stamp, the registration check and the same refusals; a spin thread decoupled from the pairing/admission worker; `rtsm ros2 probe` prints the topics, the publishers' QoS and the TF / CameraInfo state before any model loads. Runs inside a sourced ROS 2 environment on Linux; refuses with a hint elsewhere. The live path reproduces the bag reader frame for frame on session1 (test).
- `rtsm eval --save-crops`: writes each object's JPEG snapshots into `run_N/crops/<id>/<k>.jpg` with an `index.json`, for report renderers; off by default.
- Provenance: `resolved.git_dirty` and `resolved.tree_digest` (a digest of `git diff HEAD` when the checkout has uncommitted tracked changes); the report's header marks a dirty commit.
- The report's method notes define *engine-confirmed*, *re-identified* (a reassociation) and the fingerprint.
- The eval guide names the run-directory layout, the schemas and the `rtsm.evaluation` modules as the surface other tools build on.

### Changed
- The bag reader's world-frame rule is a shared function (`choose_world_frame`), used by bags and the live source alike.
- `/stats` carries a `source` object with the ingest source's own counters for sources that keep them (`bag`, `ros2`).
- The runner stops the ingest source on exit (`source.stop()`), so a live rclpy node is torn down before the interpreter finalises.

### Fixed
- `python -m rtsm.cli ...` ran nothing (no main guard); it now behaves like the `rtsm` console script.

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
