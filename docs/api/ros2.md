# ROS 2 Ingest

The contract a ROS 2 graph has to meet for RTSM to consume it. The `ros2` source is a **subscriber-only node**: it reads RGB, depth, camera intrinsics and pose from the graph and feeds the same ingest front-end as the bag reader. By default it publishes nothing; RTSM's outputs stay on the [REST API](rest-api.md), [MCP](mcp.md) and the [WebSocket](websocket.md). `io.ros2.publish` adds an optional state snapshot on two topics ([below](#publishing-optional-off-by-default)).

This page is the reference. Running it (environment, the probe walkthrough, threads and timing, replaying RTSM's own recordings into a graph) is in the [Ingest Sources guide](../guides/ingest-sources.md#ros-2-live); the discovery rules it shares with the bag reader are in [Reading Bags](../guides/bags.md#what-the-reader-needs-and-how-it-finds-it).

## Enabling

```yaml
io:
  receiver: ros2        # or: python -m rtsm --ros2
  ros2:
    node_name: rtsm
    qos: auto
    topics: {}          # role -> topic; empty = discovered by rule
```

```bash
source /opt/ros/humble/setup.bash
rtsm ros2 probe --seconds 5      # what the node would subscribe to, no models loaded
python -m rtsm --ros2
```

The node is named `io.ros2.node_name` (`rtsm`); the probe runs as `rtsm_probe`. A run's `session_id` defaults to `ros2-<start time>`. Live runs use `ingest.policy: latest` (the default) or `legacy`; `lossless` is replay-only.

## What to publish

One topic per role. Roles are found by the bag reader's rules or set under `io.ros2.topics.<role>`; every choice is logged with the rule that made it.

| role | message type | discovery rule | required |
|---|---|---|---|
| `rgb` | `sensor_msgs/Image` or `CompressedImage` | an image topic matching `color`, `rgb`, `image_raw` and not `depth`, `ir`, `mono`, `left`, `right` | yes |
| `depth` | `sensor_msgs/Image` or `CompressedImage` (`compressedDepth`) | an image topic matching `depth`; `aligned_depth_to_color` / `depth_registered` preferred | yes |
| `rgb_info` | `sensor_msgs/CameraInfo` | the `camera_info` sharing the RGB topic's stem or prefix, or the only one | yes |
| `depth_info` | `sensor_msgs/CameraInfo` | likewise for the depth topic | no; without it registration is assumed and reported |
| `tf`, `tf_static` | `tf2_msgs/TFMessage` | `/tf`, `/tf_static` when they chain the camera frame up to a root | one pose source is required |
| `odom` | `nav_msgs/Odometry` or `geometry_msgs/PoseStamped` | the pose source when no TF chain reaches the camera frame | |
| `confidence` | image topic named `confidence` | a uint8 map (0 / 1 / 2) at the RGB size | no |
| `tracking` | `std_msgs/String` named `tracking_state` | the tracking filter is on only when present | no |
| `seq` | `std_msgs/UInt32` named `frame_seq` | the source frame id; arrival index otherwise | no |

Relative topic names and frame ids with a leading slash are normalised.

## Encodings

| payload | accepted |
|---|---|
| RGB | `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `8UC3`; `CompressedImage` `jpeg` / `png`, including image_transport's `"<orig>; <codec> compressed <order>"` format string |
| depth | `16UC1` / `mono16` (millimetres); `32FC1` (metres; NaN and 0.0 both mean invalid); `compressedDepth` PNG (16UC1 and the quantised 32FC1) |
| not accepted | big-endian images; `compressedDepth rvl` (`unsupported_depth_encoding`) |

Channel order is keyed on the message encoding, never guessed from pixels. The payloads go onto the codec layer undecoded; decoding happens after admission. A `CameraInfo` declared at another resolution is rescaled to the RGB size; per-frame intrinsics are honoured per frame.

## Stamps and pairing

RGB and depth are paired by header stamp: the nearest depth within `io.ros2.pair_tolerance_s` (0.02 s), each depth used once. Messages without a usable stamp are counted (`skipped_zero_stamp`), as are unpaired RGB and depth; nothing is dropped silently. `pair_dt_ms_max` on `/stats` is the largest pairing offset seen.

## Pose

The camera pose is composed at the image stamp from the TF chain `camera_frame` → `world_frame`: each moving hop is interpolated between its bracketing samples (lerp, shortest-path slerp), static hops are constant. A stamp outside a hop's samples by more than `io.ros2.tf_extrapolation_s` (0.05 s) gives no pose. A paired frame whose TF has not arrived waits up to `io.ros2.tf_wait_s` (0.5 s), with later frames behind it, then counts as `pose_missing`.

Defaults: `world_frame` is the TF root that reaches the camera (`map`, `world`, `odom` preferred); `camera_frame` is the RGB `CameraInfo` `frame_id`. Both can be set. Without a TF chain, `odom` is used. Poses are taken in the OpenCV camera convention (ROS optical frames); `visualization.apply_camera_flip` does not apply.

## Registration

Depth must be registered to the RGB. When a depth `CameraInfo` is present, its K scaled to the RGB size must match the RGB K within 1 %; otherwise the source refuses with `unaligned_depth`. `io.ros2.assume_aligned: true` accepts a differing K, only for depth that is in fact registered. Without a depth `CameraInfo`, registration is assumed and `/stats.source.registration` says so.

## QoS

| `io.ros2.qos` | subscription |
|---|---|
| `auto` (default) | reliable when any publisher on the topic offers reliable, best-effort otherwise, read from the publishers' offered QoS per topic |
| `reliable`, `best_effort` | forced |

`tf_static` always subscribes transient-local. Image subscriptions keep a queue depth of 100, TF 200. The probe prints the chosen and the offered QoS per role; a mismatch is the usual reason a subscriber's callback never fires.

## Startup and refusals

Discovery polls the graph for up to `discovery_timeout_s` (10 s) until an RGB, a depth and a `CameraInfo` topic are advertised, then waits up to `ready_timeout_s` (10 s) for a `CameraInfo` message and a moving TF chain to the camera frame. Pairs that arrive before the camera info, the world frame and the registration check are known are held (256) and flushed in order once the source is ready. Otherwise the source refuses before any model loads, with the bag reader's codes:

| code | meaning | what to do |
|---|---|---|
| `no_rgb_topic`, `no_depth_topic`, `no_camera_info` | discovery found nothing for the role | `io.ros2.topics.*` |
| `no_pose_source` | no TF chain from the camera frame to a root with a moving hop, and no odometry | `io.ros2.topics.tf` / `odom`, `io.ros2.world_frame` |
| `unaligned_depth` | depth K differs from RGB K by more than 1 % | align the depth at the source, or `assume_aligned: true` only if it is registered |
| `unsupported_depth_encoding` | `compressedDepth rvl` or an unknown depth encoding | publish PNG or raw depth |

`rtsm ros2 probe` exits 0 when the stream is usable, 1 with the refusal printed, 2 without `rclpy`.

## Observability

**The probe.** `rtsm ros2 probe [--seconds N] [--qos auto|reliable|best_effort] [--topic ROLE=TOPIC ...] [--world-frame F] [--camera-frame F] [--assume-aligned] [--json]`. `--json` returns `topics`, `msgtypes`, `qos` (chosen and offered per role), `advertised`, `camera_info_seen`, `tf` (roots, hops, chain, world and camera frame, pose kind), `registration`, `refusal`, `ok`.

**`/stats.source`** while running (the [REST API](rest-api.md#statistics) shows the full object):

| field | meaning |
|---|---|
| `frames_seen` | RGB messages with a usable stamp |
| `paired`, `unpaired_rgb`, `unpaired_depth` | pairing outcome |
| `pose_missing` | paired frames whose TF never arrived within `tf_wait_s` |
| `skipped_zero_stamp`, `decode_errors` | counted, not dropped silently |
| `yielded`, `enqueued`, `admit_errors`, `before_ready_dropped`, `pending`, `queued` | frames built, admitted by the front-end, and the queue state |
| `pose_kind`, `world_frame`, `camera_frame`, `tf_chain` | the pose source in use |
| `registration` | the registration check's verdict |
| `topics` | the chosen topic per role and the rule that chose it |
| `ready`, `error`, `refusal_codes` | the source's state |
| `node`, `graph` | the node name and the discovery result (topics, QoS chosen and offered, advertised topics, refusal) |

## Publishing (optional, off by default)

`io.ros2.publish.enable: true` publishes the memory's objects as a **periodic state snapshot** on two topics. Two standard message types, so no `ament` package; RTSM stays pip-installable. The publisher is independent of the ingest source: frames may come from Lens, a bag or a ROS 2 graph. It runs its own node, `<io.ros2.node_name>_publisher`, and needs `rclpy` plus the two message packages (`apt install ros-$ROS_DISTRO-vision-msgs ros-$ROS_DISTRO-visualization-msgs`); enabled without them, RTSM refuses at start with the hint.

```yaml
io:
  ros2:
    publish:
      enable: true
      objects_topic: /rtsm/objects   # vision_msgs/Detection3DArray
      markers_topic: /rtsm/markers   # visualization_msgs/MarkerArray; empty = off
      frame_id: auto                 # the ros2 source's world frame, else map; or a frame name
      rate_hz: 5.0
      confirmed_only: true
      qos: reliable                  # reliable | best_effort
      latched: true                  # transient-local durability, depth 1
      marker_size_m: 0.15
      max_hypotheses: 5
```

Every tick publishes the whole list of (confirmed) objects as it stands. That is idempotent: a subscriber that joins late, drops a message or restarts is right again at the next tick, and with `latched` the last snapshot is delivered to anyone who subscribes afterwards. There is no per-event stream and no service; queries (semantic, spatial) stay on the [REST API](rest-api.md) and [MCP](mcp.md).

**The field mapping** (the contract):

| message field | value |
|---|---|
| `Detection3DArray.header` | `frame_id` = the world frame; `stamp` = the node clock at the tick |
| `Detection3D.id` | the object id |
| `Detection3D.results[k].hypothesis` | `class_id` = a label, `score` = its smoothed score; best first, at most `max_hypotheses` |
| `Detection3D.results[k].pose.pose.position` | the object's world position; orientation identity |
| `Detection3D.bbox.center.position` | the world position |
| `Detection3D.bbox.size` | `marker_size_m` on every axis: RTSM estimates no extent |
| `Marker` ns `rtsm/objects` | a sphere of `marker_size_m` at the position; id = crc32 of the object id; dimmer when provisional |
| `Marker` ns `rtsm/labels` | `TEXT_VIEW_FACING` with the primary label above the sphere, same id |
| `MarkerArray` | a `DELETEALL` marker first, then the current objects |

`/stats.ros2_publish` carries the configuration as resolved, the node name, `frame_id_resolved`, `ticks`, `published`, `last_objects`, `last_markers`, `errors`, `last_error`.

What a consumer sees is the memory as it is: on a recording where the association gate fragments one object into several ids, every id is published.
## Limits

- No services or actions, no `ament` package, no launch files. Publishing is the optional snapshot above; nothing is published per event.
- Exercised on ROS 2 Humble under WSL2 (Ubuntu 22.04, Python 3.10) against one recording replayed with `ros2 bag play`; `tests/test_ros2_source.py` pins that a frame built live equals the frame the bag reader builds from the same stream. Jazzy is supported by the environment recipe but has not been exercised. There is no ROS 2 container in CI.
- `rclpy` exists only inside a sourced ROS 2 environment on Linux; elsewhere the source refuses at start with a one-line hint.
