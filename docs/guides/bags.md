# Reading Bags

RTSM reads recorded robot data through the **bag source**: a ROS 1 `.bag`, a rosbag2 directory (sqlite3 or MCAP storage) or a bare `.mcap` file, ingested through the same front-end as the live receivers, so a bag is evaluated exactly the way a stream is ingested (keyframes minted every N frames, the non-keyframe throttle on the sensor clock, admission before any pixel is decoded).

```bash
pip install "rtsm[eval]"
rtsm --bag recordings/session1_bag                 # our own layout (see Record & Replay → Convert a recording to MCAP)
rtsm --bag rgbd_dataset_freiburg3_long_office_household.bag   # a TUM RGB-D ROS 1 bag, discovery alone
rtsm --bag my_rosbag2_dir --set io.bag.topics.depth=/camera/aligned_depth_to_color/image_raw
python scripts/inspect_bag.py my_rosbag2_dir      # what the reader will see: topics, encodings, K, TF hops, stamps
```

`--bag` sets `io.receiver: bag` and resolves the ingest clock and policy as `--replay` does (sensor clock, lossless lane). Frames are offered as fast as the ingest queue accepts them; `--bag-speed 1.0` paces them by their header stamps instead.

## What the reader needs, and how it finds it

| Role | Discovery rule | Override (`io.bag.topics.*`) |
|---|---|---|
| RGB | a `sensor_msgs/Image` or `CompressedImage` topic matching `color`, `rgb`, `image_raw`… and not `depth`, `ir`, `mono`, `left`, `right` | `rgb` |
| Depth | an image topic matching `depth` (`aligned_depth_to_color` / `depth_registered` preferred) | `depth` |
| CameraInfo | the `camera_info` sharing the RGB topic's stem or prefix (or the only one) | `rgb_info`, `depth_info` |
| Pose | `/tf` + `/tf_static` when they chain the camera frame up to a root (`map` / `world` / `odom` preferred), else `nav_msgs/Odometry` / `PoseStamped` | `tf`, `tf_static`, `odom`; `io.bag.world_frame`, `io.bag.camera_frame` |
| Confidence, tracking state, source frame id | image topic named `confidence`; `std_msgs/String` named `tracking_state`; `std_msgs/UInt32` named `frame_seq` (our layout) | `confidence`, `tracking`, `seq` |

Every choice is logged with the rule that made it. Relative topic names (`d455_1_rgb_image`) and ROS 1 frame ids with a leading slash (`/world`) are normalised.

**Pairing.** RGB and depth are paired by header stamp: nearest within `io.bag.pair_tolerance_s` (20 ms, TUM's own tool's value), each depth used once. Unpaired RGB and depth are counted, never silently dropped.

**Pose.** The TF chain from the camera frame to the world frame is composed at the image stamp; each moving hop is interpolated (lerp + shortest-path slerp) between its bracketing samples, static hops are constant. Outside a hop's samples by more than `io.bag.tf_extrapolation_s` the frame has no pose and is counted. Bag poses are in the OpenCV camera convention already (ROS optical frames), so `visualization.apply_camera_flip` is ignored for bags.

**Encodings.** RGB `rgb8` / `bgr8` / `rgba8` / `bgra8` / `mono8` / `8UC3` and `CompressedImage` (`jpeg` / `png`, image_transport's `"<orig>; <codec> compressed <order>"`); depth `16UC1` / `mono16` (millimetres) and `32FC1` (metres; **NaN and 0.0 both mean invalid** — TUM writes NaN, the RealSense D455 writes 0) and `compressedDepth` PNG (16UC1 and the quantised 32FC1). The channel order is keyed on the message encoding, never guessed from pixels. CameraInfo declared at another resolution is rescaled to the RGB size; ARKit-style per-frame intrinsics are honoured per frame.

**Tracking filter.** Only when the bag carries a tracking-state topic; otherwise every frame counts as `normal` and the filter is off.

## What v1 refuses (with the reason, before any model loads)

| Code | Meaning | What to do |
|---|---|---|
| `no_rgb_topic` / `no_depth_topic` / `no_camera_info` | discovery found nothing | `io.bag.topics.*` |
| `no_pose_source` | no TF chain from the camera frame to a root with a moving hop, and no odometry | `io.bag.topics.tf` / `odom`, `io.bag.world_frame`, or record poses |
| `unaligned_depth` | the depth K, scaled to the RGB size, differs from the RGB K by more than 1 % (e.g. RealSense depth not aligned to colour) | re-record with `align_depth`, or `io.bag.assume_aligned: true` only if it IS registered |
| `unsupported_depth_encoding` | `compressedDepth rvl`, or an unknown depth encoding | re-record as PNG / raw |
| `unsupported_schema_encoding` | an MCAP whose schemas are `ros2idl`, not text message definitions | write with the `ros2msg` schema encoding |
| big-endian images | not supported | re-record |

`scripts/inspect_bag.py` prints the same facts the reader uses. Bags written before rosbag2 embedded message definitions (NVIDIA's r2b, rosbag2 v5) carry none: the reader assumes `io.bag.typestore` (ROS 2 Humble by default) and says so.

## Performance note
bz2-compressed ROS 1 bags (the TUM RGB-D distribution) are decompressed twice — once for the pose pass, once for the images — because ROS 1 chunks mix all topics. Convert such bags once (`rosbags-convert`) or accept the cost; uncompressed and zstd-chunked bags are read at disk speed.

## How the behaviour is pinned
- `tests/test_tf_buffer.py`: TF composition and interpolation against analytic poses (the pose-math rule).
- `tests/test_ros_codecs.py`, `tests/test_bag_reader.py`: every encoding, pairing, discovery on the corpus' topic sets, every refusal, a bare MCAP, sqlite3 storage, a `ros2idl` MCAP.
- `tests/test_bag_source.py`: **the parity predicate** — `recordings/session1_bag` through the bag source reproduces the replay receiver's 240 receiver decisions (the B1 record) and its 86 packets bit for bit.
- `eval/datasets/external-corpus.md`: the real files the reader is developed against (TUM fr1 / fr3, NVIDIA r2b) and what each demands.
