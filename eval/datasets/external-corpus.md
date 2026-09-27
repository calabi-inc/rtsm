# External bag corpus for the eval reader (P3 task 1 inputs) — fetched 2026-09-25

Real files we did not write, one per format the reader must support. They live in `recordings/external/` (gitignored,
regenerable with the commands below) and are the second development input beside `recordings/session1_bag/` (our own
layout, anchor fidelity). `scripts/inspect_bag.py <path>` prints what any of them contains; the per-file reports are
saved next to the files as `*.inspect.json`. Licences: TUM has no licence text ("please refer to the respective
publication"); NVIDIA r2b 2023 is CC BY 4.0. Internal validation only until each is re-checked for the intended use.

| File | Format | Source | Size (bytes) | sha256 |
|---|---|---|---|---|
| `rgbd_dataset_freiburg3_long_office_household.bag` | **ROS 1 bag** | https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.bag | 1 700 571 628 | `a3eb062a77e5474d4919bc601bf4cc1b0ed73009b7a7def0337c4d7b3bc286ba` |
| `rgbd_dataset_freiburg1_desk.bag` | **ROS 1 bag** | https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.bag | 389 783 328 | `91f69b853e46e69979977314bcf5889c53d8a034b0d6d780ed8d4cf5d061fee1` |
| `r2b_cafe/` (`metadata.yaml` + `r2b_cafe_0.db3`) | **rosbag2, sqlite3, zstd MESSAGE** (v5) | NGC `nvidia/isaac/r2bdataset2023` v1 (`https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/1/files/r2b_cafe/...`) | 1 212 297 216 | `005d4e4cdc729cb6d4330093b59be07468fbced30b1c8b5b74aa315a5428040c` |
| `r2b_hope/` (`metadata.yaml` + `r2b_hope.db3`) | **rosbag2, sqlite3** (v5) | same, `.../files/r2b_hope/...` | 31 162 368 | (31 MB edge-case sample) |
| `../session1_bag/` | **rosbag2, MCAP, zstd** (v9, our layout v1) | `scripts/recording_to_mcap.py recordings/session1 recordings/session1_bag` | 1 029 404 036 | regenerable |
| bare MCAP | the `.mcap` inside `session1_bag/` opened alone (no `metadata.yaml`) is the "non-rosbag2 MCAP" case for the `mcap` + `mcap-ros2-support` path | — | — | — |

```bash
mkdir -p recordings/external && cd recordings/external
curl -L -o rgbd_dataset_freiburg3_long_office_household.bag https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.bag
curl -L -o rgbd_dataset_freiburg1_desk.bag https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.bag
B=https://api.ngc.nvidia.com/v2/resources/nvidia/isaac/r2bdataset2023/versions/1/files
mkdir -p r2b_cafe r2b_hope && curl -L -o r2b_cafe/metadata.yaml $B/r2b_cafe/metadata.yaml && curl -L -o r2b_cafe/r2b_cafe_0.db3 $B/r2b_cafe/r2b_cafe_0.db3
curl -L -o r2b_hope/metadata.yaml $B/r2b_hope/metadata.yaml && curl -L -o r2b_hope/r2b_hope.db3 $B/r2b_hope/r2b_hope.db3
```

## What each file contains (from `inspect_bag.py`, 2026-09-25)

### TUM fr3/long_office_household — the first external replay target
87.3 s, 31 121 messages. `/camera/rgb/image_color` **`bgr8`** 640×480 at 29.6 Hz (2 585), `/camera/depth/image` **`32FC1`
metres, NaN = invalid (19.2 %)**, p50 2.4 m, 28.7 Hz (2 509) — RGB and depth are NOT stamp-synchronised (different
counts, ~1 ms apart at best). Both `CameraInfo` topics carry the same calibrated K (fx 537.96, fy 539.60, cx 319.18,
cy 247.05, `plumb_bob`, 5 coefficients) on `/openni_rgb_optical_frame` → depth is registered to RGB. `/tf`
(`tf/msg/tfMessage`, ROS 1 type, 139.8 Hz) holds the mocap `/world → /kinect` hop at 100 Hz plus the calibration chain
`/kinect → /openni_camera → /openni_rgb_frame → /openni_rgb_optical_frame` (constant, re-published) — four hops to
compose at the image stamp, interpolating the moving one. Frame ids carry ROS 1 leading slashes. Also
`/cortex_marker_array` (mocap markers, 99.9 Hz); no IMU in this sequence. Log time == header stamp.

### TUM fr1/desk — the quick one
23.8 s, 19 893 messages. Same layout but `/camera/rgb/image_color` is **`rgb8`** (the other sequence is `bgr8` — the
channel-order rule must key on the encoding, per file), K = the Kinect defaults (fx 525, cx 319.5), 613 RGB vs 595 depth
messages, depth NaN 16.9 %, p50 1.08 m, `/imu` at 497 Hz, same 4-hop TF chain.

### NVIDIA r2b_cafe — rosbag2 sqlite3 with zstd, D455, no poses
5.0 s, 1 993 messages, **no embedded message definitions** (a default typestore is mandatory). Topics WITHOUT a leading
slash: `d455_1_rgb_image` `rgb8` 1280×720 at 30 Hz, `d455_1_depth_image` **`32FC1` metres with 0.0 = invalid (16.4 %),
no NaN**, p50 2.1 m; `d455_1_rgb_camera_info` (fx 641.4, cx 647.9, `plumb_bob`, 8 coefficients) vs
`d455_1_depth_camera_info` (fx 654.2, cx 651.6, `distortion_model: "pinhole"`) on different optical frames
(`D455_1:rgb` vs `D455_1:depth`) → **depth is NOT aligned to RGB** and must be detected from the two K matrices /
frame ids and refused. IR pair `mono8`, HAWK stereo `rgb8` 1920×1200 (`rational_polynomial`), XT32 lidar. Only
`/tf_static` (`base_link → D455_1 → D455_1:{rgb,depth,…}`) — **no pose stream**. Log time runs ~36–45 ms after the
header stamps (the sensor clock must use header stamps).

### NVIDIA r2b_hope — the degenerate one
5 messages, `/image` `bgr8` 1920×1080 with **header stamp 0 and an empty frame_id**, no camera info, no TF.

## What the corpus demands of task 1's reader (folded into the execution plan, task 1)
1. Default typestore (ROS 2 Humble) when a rosbag2 carries no type definitions, with the assumption logged.
2. Topic-name normalisation (relative names) and frame-id normalisation (ROS 1 leading slashes).
3. `32FC1` invalid pixels: NaN (TUM) and 0.0 (D455) both → NaN.
4. Nearest-stamp RGB/depth pairing with a tolerance (TUM's tool uses 20 ms) and a count of unpaired frames.
5. Alignment check from `CameraInfo` + frame ids; unaligned depth refused with the reason (v1).
6. TF composition over multiple hops at the image stamp with interpolation on moving hops; `tf/msg/tfMessage` accepted.
7. `/tf_static`-only bags: explicit "no pose source" refusal naming the topics searched.
8. Stamp 0 / empty frame_id: counted skip, no crash.
9. Distortion recorded, warned when non-zero, not undistorted in v1.
10. Sensor clock on header stamps, never on log time.

## Not fetched (and why)
OpenLORIS-Scene (registration form — the founder's action; licence unverified), ARKitScenes / ScanNet++ (folders, need a
converter onto bag layout v1; non-commercial / unverified terms), Bonn / ETH3D (other sensors, folder formats). A public
bare MCAP with indoor RGB-D + poses was not found (Foxglove's nuScenes sample URL is dead: 404 on 2026-09-25).
