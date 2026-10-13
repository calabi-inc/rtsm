# RTAB-Map Setup

This guide covers setting up [RTAB-Map](https://github.com/introlab/rtabmap) to provide SLAM poses for RTSM.

---

## Overview

RTSM needs camera poses (position + orientation) to project 2D detections into 3D space. RTAB-Map is a popular open-source visual SLAM system that works well with RGB-D cameras.

```
RealSense → RTAB-Map → Poses → ZeroMQ → RTSM
```

---

## Installation

### Ubuntu / WSL2

```bash
sudo apt update
sudo apt install ros-humble-rtabmap-ros
```

Or build from source:

```bash
git clone https://github.com/introlab/rtabmap.git
cd rtabmap/build
cmake ..
make -j$(nproc)
sudo make install
```

### Windows

Download from [RTAB-Map releases](https://github.com/introlab/rtabmap/releases).

---

## Running with RealSense

### Option 1: Standalone (No ROS)

```bash
rtabmap-realsense
```

This opens the RTAB-Map GUI with RealSense input.

### Option 2: ROS 2

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_rect_raw \
  camera_info_topic:=/camera/color/camera_info
```

---

## ZeroMQ Bridge

RTSM expects poses via ZeroMQ. Use the bridge utility:

```bash
# From the rtsm-rtabmap-bridge repo
python rtabmap_zmq_bridge.py --rtabmap-addr localhost:5555 --zmq-pub tcp://127.0.0.1:6000
```

### Message Format

The bridge publishes pose messages as `[topic, json]` multipart frames; the
subscriber (`rtsm/io/zeromq.py`) parses this shape:

```json
{"stamp_ms": 1705312200123, "T_wc": [x, y, z, roll, pitch, yaw]}
```

`rtabmap.tracking_pose` (~30 Hz) carries `stamp_ms`; `rtabmap.kf_pose` carries
`kf_id` and should carry `stamp_ms` too. **Clock contract:** `stamp_ms` is
milliseconds on the same clock as `camera.rgbd`'s `ts_ns` (unix on the
reference bridge); frames are paired within 30 ms of the pose stamp. A
`kf_pose` without a stamp inherits the last tracking stamp (counted in the
subscriber's `kf_stamps_inherited`), so the keyframe is attached to the newest
image rather than the node's own; if that image was already admitted as a
non-keyframe and no newer frame exists yet, the keyframe re-processes it (one
duplicate GPU pass per such keyframe). Add `stamp_ms` on the bridge to remove
both the lag and the duplicate. A tracking stamp that jumps back by more than 5 s (bag loop, bridge
restart) starts a new `frame_epoch` on the robot pose.

### The two correction topics

A SLAM emits two different corrections, and the bridge publishes both:

- **`rtabmap.map_correction`** is the *pose correction*: RTAB-Map's map→odom transform (`Statistics::mapCorrection()`), `{"T_w_odom": {"t": [x, y, z], "q": [qx, qy, qz, qw]}, "map_id": int, "ts_ns": int}`, once per RTAB-Map update. The bridge's `tracking_pose` is the **odometry** pose (`OdometryEvent::pose()`, despite its `frame: "world"` label), while `kf_pose` comes from the optimised graph and is already in the map frame. RTSM composes `T_w_odom @ T_odom_c` onto every tracking pose at receive time (`io.apply_map_correction`, default true), so objects seen on non-keyframe frames land in the same frame as the keyframes; with no message yet the identity applies. The message's `map_id` is RTAB-Map's map id, which RTSM takes as the session id `rtabmap-map-<id>`: a new map (an odometry reset, `RGBD/NewMapOdomChangeDistance`, `trigger_new_map`) starts a new session (see [Sessions](../api/rest-api.md#sessions)). `/stats.source.map_correction` counts received / applied / malformed messages and the current map id.
- **`rtabmap.kf_pose_update`** is the *map correction*: a node whose optimised pose moved (loop closure, graph optimisation), `{"kf_id": int, "T_wc": [x, y, z, roll, pitch, yaw]}`, one message per node after each optimisation. RTSM translates the node id to the pipeline's keyframe (the node id rides on the keyframe packet) and coalesces the burst into one correction batch, flushed by the next `map_correction` or after 200 ms, so the working memory's objects, the TSDF, the pose history and the session registry are refined exactly as by a Lens `pose_corrections` batch. Before this, only the bridge's thumbnail cloud was re-posed.

Not emitted by the bridge today: a *link into a previous map* (`info.loop_closure_id` naming a node whose map id differs from the current one). Emitting it as a correction of kind `relocalized_in_prior_map` with `map_ref` = that map id would let the session registry merge the two sessions on its own; until then `POST /sessions/{key}/merge` does it by hand.

---

## Configuration

### RTAB-Map Parameters

For indoor robotics, these defaults work well:

```ini
Rtabmap/DetectionRate=2
Vis/MinInliers=15
RGBD/OptimizeMaxError=3.0
Mem/STMSize=30
```

### RTSM Configuration

In `config/rtsm.yaml`:

```yaml
io:
  receiver: zeromq
  camera_endpoint: tcp://172.27.240.1:5555   # D435i RGB-D frames
  rtabmap_endpoint: tcp://127.0.0.1:6000     # RTABMap pose topics
```

---

## Troubleshooting

### "No poses received"

1. Check RTAB-Map is running and tracking
2. Verify ZMQ bridge is connected: `netstat -an | grep 6000`
3. Check for firewall blocking localhost ports

### Drift / Poor Tracking

- Ensure adequate lighting
- Add visual features to the environment (avoid blank walls)
- Reduce camera motion speed
- Enable loop closure in RTAB-Map

### WSL2 USB Issues

WSL2 doesn't natively support USB. Use [usbipd-win](https://github.com/dorssel/usbipd-win):

```powershell
# PowerShell (admin)
usbipd wsl list
usbipd wsl attach --busid <BUSID>
```

---

## Throttle and pairing window

The subscriber's non-keyframe throttle reads `ingest.nonkf_min_interval_s`
(default 0.5 s, the value it had hardcoded before P1 task 6), on the ingest
clock like the websocket and replay receivers. Its pairing window, how long an
encoded camera frame waits for its pose, is `ingest.pair_window_s` (2 s) with
a frame cap of `ceil(pair_window_s × pair_window_fps × 1.5)` (90 at the
defaults); see the [configuration guide](../getting-started/configuration.md#ingest-clock-admission-timing).

## Next Steps

- [RealSense Setup](realsense-setup.md) — Camera configuration
- [Configuration](../getting-started/configuration.md) — RTSM settings
