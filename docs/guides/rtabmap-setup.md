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
