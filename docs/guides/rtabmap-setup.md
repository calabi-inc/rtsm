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
python rtabmap_zmq_bridge.py --rtabmap-addr localhost:5555 --zmq-pub tcp://*:5556
```

### Message Format

The bridge publishes pose messages:

```json
{
  "timestamp": 1705312200.123,
  "position": [1.2, 0.4, 2.1],
  "orientation": [0.0, 0.0, 0.0, 1.0],
  "frame_id": 12345
}
```

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
  zmq_slam_addr: "tcp://localhost:5556"
```

---

## Troubleshooting

### "No poses received"

1. Check RTAB-Map is running and tracking
2. Verify ZMQ bridge is connected: `netstat -an | grep 5556`
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

## Next Steps

- [RealSense Setup](realsense-setup.md) — Camera configuration
- [Configuration](../getting-started/configuration.md) — RTSM settings
