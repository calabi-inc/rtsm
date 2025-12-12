# RealSense Setup

This guide covers setting up Intel RealSense D435i for use with RTSM.

---

## Hardware

**Tested cameras**:

- Intel RealSense D435i (recommended)
- Intel RealSense D435
- Intel RealSense D455

The D435i includes an IMU, which can improve SLAM tracking.

---

## Installation

### Ubuntu / WSL2

```bash
# Add Intel repo
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"

# Install
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev
```

### Python SDK

```bash
pip install pyrealsense2
```

### Verify Installation

```bash
realsense-viewer
```

This should open the RealSense GUI showing RGB and depth streams.

---

## Running the Scanner

RTSM includes a RealSense capture utility:

```bash
# From rtsm-realsense-scanner repo
python realsense_scanner.py --zmq-pub tcp://*:5555
```

### Options

| Flag | Description |
|------|-------------|
| `--width` | Frame width (default: 640) |
| `--height` | Frame height (default: 480) |
| `--fps` | Frame rate (default: 30) |
| `--zmq-pub` | ZeroMQ publish address |

### Message Format

Published frames:

```json
{
  "timestamp": 1705312200.123,
  "frame_id": 12345,
  "rgb": "<base64 encoded JPEG>",
  "depth": "<base64 encoded 16-bit PNG>",
  "intrinsics": {
    "fx": 615.0,
    "fy": 615.0,
    "cx": 320.0,
    "cy": 240.0
  }
}
```

---

## Configuration

### Camera Settings

For indoor use:

```python
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
```

### Depth Filtering

Enable post-processing for cleaner depth:

```python
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)

temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

depth_frame = spatial.process(depth_frame)
depth_frame = temporal.process(depth_frame)
depth_frame = hole_filling.process(depth_frame)
```

### Align Depth to Color

Important for correct projection:

```python
align = rs.align(rs.stream.color)
aligned_frames = align.process(frames)
```

---

## WSL2 Setup

WSL2 requires USB passthrough via [usbipd-win](https://github.com/dorssel/usbipd-win).

### Windows Side (PowerShell Admin)

```powershell
# Install usbipd
winget install usbipd

# List devices
usbipd wsl list

# Attach RealSense (find the Bus ID from list)
usbipd wsl attach --busid 2-3
```

### WSL2 Side

```bash
# Verify device is visible
lsusb | grep Intel

# May need permissions
sudo chmod 666 /dev/video*
```

### Persistent Attachment

Create a script to auto-attach on boot:

```powershell
# attach-realsense.ps1
usbipd wsl attach --busid 2-3
```

---

## Troubleshooting

### "No device connected"

1. Check USB connection
2. Try different USB port (USB 3.0 required)
3. WSL2: Verify usbipd attachment

### Poor Depth Quality

- Clean the IR sensors
- Ensure adequate lighting (not too bright/dark)
- Check for reflective surfaces
- Enable depth filtering (see above)

### Frame Drops

- Reduce resolution or FPS
- Check USB bandwidth (avoid hubs)
- Close other applications using USB

---

## Next Steps

- [RTAB-Map Setup](rtabmap-setup.md) — SLAM configuration
- [Quick Start](../getting-started/quick-start.md) — Run RTSM
