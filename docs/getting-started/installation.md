# Installation

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (tested on RTX 3080, RTX 5090)
- RGB-D camera (Intel RealSense D435i tested)
- SLAM system providing poses (RTAB-Map, ORB-SLAM3)

!!! note "WSL2 Users"
    Tested on WSL2 Ubuntu 22.04 with RTAB-Map. WSL2 has USB passthrough limitations — you may need [usbipd-win](https://github.com/dorssel/usbipd-win) for camera access.

---

## Install RTSM

```bash
git clone https://github.com/calabi-inc/rtsm.git
cd rtsm
# Install dependencies
pip install -r requirements.txt

# Install RTSM in editable mode
pip install -e .
```

---

## Download Models

### FastSAM Weights

```bash
mkdir -p model_store/fastsam
# Download FastSAM-x.pt to model_store/fastsam/
```

Download from: [FastSAM Releases](https://github.com/CASIA-IVA-Lab/FastSAM/releases)

### CLIP Weights

CLIP weights are auto-downloaded on first run, or you can pre-fetch:

```bash
python scripts/fetch_clip.py
```

---

## Verify Installation

```bash
python -c "import rtsm; print(rtsm.__version__)"
```

---

## Next Steps

- [Quick Start](quick-start.md) — Run your first query
- [Configuration](configuration.md) — Customize for your setup
