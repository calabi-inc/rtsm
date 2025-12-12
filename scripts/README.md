## RTSM Scripts

### Dataset Replay

`replay_dataset.py` replays frames from `test_dataset` using original timing derived from the numeric timestamps in filenames.

Expected dataset layout:

```
test_dataset/
  rgb/
    1754989062.627478.png
    1754989062.727512.png
    ...
  depth_npy/
    1754989062.627478.npy   # float32 depth in meters, HxW
    1754989062.727512.npy
    ...
```

Basic usage:

```bash
python scripts/replay_dataset.py --dataset test_dataset --speed 1.0 --show
```

Options:

- `--dataset <path>`: Root containing `rgb/` and `depth_npy/` (default: `test_dataset`)
- `--speed <factor>`: Playback speed (e.g., `0.5` = half, `2.0` = double; default `1.0`)
- `--max-delay <seconds>`: Cap the per-frame waiting time (useful when catching up)
- `--loop`: Repeat playback
- `--show`: Open a simple RGB | Depth preview window (requires OpenCV)

ZeroMQ options:

- `--zmq`: Enable ZeroMQ publishing (PUB)
- `--zmq-endpoint <addr>`: Bind address for the PUB socket (default: `tcp://127.0.0.1:6001`)
- `--zmq-rgb-topic <topic>`: Topic for RGB frames (default: `rgb`)
- `--zmq-depth-topic <topic>`: Topic for depth frames (default: `depth`)

Notes:

- Timestamps are parsed from filename stems; depth frames are matched by identical timestamp.
- If some depth frames are missing, the script will still replay RGB and print a warning.
- The preview shows RGB next to a normalized depth visualization; press `ESC` to exit.

Examples:

```bash
# Real-time speed, single pass, no preview
python scripts/replay_dataset.py --dataset test_dataset

# 2x speed with preview
python scripts/replay_dataset.py --dataset test_dataset --speed 2.0 --show

# Loop forever and cap wait per frame to 50 ms
python scripts/replay_dataset.py --dataset test_dataset --loop --max-delay 0.05
```

#### Publish via ZeroMQ

Start a simple subscriber (prints received multipart frames):

```bash
python zeromq/sub.py
```

Run the replay and publish frames over ZeroMQ:

```bash
python scripts/replay_dataset.py \
  --dataset test_dataset \
  --zmq \
  --zmq-endpoint tcp://127.0.0.1:6001
```

Change topic filters in the subscriber by editing `zeromq/sub.py`:

```python
s.setsockopt(zmq.SUBSCRIBE, b"rgb")   # or b"depth" to receive depth only
```

Message format (multipart ZeroMQ):

- `[topic, ts_nanos, payload]`
- `topic`: `b"rgb"` or `b"depth"` (configurable via flags)
- `ts_nanos`: ASCII-encoded integer nanoseconds since epoch
- `payload` for `rgb`: PNG-encoded bytes (RGB8)
- `payload` for `depth`: compressed NPZ bytes with key `depth` (float32 meters, HxW)


