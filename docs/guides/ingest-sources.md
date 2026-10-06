# Ingest Sources

RTSM takes frames from a *source*: a transport adapter that turns bytes from somewhere (a Calabi Lens websocket, a RealSense + RTAB-Map ZeroMQ bridge, a recording) into frames and poses. Since the ingest front-end extraction, every source runs the **same admission chain**, so what the determinism gates and the benchmarks say about one source holds for all of them, and a new source is a small adapter, not a fourth copy of the receiver.

```text
transport bytes ──adapter──▶ RawFrame / PoseSample ──front-end──▶ FramePacket ──▶ ingest queue ──▶ pipeline
                 (rtsm/io/websocket.py,            (rtsm/io/ingest_frontend.py,
                  zeromq.py, replayer.py,           one implementation)
                  your plug-in)
```

## The pieces

| Layer | Module | Job |
|---|---|---|
| Contracts | `rtsm/io/contracts.py` | The versioned seam: `RawFrame` (a `FrameHeader` + still-encoded payloads), `PoseSample`, `FrameCorrection`, `TrackingStatus`, the `Source` protocol, `SourceContext`. `CONTRACT_VERSION = 1`. |
| Codecs | `rtsm/io/codecs.py` | Pixels and poses: JPEG / PNG / BGRA / NV12 RGB, `uint16_mm` / `float32_m` / PNG depth, confidence maps, intrinsics rescaling, ARKit and RTAB-Map pose formats, the ARKit→OpenCV convention flip. |
| Front-end | `rtsm/io/ingest_frontend.py` | The chain: tracking filter → pose parse → receive-time pose mailbox → depth decode + pose ledger → keyframe rule → non-keyframe throttle → lane admission (admit-before-decode) → decode on admit (memoised) → clearance / confidence filter → `FramePacket` → enqueue → callbacks → frame-flow trace. |
| Policy | `FrontEndPolicy` (same module) | The per-source flavour of the chain. Two ship: `WEBSOCKET_POLICY` (minted keyframes, pose rides with the frame, depth decoded before admission) and `ZEROMQ_POLICY` (SLAM keyframes, poses as separate events, decode after admission, repeat-stamp dedup, receiver-minted epochs). |
| Registry | `rtsm/io/sources.py` | `make_source(name, cfg, ctx, **options)` builds a source by name from one `SourceContext`. Built-ins `websocket`, `zeromq`, `replay`, `bag` (see [Reading Bags](bags.md)); plug-ins via the `rtsm.sources` entry-point group. |

`io.receiver` in the config names the source (`websocket` or `zeromq`, or a plug-in name); `--replay <dir>` selects the replay source regardless. See [Configuration → I/O & Receiver](../getting-started/configuration.md#io--receiver).

## What an adapter does, and does not do

An adapter owns the transport (sockets, files, pairing of poses with camera frames on a bridge that publishes them separately) and produces:

- one `RawFrame` per candidate frame: header fields (source frame id, sensor stamp, sender wall stamp, tracking state, the source's own keyframe flag if it has one), the **still-encoded** RGB / depth / confidence payloads with their encodings and declared sizes, intrinsics at RGB resolution (`codecs.rescale_intrinsics`), and the pose as the transport delivered it plus `pose_format` / `pose_convention` tags;
- `PoseSample` events for poses that arrive separately from frames (`IngestFrontEnd.pose_event`), so the receive-time robot pose and the pose ledger see every pose at input rate;
- `FrameCorrection` events for retroactive pose corrections (loop closure, relocalisation).

It never decodes pixels, never applies keyframe or throttle logic, never touches the ingest queue. Those belong to the front-end; that is what keeps the session1 anchor (`ad6f71a5b89c8506`, 124 objects / 65 confirmed at 53 frames) meaningful across sources.

The adapter then calls `fe.admit(raw)` (or `fe.offer(raw)` = admit + enqueue). `admit` returns the `FramePacket` to enqueue, `None` when the chain dropped the frame (the drop line is already written), and re-raises after writing a `parse_error` line when the pose or a payload fails to parse: the adapter logs and continues, as the receivers always did. Transport-level drops the front-end never saw (malformed framing, no camera frame to pair with) are reported with `fe.reject(reason, ...)` so the frame-flow trace stays complete.

## ROS 2 (live)

`io.receiver: ros2` (or `python -m rtsm --ros2`) subscribes to a live ROS 2 graph. It is the bag reader's twin on the same ingest front-end: the topic roles are found by the [bag reader's rules](bags.md#what-the-reader-needs-and-how-it-finds-it) or set under `io.ros2.topics`, the `sensor_msgs` encodings go onto the codec layer without decoding, RGB and depth are paired by header stamp, the camera pose is composed from `/tf` (or odometry) at the image stamp, depth must be registered to the RGB, and the refusals carry the same codes. A frame built live is the frame the bag reader builds from a recording of the same stream; `tests/test_ros2_source.py` pins that on `session1_bag`.

**Environment.** rclpy only exists inside a sourced ROS 2 environment on Linux (Humble on Ubuntu 22.04 with Python 3.10, Jazzy on 24.04 with 3.12). Give an isolated venv a view of the ROS packages through a `.pth` file rather than `--system-site-packages`: the latter also pulls in apt's numpy and scipy and anything under `~/.local`, and a numpy-1 build next to torch's numpy 2 ends in `_ARRAY_API not found`.

```bash
source /opt/ros/humble/setup.bash                                  # or jazzy
python3 -m venv ~/rtsm-env                                         # add --without-pip and run get-pip.py if ensurepip is missing
source ~/rtsm-env/bin/activate
printf '%s
' /opt/ros/$ROS_DISTRO/local/lib/python3.*/dist-packages /opt/ros/$ROS_DISTRO/lib/python3.*/site-packages   > "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/ros.pth"
export PYTHONNOUSERSITE=1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "rtsm[gpu,eval]"
python -c "import rclpy, torch, rtsm; print(torch.cuda.is_available())"
```

Anywhere else the source refuses at start with a one-line hint. The Windows development box runs none of this; WSL2 with a ROS 2 distribution does. Live runs use `ingest.policy: latest` (the default) or `legacy`; `lossless` is replay-only. To replay one of RTSM's own MCAP recordings into a Humble graph with `ros2 bag play`, convert it to sqlite3 storage with metadata version 5 first: `rosbags-convert --src <bag> --dst <out> --dst-storage sqlite3 --dst-version 5`.

**Before the models load: `rtsm ros2 probe`.** The two classic silent failures of a ROS 2 subscriber are a QoS mismatch (the callback never fires) and a TF chain that never reaches the camera frame. The probe listens for a few seconds and prints what the node would use:

```text
$ rtsm ros2 probe --seconds 5
ros2 probe (5.0 s): OK
topics:
  rgb        /camera/color/image_raw  [image topic matching color|rgb|image_raw and not depth|ir|mono|left|right]  subscribe reliable/volatile depth 100  (offered: rosbag2_player:reliable)
  depth      /camera/depth/image_rect_raw  [image topic matching depth (aligned_depth_to_color preferred)]  subscribe reliable/volatile depth 100  (offered: rosbag2_player:reliable)
  rgb_info   /camera/color/camera_info  [...]
  tf         /tf  [TF message topic]  subscribe reliable/volatile depth 200  (offered: rosbag2_player:reliable)
camera info: rgb seen, depth not seen
tf: roots ['world']; 1 hops; chain world -> camera_color_optical_frame; pose tf
registration: no depth CameraInfo: registration assumed
```

Exit code 0 when the stream is usable, 1 with the refusal otherwise, 2 without rclpy. `--json` prints everything as data; `--topic ROLE=TOPIC` overrides a role for the probe only.

**Probe reference.**

| option | meaning |
|---|---|
| `--seconds N` | how long to listen for CameraInfo and TF after discovery (default 5) |
| `--qos {auto, reliable, best_effort}` | the subscription reliability to evaluate (default `auto`) |
| `--topic ROLE=TOPIC` | override a role for this probe only (repeatable) |
| `--world-frame`, `--camera-frame`, `--assume-aligned` | as the `io.ros2` keys |
| `--json` | the full result as data: `topics`, `msgtypes`, `qos` (chosen + offered per role), `advertised`, `camera_info_seen`, `tf` (roots, hops, chain, world/camera frame, pose kind), `registration`, `refusal`, `ok` |

Exit codes: 0 the stream is usable, 1 not usable (the refusal is printed), 2 rclpy not importable.

**QoS.** `io.ros2.qos: auto` reads the publishers' offered QoS per topic and subscribes reliable when any publisher is reliable, best-effort otherwise; `reliable` and `best_effort` force it. `tf_static` always subscribes transient-local. Image subscriptions keep a depth of 100 so a reliable publisher is not dropped while the lane admits; TF keeps 200.

**Threads and timing.** The executor only appends messages to a queue; a worker thread pairs, looks the pose up, admits and enqueues, so a lossless lane that blocks never stalls the executor. A pair whose TF has not arrived waits up to `io.ros2.tf_wait_s` (0.5 s) with later frames behind it, then counts as `pose_missing`. Pairs resolved before the camera info, the world frame and the registration check are known are held (256) and flushed in order. Discovery waits `discovery_timeout_s` for the RGB, depth and CameraInfo topics to be advertised and `ready_timeout_s` for a CameraInfo message and a moving TF chain, then refuses with the reasons.

**What it is not.** Not an `ament` package, no launch files, no publishing of RTSM's outputs onto ROS topics, no ROS 2 container in CI. Those are the next step when a live-ROS user needs them; the source's stats (`frames_seen`, `paired`, `unpaired_rgb`, `pose_missing`, `enqueued`, the chosen QoS) are on `/stats` like the bag source's.

## Writing a source

```python
# my_package/rtsm_source.py
import threading, time
import numpy as np
from rtsm.io.contracts import EncodedImage, FrameHeader, RawFrame, SourceContext, POSE_FMT_PREPARED
from rtsm.io.ingest_frontend import IngestFrontEnd, WEBSOCKET_POLICY

class BagSource:
    name = "mybag"

    def __init__(self, cfg: dict, ctx: SourceContext, **options):
        self._path = options.get("recording_dir") or cfg["io"]["mybag"]["path"]
        self._fe = IngestFrontEnd(
            source=self.name, policy=WEBSOCKET_POLICY, ingest_queue=ctx.ingest_queue,
            throttle_clock=ctx.clock_mode, keyframe_every_n=ctx.keyframe_every_n,
            nonkf_min_interval_s=ctx.nonkf_min_interval_s,
            require_tracking_normal=ctx.require_tracking_normal,
            confidence_threshold=ctx.confidence_threshold,
            pose_sink=ctx.pose_sink, event_sink=ctx.event_sink, ledger_sink=ctx.ledger_sink,
            latency_analytics=ctx.latency_analytics,
            on_camera_frame=ctx.on_camera_frame, on_keyframe=ctx.on_keyframe,
        )
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True, name=self.name)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def liveness(self) -> dict:
        t = self._thread
        return {"alive": bool(t and t.is_alive()), **self._fe.liveness()}

    def _loop(self):
        for msg in read_my_bag(self._path):                 # your transport
            if self._stop.is_set():
                break
            self._fe.last_rx_mono = time.monotonic()
            raw = RawFrame(header=FrameHeader(
                source=self.name, seq=msg.seq, t_sensor_ns=msg.stamp_ns, t_wall_utc_s=None,
                tracking_state="normal", keyframe_hint=None,
                rgb=EncodedImage(msg.jpeg, "jpeg", msg.w, msg.h),
                depth=EncodedImage(msg.depth_u16_bytes, "uint16_mm", msg.w, msg.h, 0.001),
                intrinsics=msg.intrinsics,
                pose_raw=(msg.t_wc, msg.q_wc_xyzw), pose_format=POSE_FMT_PREPARED,
            ))
            try:
                self._fe.offer(raw)
            except Exception as e:                          # parse_error line already written
                logging.getLogger(__name__).warning("bad frame %s: %s", msg.seq, e)
```

Register it:

```toml
# your package's pyproject.toml
[project.entry-points."rtsm.sources"]
mybag = "my_package.rtsm_source:BagSource"
```

then `io.receiver: mybag`. The factory signature is `factory(cfg, ctx, **options) -> Source`; a class with that constructor works. In-process registration (`rtsm.io.sources.register_source("mybag", BagSource)`) does the same without packaging. A plug-in name that collides with a built-in is ignored with a warning.

The runner passes these options today: `recording_dir` and `replay_speed` (from `--replay` / `--replay-speed`), `pair_window_s` and `pair_window_frames` (from `ingest.*`, used by the ZeroMQ source). Read what you need, ignore the rest.

### Choosing a policy

- Your source has no keyframe notion of its own and one pose per frame (a phone, a recording, most bags): `WEBSOCKET_POLICY`.
- Your source is a SLAM bridge that publishes poses and keyframes separately from camera frames, so you pair them yourself: `ZEROMQ_POLICY`, and drive `pose_event` / `is_repeat_stamp` / `throttle_due` / `reject` as `rtsm/io/zeromq.py` does.
- Anything else: `dataclasses.replace(WEBSOCKET_POLICY, ...)`. Each flag is documented on `FrontEndPolicy`. A new flavour is a new behaviour, so gate it (see below) before relying on it.

### Encodings the codec layer understands

| Payload | `encoding` values |
|---|---|
| RGB | `jpeg`, `png`, `bgra`, `nv12`, `raw_bgr` (an ndarray) |
| Depth | `uint16_mm`, `float32_m`, `png_uint16` (zeros → invalid), `png_uint16_raw` (RTAB-Map bridge: zeros kept), `raw_depth_m` (an ndarray) |
| Confidence | a uint8 map (0 / 1 / 2) at the declared size |
| Pose | `matrix4x4_col_major` (ARKit 16 floats), `quat_translation` (7 floats), `rtabmap_euler` (`[x, y, z, roll, pitch, yaw]`), `prepared` (`(t_wc, q_wc_xyzw)` from the adapter) |
| Convention | `opencv` (identity) or `arkit` (flipped once at ingest) |

An unknown value raises `codecs.UnsupportedEncoding`, which the front-end reports as a `parse_error` line.

## How the behaviour is pinned

- **Golden traces** (`tests/test_ingest_golden.py`): two fixed message streams, one Lens-shaped and one bridge-shaped, recorded from the receivers *before* the extraction. Every receiver line, pose-mailbox call, pose-ledger line, packet digest and end state must match byte for byte. Regenerate only with `RTSM_WRITE_GOLDEN=1` and a reason in the commit.
- **Unit tests** (`tests/test_ingest_frontend.py`): the chain's order (tracking filter before pose parse, refusal before decode, throttle before enqueue), the two policies, the throttle, the codecs, the registry.
- **The session1 anchor** (124 objects / 65 confirmed at 53 processed frames, fingerprint `ad6f71a5b89c8506`, 86 dequeue + 240 receiver lines): the headless replay of the reference recording must reproduce the anchor fingerprint and the per-line receiver / dequeue trace. Run it after any change to the front-end, a policy, or a codec.

A change that moves any of these is a behaviour change, not a refactor; it needs a new anchor and a note in the changelog.
