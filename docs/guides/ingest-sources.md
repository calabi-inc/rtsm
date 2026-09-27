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
| Registry | `rtsm/io/sources.py` | `make_source(name, cfg, ctx, **options)` builds a source by name from one `SourceContext`. Built-ins `websocket`, `zeromq`, `replay`; plug-ins via the `rtsm.sources` entry-point group. |

`io.receiver` in the config names the source (`websocket` or `zeromq`, or a plug-in name); `--replay <dir>` selects the replay source regardless. See [Configuration → I/O & Receiver](../getting-started/configuration.md#io--receiver).

## What an adapter does, and does not do

An adapter owns the transport (sockets, files, pairing of poses with camera frames on a bridge that publishes them separately) and produces:

- one `RawFrame` per candidate frame: header fields (source frame id, sensor stamp, sender wall stamp, tracking state, the source's own keyframe flag if it has one), the **still-encoded** RGB / depth / confidence payloads with their encodings and declared sizes, intrinsics at RGB resolution (`codecs.rescale_intrinsics`), and the pose as the transport delivered it plus `pose_format` / `pose_convention` tags;
- `PoseSample` events for poses that arrive separately from frames (`IngestFrontEnd.pose_event`), so the receive-time robot pose and the pose ledger see every pose at input rate;
- `FrameCorrection` events for retroactive pose corrections (loop closure, relocalisation).

It never decodes pixels, never applies keyframe or throttle logic, never touches the ingest queue. Those belong to the front-end; that is what keeps the session1 anchor (`ad6f71a5b89c8506`, 124 objects / 65 confirmed at 53 frames) meaningful across sources.

The adapter then calls `fe.admit(raw)` (or `fe.offer(raw)` = admit + enqueue). `admit` returns the `FramePacket` to enqueue, `None` when the chain dropped the frame (the drop line is already written), and re-raises after writing a `parse_error` line when the pose or a payload fails to parse: the adapter logs and continues, as the receivers always did. Transport-level drops the front-end never saw (malformed framing, no camera frame to pair with) are reported with `fe.reject(reason, ...)` so the frame-flow trace stays complete.

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
- **The session1 anchor** (`eval/baselines/`): the headless replay of the reference recording must reproduce the anchor fingerprint and the per-line receiver / dequeue trace. Run it after any change to the front-end, a policy, or a codec.

A change that moves any of these is a behaviour change, not a refactor; it needs a new anchor and a note in the changelog.
