"""
The ingest front-end (Gate 4.5 plan, P3 task 0.5): ONE implementation of the
policy chain every source goes through.

    tracking filter -> pose parse (codec) -> receive-time pose mailbox -> depth
    decode + P2 pose ledger -> keyframe rule -> non-keyframe throttle (wall or
    sensor clock) -> lane admission (admit-before-decode) -> decode on admit
    (codec, memoised) -> clearance / confidence filter -> FramePacket ->
    enqueue -> callbacks -> frame-flow trace + analytics.

Extracted from the websocket receiver, whose chain the P1 tasks made precise
(admit-before-decode, sensor clock, lanes, pose mailbox) and gated (G1-A /
G1-B). The ZeroMQ receiver's variant of the same chain is expressed as a
``FrontEndPolicy`` (source keyframes instead of minted ones, poses as
separate events, decode after admission, repeat-stamp dedup, receiver-minted
epochs). Adapters (``rtsm/io/websocket.py``, ``replayer.py``, ``zeromq.py``,
the P3 bag readers) only turn transport bytes into ``RawFrame`` / ``PoseSample``
and call ``admit`` / ``enqueue`` / ``reject`` / ``pose_event``.

Behaviour is pinned by the session1 anchor and by ``tests/test_ingest_golden.py``
(two golden traces recorded from the receivers before this extraction): the
ORDER of the steps below, where the frame counter increments, where each
analytics counter fires and which fields each trace line carries are all
part of that contract. Change them only with a new anchor.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from rtsm.core.datamodel import FramePacket, IngestMeta, PoseStamped, TimeBundle
from rtsm.evaluation.event_log import (
    RX_DROPPED, RX_DUPLICATE_TS, RX_ENQUEUED, RX_PARSE_ERROR, RX_QUEUE_FULL, RX_THROTTLE, RX_TRACKING,
    TS_NORMAL, PoseEvent, ReceiverEvent,
)
from rtsm.io import codecs
from rtsm.io.contracts import PoseSample, RawFrame
from rtsm.io.ingest_lanes import KF_MINTED, KF_SOURCE

logger = logging.getLogger(__name__)

# A tracking stamp that goes back by more than this within the stream is a
# restarted source (bag loop, bridge restart) -> new receiver-minted epoch.
# Same constant as SensorClock.rebase_after_s so the clock and the mailbox
# agree on what a restart is.
POSE_EPOCH_REBASE_S = 5.0


# ───────────────────────────── throttle ─────────────────────────────

class NonKfThrottle:
    """Min-interval throttle for non-keyframes, on the wall or the sensor clock
    (``ingest.clock``). The websocket receiver's ``_admit_nonkf`` and the ZeroMQ
    receiver's ``_nonkf_due`` / ``_stamp_nonkf`` were the same algorithm; this
    is that algorithm once.

    sensor mode: compare the frame's sensor stamp with the last ADMITTED
    non-keyframe's; a negative delta (new session / restarted clock) is due
    and re-stamps; a missing / non-positive stamp falls back to wall.
    wall mode: compare process-monotonic time.
    """

    def __init__(self, clock: str = "wall", interval_s: float = 0.5) -> None:
        self.clock = "sensor" if str(clock).lower() == "sensor" else "wall"
        self.interval_s = float(interval_s)
        self.last_admit_mono: float = 0.0
        self.last_admit_sensor_ns: Optional[int] = None

    def sensor_active(self, ts_ns: Any) -> bool:
        if self.clock != "sensor" or not ts_ns:
            return False
        try:
            return int(ts_ns) > 0
        except (TypeError, ValueError):
            return False

    def due(self, ts_ns: Any) -> bool:
        """Would a non-keyframe with this stamp pass? No side effect."""
        if self.sensor_active(ts_ns):
            last = self.last_admit_sensor_ns
            return not (last is not None and 0 <= (int(ts_ns) - last) < int(self.interval_s * 1e9))
        return (time.monotonic() - self.last_admit_mono) >= self.interval_s

    def stamp(self, ts_ns: Any) -> None:
        """Record an ATTEMPT (a refused frame still burns the window)."""
        if self.sensor_active(ts_ns):
            self.last_admit_sensor_ns = int(ts_ns)
        else:
            self.last_admit_mono = time.monotonic()

    def admit(self, ts_ns: Any) -> bool:
        """``due`` then ``stamp`` when due (the websocket receiver's one-call form)."""
        if not self.due(ts_ns):
            return False
        self.stamp(ts_ns)
        return True

    def reset(self) -> None:
        self.last_admit_mono = 0.0
        self.last_admit_sensor_ns = None


# ───────────────────────────── policy ─────────────────────────────

@dataclass(frozen=True)
class FrontEndPolicy:
    """The per-source flavour of the chain. Two shipped flavours; a reader picks
    one or defines its own -- adapters never carry the logic themselves."""
    keyframe_rule: str = "minted"              # minted (frame_count == 1 or % keyframe_every_n) | source (header.keyframe_hint)
    pose_with_frame: bool = True               # True: pose sink + pose ledger run inside admit(); False: poses arrive via pose_event()
    depth_before_admission: bool = True        # decode depth (and depth_valid_frac) before the keyframe rule / throttle / refusal
    throttle_in_admit: str = "admit"           # admit (due + stamp inside admit) | stamp_only (the adapter checked `throttle_due` before pairing)
    dedup_repeat_nonkf_stamp: bool = False     # a non-keyframe whose stamp equals the last enqueued frame's is duplicate_ts (ZeroMQ, adapter-queried)
    mint_epoch_on_stamp_regression: bool = False   # pose_event bumps the epoch when the stamp goes back by > POSE_EPOCH_REBASE_S
    trace_frame_count: bool = True             # receiver lines carry frame_count (websocket) or None (ZeroMQ)
    rx_seq_fallback_to_current: bool = True    # drop lines without an explicit rx_seq carry the current one (websocket); ZeroMQ passes it explicitly (None on malformed)
    depth_decode_failure: str = "none"         # none: a PNG depth that fails to decode leaves depth None (websocket); raise: it is a parse_error (ZeroMQ)
    parse_error_carries_keyframe: bool = False # parse_error lines carry is_keyframe (ZeroMQ: the attempt knows it) or None (websocket)
    default_pose_clock: str = "sender"         # what pose_clock reads when the header carries a wall stamp


WEBSOCKET_POLICY = FrontEndPolicy()
ZEROMQ_POLICY = FrontEndPolicy(
    keyframe_rule="source", pose_with_frame=False, depth_before_admission=False,
    throttle_in_admit="stamp_only", dedup_repeat_nonkf_stamp=True,
    mint_epoch_on_stamp_regression=True, trace_frame_count=False, rx_seq_fallback_to_current=False,
    depth_decode_failure="raise", parse_error_carries_keyframe=True, default_pose_clock="server",
)


# ───────────────────────────── front-end ─────────────────────────────

class IngestFrontEnd:
    """See the module docstring. One instance per source; the receive thread
    of that source is the only caller (no locking beyond the queue's own)."""

    def __init__(
        self,
        *,
        source: str,
        policy: FrontEndPolicy,
        ingest_queue: Any,
        admission_queue: Any = None,
        throttle_clock: str = "wall",
        keyframe_every_n: int = 30,
        nonkf_min_interval_s: float = 0.5,
        require_tracking_normal: bool = True,
        confidence_threshold: int = 1,
        pose_sink: Optional[Callable[..., Any]] = None,
        clearance_sink: Optional[Callable[..., Any]] = None,
        event_sink: Optional[Callable[[Any], None]] = None,
        ledger_sink: Optional[Callable[[Any], None]] = None,
        latency_analytics: Any = None,
        on_camera_frame: Optional[Callable[[Any], None]] = None,
        on_keyframe: Optional[Callable[[Any], None]] = None,
        decode_cache_size: int = 4,
    ) -> None:
        self.source = str(source)
        self.policy = policy
        self.ingest_q = ingest_queue
        self.admission_queue = admission_queue if admission_queue is not None else ingest_queue
        self.throttle = NonKfThrottle(throttle_clock, nonkf_min_interval_s)
        self.keyframe_every_n = max(1, int(keyframe_every_n))
        self.require_tracking_normal = bool(require_tracking_normal)
        self.confidence_threshold = int(confidence_threshold)
        self.pose_sink = pose_sink
        self.clearance_sink = clearance_sink
        self.event_sink = event_sink
        self.ledger_sink = ledger_sink
        self.latency_analytics = latency_analytics
        self.on_camera_frame = on_camera_frame
        self.on_keyframe = on_keyframe
        # Per-session counters (the websocket adapter resets them per connection)
        self.frame_count: int = 0
        self.last_enq_ts_ns: Optional[int] = None
        # Receiver-local running count of front-end interactions (the trace
        # join key: binary messages on websocket / replay, attempts on ZeroMQ)
        self.rx_seq: int = 0
        self.cur_rx_seq: Optional[int] = None
        # Header ids of the frame being admitted (parse-error lines)
        self._hdr_seq: Any = None
        self._hdr_ts: Any = None
        # Frame epoch: session-minted (websocket hello) or receiver-minted (ZeroMQ stamp regression)
        self.frame_epoch: int = 0
        self._epoch_session_id: Optional[str] = None
        self.last_pose_ts_ns: Optional[int] = None
        # Liveness (read by the watchdog through the adapter)
        self.tracking_drops: int = 0
        self.last_rx_mono: Optional[float] = None
        self.last_enqueue_mono: Optional[float] = None
        # Decode memo keyed by header.decode_key (ZeroMQ: the paired camera stamp)
        self._decode_cache: "OrderedDict[Any, Tuple[np.ndarray, Optional[np.ndarray]]]" = OrderedDict()
        self._decode_cache_max = int(decode_cache_size)

    # ── sessions / epochs ──

    def new_session(self, session_id: Optional[str]) -> int:
        """Advance the frame epoch iff ``session_id`` differs from the last one
        (a same-id reconnect keeps the epoch)."""
        if session_id != self._epoch_session_id:
            self._epoch_session_id = session_id
            self.frame_epoch += 1
        return self.frame_epoch

    def reset_session_state(self) -> None:
        """Per-connection resets the websocket receiver performs after a handshake."""
        self.frame_count = 0
        self.throttle.reset()
        self.last_enq_ts_ns = None

    # ── counters ──

    def next_rx_seq(self) -> int:
        self.rx_seq += 1
        self.cur_rx_seq = self.rx_seq
        return self.rx_seq

    # ── queries the ZeroMQ flavour asks before pairing ──

    def is_repeat_stamp(self, t_sensor_ns: Any) -> bool:
        return self.policy.dedup_repeat_nonkf_stamp and self.last_enq_ts_ns == t_sensor_ns

    def throttle_due(self, t_sensor_ns: Any) -> bool:
        return self.throttle.due(t_sensor_ns)

    # ── pose events (poses that arrive separately from frames) ──

    def pose_event(self, sample: PoseSample, *, mailbox_write: bool = True) -> None:
        """A receive-time pose: optional epoch minting on a stamp regression,
        the pose mailbox write, the P2 pose ledger line. Never raises."""
        ts = sample.t_sensor_ns
        stamp = int(ts) if (ts is not None and int(ts) > 0) else None
        if self.policy.mint_epoch_on_stamp_regression and stamp is not None:
            last = self.last_pose_ts_ns
            if last is not None and stamp < int(last) - int(POSE_EPOCH_REBASE_S * 1e9):
                self.frame_epoch += 1
                logger.warning(
                    "[%s] tracking stamp jumped back %.1f s (%d -> %d): new frame_epoch %d",
                    self.source, (int(last) - stamp) / 1e9, int(last), stamp, self.frame_epoch,
                )
            self.last_pose_ts_ns = stamp
        wall, clock = self._wall_and_clock(sample.t_wall_utc_s)
        if mailbox_write and self.pose_sink is not None:
            try:
                self.pose_sink(sample.t_wc, sample.q_wc_xyzw, wall, self.frame_epoch,
                               sensor_ts_ns=stamp, pose_clock=clock)
            except Exception as e:
                logger.error(f"[{self.source}] pose_sink callback error: {e}")
        self._trace_pose(tracking_state=sample.tracking_raw, t_wc=sample.t_wc, q_xyzw=sample.q_wc_xyzw,
                         unix_ts=wall, pose_clock=clock, mailbox_write=(mailbox_write and self.pose_sink is not None),
                         seq=None, ts=stamp, rx_seq=None)

    # ── transport-level drops ──

    def reject(self, reason: str, *, seq: Any = None, ts: Any = None, is_keyframe: Optional[bool] = None,
               frame_count: Optional[int] = None, rx_seq: Optional[int] = None, consume_seq: bool = True) -> None:
        """A drop the transport decided (malformed framing, no camera frame,
        or the ZeroMQ dedup / throttle queries): one receiver line."""
        if consume_seq:
            rx_seq = self.next_rx_seq()
        self.trace(RX_DROPPED, reason, seq=seq, ts=ts, is_kf=is_keyframe, frame_count=frame_count, rx_seq=rx_seq)

    # ── the chain ──

    def admit(self, raw: RawFrame, *, rx_seq: Optional[int] = None) -> Optional[FramePacket]:
        """Run the chain on one frame. Returns the FramePacket to enqueue, or
        None when the chain dropped it (the drop line is written here). An
        exception after the header (bad pose, decode failure, unsupported
        encoding) is traced as ``parse_error`` and RE-RAISED: the caller logs
        and continues, as the receivers always did."""
        h = raw.header
        p = self.policy
        if rx_seq is None:
            rx_seq = self.next_rx_seq()
        else:
            self.cur_rx_seq = rx_seq
        self._hdr_seq, self._hdr_ts = h.seq, h.t_sensor_ns
        try:
            return self._admit_impl(raw, rx_seq)
        except Exception:
            self.trace(RX_DROPPED, RX_PARSE_ERROR, seq=h.seq, ts=h.t_sensor_ns,
                       is_kf=(bool(h.keyframe_hint) if p.parse_error_carries_keyframe else None),
                       frame_count=(self.frame_count if p.trace_frame_count else None), rx_seq=rx_seq)
            raise

    def _admit_impl(self, raw: RawFrame, rx_seq: int) -> Optional[FramePacket]:
        h = raw.header
        p = self.policy
        seq, ts = h.seq, (int(h.t_sensor_ns) if h.t_sensor_ns else None)
        want_stats = self.event_sink is not None or self.ledger_sink is not None

        # 1. Tracking-state filter (sources with a tracking state)
        if self.require_tracking_normal and h.tracking_state != TS_NORMAL:
            if self.ledger_sink is not None:
                lt = lq = None
                err = None
                try:
                    lt, lq = self._pose_from_header(h)
                except Exception as e:  # noqa: BLE001 -- ledger only; the frame is dropped anyway
                    err = f"{type(e).__name__}: {e}"
                wall, clock = self._wall_and_clock(h.t_wall_utc_s)
                self._trace_pose(tracking_state=h.tracking_state, t_wc=lt, q_xyzw=lq, unix_ts=wall, pose_clock=clock,
                                 mailbox_write=False, pose_error=err, seq=seq, ts=ts, rx_seq=rx_seq)
            self.tracking_drops += 1
            if self.latency_analytics:
                self.latency_analytics.record_tracking_drop()
            logger.info(f"[{self.source}] dropping frame: tracking_state={h.tracking_state}")
            self.trace(RX_DROPPED, RX_TRACKING, seq=seq, ts=ts, rx_seq=rx_seq)
            return None

        # 2. Pose (raises -> parse_error, AFTER the filter, BEFORE frame_count)
        t_wc, q_xyzw = self._pose_from_header(h)
        wall, clock = self._wall_and_clock(h.t_wall_utc_s)

        # 3. Receive-time pose mailbox (every frame that reached here, at input rate)
        if p.pose_with_frame and self.pose_sink is not None:
            try:
                self.pose_sink(t_wc, q_xyzw, wall, self.frame_epoch, sensor_ts_ns=ts, pose_clock=clock)
            except Exception as e:
                logger.error(f"[{self.source}] pose_sink callback error: {e}")

        # 4. Depth before admission (cheap; feeds the trace / ledger statistics)
        depth_m: Optional[np.ndarray] = None
        dvf: Optional[float] = None
        conf_map: Optional[np.ndarray] = None
        if p.depth_before_admission:
            depth_m = self._decode_depth(h)
            dvf = codecs.depth_valid_fraction(depth_m) if want_stats else None
            if p.pose_with_frame and self.ledger_sink is not None:
                conf_map = self._decode_confidence(h)
                conf_hist = (np.bincount(conf_map.ravel(), minlength=3)[:3].tolist() if conf_map is not None else None)
                self._trace_pose(tracking_state=h.tracking_state, t_wc=t_wc, q_xyzw=q_xyzw, unix_ts=wall,
                                 pose_clock=clock, mailbox_write=(self.pose_sink is not None),
                                 depth_valid_frac=dvf, conf_hist=conf_hist, seq=seq, ts=ts, rx_seq=rx_seq)

        # 5. Keyframe decision
        self.frame_count += 1
        if p.keyframe_rule == "source":
            is_keyframe = bool(h.keyframe_hint)
            kf_origin = KF_SOURCE if is_keyframe else None
        else:
            is_keyframe = self.frame_count == 1 or self.frame_count % self.keyframe_every_n == 0
            kf_origin = KF_MINTED if is_keyframe else None
        fc = self.frame_count if p.trace_frame_count else None

        # 6. Non-keyframe throttle (the stamp advances on the ADMIT decision,
        #    before decode and enqueue: a full queue must not stop it thinning)
        if not is_keyframe:
            if p.throttle_in_admit == "stamp_only":
                self.throttle.stamp(ts)
            elif not self.throttle.admit(ts):
                if self.latency_analytics:
                    self.latency_analytics.record_throttle_skip()
                self.trace(RX_DROPPED, RX_THROTTLE, seq=seq, ts=ts, is_kf=False, frame_count=fc,
                           depth_valid_frac=dvf, rx_seq=rx_seq)
                return None

        # 7. Lane admission BEFORE the RGB decode (admit-before-decode)
        aq = self.admission_queue
        refusal = aq.refusal(is_keyframe, kf_origin) if aq is not None else None
        if refusal is not None:
            if self.latency_analytics:
                self.latency_analytics.sample_queue_depth(aq.qsize())
                self.latency_analytics.record_queue_drop()
            logger.warning(f"[{self.source}] ingest queue refused {'keyframe' if is_keyframe else 'non-KF'} before decode ({refusal})")
            self.trace(RX_DROPPED, refusal, seq=seq, ts=ts, is_kf=is_keyframe, frame_count=fc,
                       depth_valid_frac=dvf, rx_seq=rx_seq)
            return None

        # 8. Decode on admit (memoised per decode_key)
        rgb, depth_m = self._decode_on_admit(h, depth_m)
        raw_jpeg = (bytes(h.rgb.data) if (h.keep_encoded_rgb and h.rgb.encoding == "jpeg"
                                          and not isinstance(h.rgb.data, np.ndarray)) else None)

        # 9. Receive-time clearance (pre-filter depth), confidence filter, packet
        if self.clearance_sink is not None:
            try:
                c_m, c_frac = codecs.forward_clearance_from_depth(depth_m)
                self.clearance_sink(c_m, c_frac, wall)
            except Exception as e:  # noqa: BLE001 -- guard telemetry never breaks ingest
                logger.error(f"[{self.source}] clearance_sink error: {e}")
        if conf_map is None and h.confidence is not None:
            conf_map = self._decode_confidence(h)
        if conf_map is not None and depth_m is not None and self.confidence_threshold > 0:
            depth_m, conf_map = codecs.apply_confidence_filter(depth_m, conf_map, self.confidence_threshold)
        if h.intrinsics is None and h.extra.get("intrinsics_error"):
            raise KeyError(h.extra["intrinsics_error"])      # the Lens header lacked fx/fy/cx/cy (parse_error, as before)
        stamp_ns = int(h.t_sensor_ns) if h.t_sensor_ns is not None else None
        tb = TimeBundle(t_mono_s=time.monotonic(), t_wall_utc_s=wall, t_sensor_ns=stamp_ns, seq=seq)
        pose = PoseStamped(stamp_ns=stamp_ns, frame_id=h.pose_frame_id, t_wc=t_wc, q_wc_xyzw=q_xyzw)
        return FramePacket(
            time=tb, rgb=rgb, depth_m=depth_m, pose=pose, intr=h.intrinsics, is_keyframe=is_keyframe,
            confidence=conf_map, rgb_jpeg=raw_jpeg, frame_epoch=self.frame_epoch,
            ingest=IngestMeta(keyframe_origin=kf_origin, depth_valid_frac=dvf, rx_seq=rx_seq),
        )

    def enqueue(self, pkt: FramePacket) -> bool:
        """put() + the enqueued / refused trace line + the viz and keyframe
        callbacks + the liveness stamps. Returns whether the queue took it."""
        if self.latency_analytics:
            self.latency_analytics.sample_queue_depth(self.ingest_q.qsize())
        ok = self.ingest_q.put(pkt, block=False)
        if ok:
            self.last_enqueue_mono = time.monotonic()
            self.last_enq_ts_ns = pkt.time.t_sensor_ns
            self.trace(RX_ENQUEUED, "", pkt=pkt)
            if self.on_camera_frame is not None:
                try:
                    self.on_camera_frame(pkt)
                except Exception as e:
                    logger.error(f"[{self.source}] on_camera_frame callback error: {e}")
            if pkt.is_keyframe and self.on_keyframe is not None:
                try:
                    self.on_keyframe(pkt)
                except Exception as e:
                    logger.error(f"[{self.source}] on_keyframe callback error: {e}")
            logger.debug(f"[{self.source}] enqueued {'KF' if pkt.is_keyframe else 'frame'} -> queue={self.ingest_q.qsize()}")
            return True
        if self.latency_analytics:
            self.latency_analytics.record_queue_drop()
        reason = getattr(getattr(pkt, "ingest", None), "drop_reason", None) or RX_QUEUE_FULL
        logger.warning(f"[{self.source}] ingest queue refused frame ({reason}); dropping")
        self.trace(RX_DROPPED, reason, pkt=pkt)
        return False

    def offer(self, raw: RawFrame, *, rx_seq: Optional[int] = None) -> bool:
        """admit + enqueue. Exceptions propagate as from admit()."""
        pkt = self.admit(raw, rx_seq=rx_seq)
        return self.enqueue(pkt) if pkt is not None else False

    # ── decode helpers ──

    def _pose_from_header(self, h) -> Tuple[np.ndarray, np.ndarray]:
        t, q = codecs.parse_pose(h.pose_raw, h.pose_format, pose_scale=float(h.extra.get("pose_scale", 1.0)))
        return codecs.normalize_pose_convention(t, q, h.pose_convention)

    def _decode_depth(self, h) -> Optional[np.ndarray]:
        d = h.depth
        if d is None:
            return None
        return codecs.decode_depth(d.data, d.encoding, d.width, d.height, d.scale)

    def _decode_confidence(self, h) -> Optional[np.ndarray]:
        c = h.confidence
        if c is None:
            return None
        return codecs.decode_confidence(c.data, c.width, c.height)

    def _decode_on_admit(self, h, depth_m: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        key = h.decode_key
        if key is not None:
            hit = self._decode_cache.get(key)
            if hit is not None:
                self._decode_cache.move_to_end(key)
                return hit
        rgb = codecs.decode_rgb(h.rgb.data, h.rgb.encoding, h.rgb.width, h.rgb.height)
        if depth_m is None and h.depth is not None:
            depth_m = self._decode_depth(h)
            if depth_m is None and self.policy.depth_decode_failure == "raise":
                raise ValueError("depth decode failed")
        if key is not None:
            self._decode_cache[key] = (rgb, depth_m)
            while len(self._decode_cache) > self._decode_cache_max:
                self._decode_cache.popitem(last=False)
        return rgb, depth_m

    def _wall_and_clock(self, t_wall_utc_s: Optional[float]) -> Tuple[float, str]:
        """(wall stamp, pose_clock): the sender's stamp tagged "sender", or this
        process's time.time() tagged "server" when it is missing / zero."""
        if t_wall_utc_s:
            return float(t_wall_utc_s), self.policy.default_pose_clock
        return time.time(), "server"

    # ── trace (the one ReceiverEvent builder) ──

    def trace(self, decision: str, reason: str = "", *, pkt: Optional[FramePacket] = None,
              seq: Any = None, ts: Any = None, is_kf: Optional[bool] = None,
              frame_count: Optional[int] = None, queue_depth: Optional[int] = None,
              depth_valid_frac: Optional[float] = None, lane: Optional[str] = None,
              rx_seq: Optional[int] = None) -> None:
        """One frame-flow trace line (no-op without a sink). With a packet its
        ids and ingest bookkeeping are used (the pre-filter depth statistic
        from pkt.ingest, never recomputed). Never raises into the receive path."""
        sink = self.event_sink
        if sink is None:
            return
        try:
            if pkt is not None:
                seq, ts, is_kf = pkt.time.seq, pkt.time.t_sensor_ns, bool(pkt.is_keyframe)
                if frame_count is None and self.policy.trace_frame_count:
                    frame_count = self.frame_count
                meta = getattr(pkt, "ingest", None)
                if depth_valid_frac is None:
                    depth_valid_frac = getattr(meta, "depth_valid_frac", None)
                if lane is None:
                    lane = getattr(meta, "lane", None)
                if rx_seq is None:
                    rx_seq = getattr(meta, "rx_seq", None)
            elif rx_seq is None and self.policy.rx_seq_fallback_to_current:
                rx_seq = self.cur_rx_seq
            if queue_depth is None:
                q = self.admission_queue
                queue_depth = int(q.qsize()) if q is not None else None
            sink(ReceiverEvent(
                timestamp=time.monotonic(),
                source=self.source,
                decision=decision,
                reason=reason,
                frame_seq=(int(seq) if seq is not None else None),
                t_sensor_ns=(int(ts) if ts is not None else None),
                is_keyframe=is_kf,
                frame_count=frame_count,
                queue_depth=queue_depth,
                depth_valid_frac=depth_valid_frac,
                lane=lane,
                rx_seq=rx_seq,
            ))
        except Exception:
            logger.debug(f"[{self.source}] frame-flow trace failed", exc_info=True)

    def _trace_pose(self, *, tracking_state: str, t_wc: Optional[np.ndarray], q_xyzw: Optional[np.ndarray],
                    unix_ts: float, pose_clock: str, mailbox_write: bool, seq: Any, ts: Any,
                    rx_seq: Optional[int], depth_valid_frac: Optional[float] = None,
                    conf_hist: Optional[list] = None, pose_error: Optional[str] = None) -> None:
        """P2 pose ledger line (no-op without a ledger sink). Never raises."""
        sink = self.ledger_sink
        if sink is None:
            return
        try:
            sink(PoseEvent(
                timestamp=time.monotonic(),
                source=self.source,
                rx_seq=rx_seq,
                frame_seq=(int(seq) if seq is not None else None),
                t_sensor_ns=(int(ts) if ts else None),
                t_wall_utc_s=float(unix_ts),
                pose_clock=str(pose_clock),
                epoch=int(self.frame_epoch),
                tracking_state=str(tracking_state),
                mailbox_write=bool(mailbox_write),
                t_wc=([float(v) for v in t_wc] if t_wc is not None else None),
                q_wc_xyzw=([float(v) for v in q_xyzw] if q_xyzw is not None else None),
                pose_error=pose_error,
                depth_valid_frac=depth_valid_frac,
                conf_hist=conf_hist,
            ))
        except Exception:
            logger.debug(f"[{self.source}] pose ledger write failed", exc_info=True)

    # ── liveness ──

    def liveness(self) -> dict:
        return {
            "last_rx_mono": self.last_rx_mono,
            "last_enqueue_mono": self.last_enqueue_mono,
            "tracking_drops": self.tracking_drops,
            "frame_epoch": self.frame_epoch,
        }
