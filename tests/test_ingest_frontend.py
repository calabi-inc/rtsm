"""
P3 task 0.5 -- the ingest front-end, its contracts, the codec layer and the
source registry, driven by an in-process fake source (no transport).

The ORDER of the chain is the contract the session1 anchor and the golden
traces (tests/test_ingest_golden.py) pin from the outside; these tests pin
it from the inside, step by step, with frames a test can build by hand
(raw_bgr / raw_depth_m payloads, prepared poses).
"""
from __future__ import annotations

import numpy as np
import pytest

from rtsm.core.datamodel import PinholeIntrinsics
from rtsm.evaluation.event_log import (
    RX_DROPPED, RX_ENQUEUED, RX_PARSE_ERROR, RX_QUEUE_FULL, RX_THROTTLE, RX_TRACKING, TS_NOT_AVAILABLE,
)
from rtsm.io import codecs, sources
from rtsm.io.contracts import (
    CONTRACT_VERSION, CONVENTION_ARKIT, CONVENTION_OPENCV, POSE_FMT_PREPARED, EncodedImage, FrameHeader,
    PoseSample, RawFrame, Source, SourceContext, TrackingStatus,
)
from rtsm.io.ingest_frontend import (
    POSE_EPOCH_REBASE_S, WEBSOCKET_POLICY, ZEROMQ_POLICY, IngestFrontEnd, NonKfThrottle,
)
from rtsm.io.ingest_lanes import KF_MINTED, KF_SOURCE
from rtsm.io.ingest_queue import IngestQueue
import json

INTR = PinholeIntrinsics(width=8, height=6, fx=5.0, fy=5.0, cx=4.0, cy=3.0)
IDENT_Q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _raw(seq: int, ts_ns: int, *, tracking: str = "normal", pose=None, kf=None, depth="ok",
         decode_key=None, source: str = "fake") -> RawFrame:
    rgb = np.full((6, 8, 3), (seq or 0) % 255, dtype=np.uint8)   # ZeroMQ-shaped frames have no seq
    if depth == "ok":
        d = np.ones((6, 8), dtype=np.float32)
        d[0, :4] = np.nan                                   # 44 / 48 valid
        depth_img = EncodedImage(d, "raw_depth_m", 8, 6)
    elif depth == "badpng":
        depth_img = EncodedImage(b"not a png", "png_uint16_raw", 8, 6, 0.001)
    else:
        depth_img = None
    if pose is None:
        pose = (np.array([float(seq or 0), 0.0, 0.0], dtype=np.float32), IDENT_Q)
    return RawFrame(header=FrameHeader(
        source=source, seq=seq, t_sensor_ns=ts_ns, t_wall_utc_s=((1.7e9 + seq) if seq is not None else None),
        tracking_state=tracking,
        keyframe_hint=kf, rgb=EncodedImage(rgb, "raw_bgr", 8, 6), depth=depth_img, intrinsics=INTR,
        pose_raw=pose, pose_format=POSE_FMT_PREPARED, pose_convention=CONVENTION_OPENCV,
        decode_key=decode_key,
    ))


class Harness:
    """A front-end with list sinks: the fake source's whole world."""

    def __init__(self, policy, *, maxsize: int = 8, clock: str = "sensor", **kw):
        self.q = IngestQueue(maxsize)
        self.events: list = []
        self.ledger: list = []
        self.poses: list = []
        kw.setdefault("keyframe_every_n", 3)
        kw.setdefault("nonkf_min_interval_s", 0.5)
        self.fe = IngestFrontEnd(
            source="fake", policy=policy, ingest_queue=self.q, throttle_clock=clock,
            event_sink=self.events.append, ledger_sink=self.ledger.append,
            pose_sink=lambda t, q, wall, epoch, **k: self.poses.append((tuple(float(v) for v in t), epoch, k)),
            **kw,
        )

    def lines(self):
        return [(e.decision, e.reason, e.frame_seq, e.is_keyframe, e.frame_count, e.rx_seq) for e in self.events]

    def packets(self):
        out = []
        while True:
            p = self.q.get(timeout=0)
            if p is None:
                return out
            out.append(p)


# ───────────────────────────── websocket flavour ─────────────────────────────

def test_websocket_chain_order_keyframes_throttle_and_sinks():
    h = Harness(WEBSOCKET_POLICY)
    fe = h.fe
    results = [fe.offer(_raw(i, i * 200_000_000)) for i in range(1, 6)]
    # KF minted at frame 1 and every 3rd; the 0.5 s sensor throttle drops
    # frame 4 (0.4 s after admitted frame 2) and admits frame 5 (0.6 s after)
    assert results == [True, True, True, False, True]
    assert h.lines() == [
        (RX_ENQUEUED, "", 1, True, 1, 1),
        (RX_ENQUEUED, "", 2, False, 2, 2),
        (RX_ENQUEUED, "", 3, True, 3, 3),
        (RX_DROPPED, RX_THROTTLE, 4, False, 4, 4),
        (RX_ENQUEUED, "", 5, False, 5, 5),
    ]
    # the pose mailbox is written at INPUT rate, before the throttle
    assert [p[0][0] for p in h.poses] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(p[2]["pose_clock"] == "sender" and p[1] == 0 for p in h.poses)
    # one pose-ledger line per frame that passed the tracking filter, with the pre-admission depth statistic
    assert len(h.ledger) == 5
    assert all(abs(l.depth_valid_frac - 44 / 48) < 1e-6 and l.mailbox_write for l in h.ledger)
    assert h.events[3].depth_valid_frac == pytest.approx(44 / 48)     # the throttle line carries it too
    pk = h.packets()
    assert [p.is_keyframe for p in pk] == [True, False, True, False]
    assert [p.ingest.keyframe_origin for p in pk] == [KF_MINTED, None, KF_MINTED, None]
    assert [p.ingest.rx_seq for p in pk] == [1, 2, 3, 5]
    assert pk[0].rgb_jpeg is None and pk[0].pose.frame_id == "world" and pk[0].frame_epoch == 0
    assert pk[0].intr is INTR and pk[0].time.t_sensor_ns == 200_000_000 and pk[0].time.seq == 1


def test_tracking_filter_runs_before_pose_parse_and_counts_nothing():
    h = Harness(WEBSOCKET_POLICY)
    bad_pose = ("not", "a", "pose")
    assert h.fe.offer(_raw(1, 100, tracking="limited", pose=bad_pose)) is False
    assert h.lines() == [(RX_DROPPED, RX_TRACKING, 1, None, None, 1)]
    assert h.fe.frame_count == 0 and h.fe.tracking_drops == 1 and h.poses == []
    # the ledger still gets a line for the dropped frame, with the parse failure recorded
    assert len(h.ledger) == 1 and h.ledger[0].mailbox_write is False and h.ledger[0].pose_error
    assert h.ledger[0].tracking_state == "limited"
    # a tracking-normal frame whose pose fails -> parse_error line, re-raised, frame_count untouched
    with pytest.raises(Exception):
        h.fe.offer(_raw(2, 200, pose=bad_pose))
    assert h.lines()[-1] == (RX_DROPPED, RX_PARSE_ERROR, 2, None, 0, 2)
    assert h.fe.frame_count == 0 and h.q.qsize() == 0


def test_queue_refusal_happens_before_the_rgb_decode(monkeypatch):
    calls = []
    real = codecs.decode_rgb
    monkeypatch.setattr(codecs, "decode_rgb", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    h = Harness(WEBSOCKET_POLICY, maxsize=1)
    assert h.fe.offer(_raw(1, 1_000_000_000)) is True
    assert h.fe.offer(_raw(2, 2_000_000_000)) is False        # queue full: refused BEFORE decode
    assert len(calls) == 1
    assert h.lines()[-1] == (RX_DROPPED, RX_QUEUE_FULL, 2, False, 2, 2)
    assert h.fe.throttle.last_admit_sensor_ns == 2_000_000_000  # the throttle still advanced (admit decision)


def test_websocket_depth_decode_failure_leaves_depth_none():
    h = Harness(WEBSOCKET_POLICY)
    assert h.fe.offer(_raw(1, 100, depth="badpng")) is True
    pk = h.packets()
    assert pk[0].depth_m is None and h.ledger[0].depth_valid_frac is None


def test_session_epoch_and_per_connection_reset():
    h = Harness(WEBSOCKET_POLICY)
    fe = h.fe
    assert fe.new_session("a") == 1
    assert fe.new_session("a") == 1          # same-id reconnect keeps the epoch
    assert fe.new_session("b") == 2
    fe.offer(_raw(1, 1_000_000_000))
    fe.offer(_raw(2, 1_200_000_000))
    assert fe.frame_count == 2 and fe.last_enq_ts_ns == 1_200_000_000
    fe.reset_session_state()
    assert fe.frame_count == 0 and fe.last_enq_ts_ns is None and fe.throttle.last_admit_sensor_ns is None
    assert h.packets()[0].frame_epoch == 2


# ───────────────────────────── zeromq flavour ─────────────────────────────

def _zmq_harness(**kw):
    return Harness(ZEROMQ_POLICY, require_tracking_normal=False, confidence_threshold=0, **kw)


def test_zeromq_pose_events_mint_epochs_on_stamp_regression():
    h = _zmq_harness()
    fe = h.fe
    t = np.zeros(3, dtype=np.float32)
    for stamp in (10_000_000_000, 10_100_000_000, 2_000_000_000, 1_500_000_000):
        fe.pose_event(PoseSample(t_sensor_ns=stamp, t_wall_utc_s=None, t_wc=t, q_wc_xyzw=IDENT_Q,
                                 tracking=TrackingStatus.from_rtabmap(), tracking_raw=TS_NOT_AVAILABLE))
    # 10.1 s -> 2.0 s is a restart (> POSE_EPOCH_REBASE_S); 2.0 -> 1.5 s is not
    assert POSE_EPOCH_REBASE_S == 5.0
    assert [l.epoch for l in h.ledger] == [0, 0, 1, 1]
    assert [p[1] for p in h.poses] == [0, 0, 1, 1]
    assert all(l.pose_clock == "server" and l.tracking_state == TS_NOT_AVAILABLE and l.mailbox_write for l in h.ledger)
    assert all(p[2]["pose_clock"] == "server" for p in h.poses)
    assert fe.last_pose_ts_ns == 1_500_000_000
    assert h.events == []                                     # pose events write no receiver line


def test_zeromq_source_keyframes_dedup_and_stamp_only_throttle():
    h = _zmq_harness()
    fe = h.fe
    assert fe.is_repeat_stamp(1_000_000_000) is False
    assert fe.throttle_due(1_000_000_000) is True
    rx = fe.next_rx_seq()
    pkt = fe.admit(_raw(None, 1_000_000_000, kf=False), rx_seq=rx)
    assert pkt is not None and fe.enqueue(pkt)
    assert pkt.is_keyframe is False and pkt.ingest.keyframe_origin is None and pkt.ingest.rx_seq == 1
    # stamp_only: the adapter asked `throttle_due` before pairing; admit() only stamps
    assert fe.throttle_due(1_200_000_000) is False and fe.throttle_due(1_600_000_000) is True
    assert fe.is_repeat_stamp(1_000_000_000) is True          # the last enqueued stamp
    # a source keyframe with the same stamp is never a duplicate and carries the source origin
    rx = fe.next_rx_seq()
    kf = fe.admit(_raw(None, 1_000_000_000, kf=True), rx_seq=rx)
    assert kf.is_keyframe is True and kf.ingest.keyframe_origin == KF_SOURCE and fe.enqueue(kf)
    assert h.lines() == [(RX_ENQUEUED, "", None, False, None, 1), (RX_ENQUEUED, "", None, True, None, 2)]
    # transport-level drops: the adapter's own rx_seq, no fallback to the current one
    fe.reject("duplicate_ts", ts=1_000_000_000, is_keyframe=False, rx_seq=7, consume_seq=False)
    fe.reject("malformed", is_keyframe=True, rx_seq=None, consume_seq=False)
    assert h.lines()[2:] == [(RX_DROPPED, "duplicate_ts", None, False, None, 7),
                             (RX_DROPPED, "malformed", None, True, None, None)]
    assert fe.rx_seq == 2


def test_zeromq_parse_error_carries_keyframe_and_depth_failure_raises():
    h = _zmq_harness()
    fe = h.fe
    with pytest.raises(Exception):
        fe.admit(_raw(None, 5, kf=True, pose=("bad",)), rx_seq=fe.next_rx_seq())
    assert h.lines()[-1] == (RX_DROPPED, RX_PARSE_ERROR, None, True, None, 1)
    with pytest.raises(ValueError, match="depth decode failed"):
        fe.admit(_raw(None, 6, kf=False, depth="badpng"), rx_seq=fe.next_rx_seq())
    assert h.lines()[-1] == (RX_DROPPED, RX_PARSE_ERROR, None, False, None, 2)
    assert h.q.qsize() == 0 and h.poses == []                 # ZMQ poses never ride with the frame


def test_decode_memo_keyed_by_decode_key(monkeypatch):
    calls = []
    real = codecs.decode_rgb
    monkeypatch.setattr(codecs, "decode_rgb", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    h = _zmq_harness(decode_cache_size=2)
    fe = h.fe
    a = fe.admit(_raw(1, 1_000_000_000, kf=True, decode_key="cam-1"), rx_seq=fe.next_rx_seq())
    b = fe.admit(_raw(2, 1_000_000_001, kf=True, decode_key="cam-1"), rx_seq=fe.next_rx_seq())
    assert len(calls) == 1 and b.rgb is a.rgb                 # same camera frame: decoded once
    fe.admit(_raw(3, 2_000_000_000, kf=True, decode_key="cam-2"), rx_seq=fe.next_rx_seq())
    fe.admit(_raw(4, 3_000_000_000, kf=True, decode_key="cam-3"), rx_seq=fe.next_rx_seq())
    assert list(fe._decode_cache) == ["cam-2", "cam-3"]       # bounded, oldest evicted


# ───────────────────────────── throttle ─────────────────────────────

def test_nonkf_throttle_sensor_mode_with_wall_fallback():
    th = NonKfThrottle("sensor", 0.5)
    assert th.due(None) is True                               # no stamp -> wall mode, never admitted yet
    assert th.admit(1_000_000_000) is True
    assert th.due(1_200_000_000) is False and th.due(1_500_000_000) is True
    assert th.due(500_000_000) is True                        # a stamp that went back is due (new clock)
    th.reset()
    assert th.last_admit_sensor_ns is None and th.due(1_100_000_000) is True


# ───────────────────────────── contracts / codecs ─────────────────────────────

def test_contract_version_and_tracking_status_mapping():
    assert CONTRACT_VERSION == 1
    assert TrackingStatus.from_arkit("normal") is TrackingStatus.OK
    assert TrackingStatus.from_arkit("limited_excessive_motion") is TrackingStatus.DEGRADED
    assert TrackingStatus.from_arkit("not_available") is TrackingStatus.LOST
    assert TrackingStatus.from_arkit(None) is TrackingStatus.UNKNOWN
    assert TrackingStatus.from_rtabmap() is TrackingStatus.UNKNOWN

    class Minimal:
        name = "minimal"
        def start(self): ...
        def stop(self): ...
        def liveness(self): return {}

    assert isinstance(Minimal(), Source)
    assert not isinstance(object(), Source)


def test_codecs_pose_formats_and_conventions():
    t, q = codecs.parse_pose((np.array([1, 2, 3]), IDENT_Q), POSE_FMT_PREPARED)
    assert t.dtype == np.float32 and t.tolist() == [1.0, 2.0, 3.0]
    t, q = codecs.parse_pose([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], "rtabmap_euler", pose_scale=0.5)
    assert t.tolist() == [0.5, 1.0, 1.5] and np.allclose(q, IDENT_Q)
    mat = np.eye(4); mat[:3, 3] = [4, 5, 6]
    t, q = codecs.parse_pose(mat.flatten(order="F").tolist(), "matrix4x4_col_major")
    assert t.tolist() == [4.0, 5.0, 6.0]
    with pytest.raises(codecs.UnsupportedEncoding):
        codecs.parse_pose([0] * 7, "nope")
    # opencv = identity; arkit = the one-time flip the websocket receiver always applied
    assert codecs.normalize_pose_convention(t, q, CONVENTION_OPENCV) == (t, q)
    t2, q2 = codecs.normalize_pose_convention(t, q, CONVENTION_ARKIT)
    assert t2.tolist() == [4.0, 5.0, 6.0]                     # a pure flip leaves the translation alone
    assert not np.allclose(q2, q)
    with pytest.raises(codecs.UnsupportedEncoding):
        codecs.normalize_pose_convention(t, q, "ros")


def test_codecs_confidence_filter_and_intrinsics_rescale():
    depth = np.ones((4, 4), dtype=np.float32)
    conf = np.array([[0, 2], [2, 0]], dtype=np.uint8)         # half resolution
    out, used = codecs.apply_confidence_filter(depth, conf, 1)
    assert used.shape == (4, 4) and out is depth
    assert np.isnan(out[0, 0]) and np.isnan(out[3, 3]) and out[0, 3] == 1.0
    same_d, same_c = codecs.apply_confidence_filter(np.ones((2, 2), np.float32), conf, 0)
    assert same_c is conf                                     # threshold 0 = off, untouched
    intr = codecs.rescale_intrinsics(100.0, 100.0, 50.0, 40.0, from_wh=(100, 80), to_wh=(200, 160))
    assert (intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height) == (200.0, 200.0, 100.0, 80.0, 200, 160)
    same = codecs.rescale_intrinsics(100.0, 100.0, 50.0, 40.0, from_wh=(0, 0), to_wh=(200, 160))
    assert (same.fx, same.cx) == (100.0, 50.0)                # missing declared size: values unchanged


# ───────────────────────────── registry ─────────────────────────────

class FakeSource:
    name = "fake"

    def __init__(self, cfg, ctx, **options):
        self.cfg, self.ctx, self.options = cfg, ctx, options
        self.started = False

    def start(self): self.started = True
    def stop(self): self.started = False
    def liveness(self): return {"alive": self.started}


@pytest.fixture
def clean_registry():
    yield
    sources.unregister_source("fake")
    sources.unregister_source("fake2")


def test_registry_builtins_registration_and_errors(clean_registry):
    ctx = SourceContext(ingest_queue=IngestQueue(2))
    assert {"websocket", "zeromq", "replay"} <= set(sources.available_sources())
    sources.register_source("fake", FakeSource)
    src = sources.make_source("FAKE", {"io": {}}, ctx, extra=1)
    assert isinstance(src, FakeSource) and isinstance(src, Source) and src.options == {"extra": 1}
    src.start(); assert src.liveness() == {"alive": True}
    with pytest.raises(ValueError, match="already registered"):
        sources.register_source("fake", FakeSource)
    sources.register_source("fake", FakeSource, replace=True)
    with pytest.raises(ValueError, match="built-in"):
        sources.register_source("websocket", FakeSource)
    with pytest.raises(sources.UnknownSourceError, match="Available: fake, replay, websocket, zeromq"):
        sources.make_source("rosbag", {}, ctx)
    assert issubclass(sources.UnknownSourceError, ValueError)
    sources.register_source("fake2", lambda cfg, ctx, **o: object())
    with pytest.raises(TypeError, match="lacks name/start/stop/liveness"):
        sources.make_source("fake2", {}, ctx)


def test_registry_discovers_entry_points_and_protects_builtins(monkeypatch, caplog):
    class EP:
        def __init__(self, name, obj=None, fail=False):
            self.name, self._obj, self._fail = name, obj, fail
        def load(self):
            if self._fail:
                raise ImportError("boom")
            return self._obj

    def fake_entry_points(*, group):
        assert group == sources.ENTRY_POINT_GROUP
        return [EP("bagfake", FakeSource), EP("websocket", FakeSource), EP("broken", fail=True)]

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    with caplog.at_level("WARNING", logger="rtsm.io.sources"):
        avail = sources.available_sources()
    assert avail["bagfake"] is FakeSource
    assert avail["websocket"] is sources.websocket_source      # the built-in wins over a colliding plug-in
    assert "broken" not in avail
    assert "collides with a built-in" in caplog.text and "failed to load" in caplog.text


def test_builtin_factories_build_the_adapters(tmp_path):
    ctx = SourceContext(ingest_queue=IngestQueue(2), clock_mode="sensor", keyframe_every_n=4,
                        nonkf_min_interval_s=0.25, require_tracking_normal=False, confidence_threshold=2)
    (tmp_path / "messages.bin").write_bytes(b"")
    (tmp_path / "index.jsonl").write_text("")
    rp = sources.make_source("replay", {}, ctx, recording_dir=str(tmp_path), replay_speed=2.0)
    assert rp.name == "replay" and isinstance(rp, Source)
    assert rp.frontend.policy is WEBSOCKET_POLICY and rp.frontend.throttle.clock == "sensor"
    assert rp.frontend.keyframe_every_n == 4 and rp.frontend.require_tracking_normal is False
    with pytest.raises(ValueError, match="recording_dir"):
        sources.make_source("replay", {}, ctx)
    ws = sources.make_source("websocket", {"io": {"websocket": {"port": 8799}}}, ctx)
    assert ws.name == "websocket" and ws.frontend.policy is WEBSOCKET_POLICY
    assert ws.frontend.confidence_threshold == 2 and ws.frontend.throttle.interval_s == 0.25
    zq = sources.make_source("zeromq", {"units": {"pose_m_per_unit": 2.0}}, ctx, pair_window_s=1.0, pair_window_frames=7)
    try:
        assert zq.name == "zeromq" and zq.frontend.policy is ZEROMQ_POLICY
        assert zq.frontend.require_tracking_normal is False and zq._nonkf_min_interval_s == 0.25
        assert zq._pose_scale == 2.0
    finally:
        zq.close()


# ───────────────────────────── source independence ─────────────────────────────

def test_bare_front_end_reproduces_the_websocket_golden_trace():
    """The extraction's proof: the golden websocket stream (recorded from the
    pre-extraction receiver), pushed through a BARE IngestFrontEnd by way of
    the Lens framing function only -- no WebSocketReceiver, no delegates --
    reproduces the fixture's receiver lines, pose-ledger lines, pose-sink
    calls and packets. Nothing in the decisions depends on the transport."""
    import test_ingest_golden as G
    from rtsm.evaluation.event_log import RX_MALFORMED
    from rtsm.io.websocket import LensFramingError, lens_raw_frame

    fixture = json.loads((G.FIXTURES / "ingest_golden_websocket.json").read_text(encoding="utf-8"))
    events, poses, sink_calls, packets = [], [], [], []
    q = IngestQueue(maxsize=2)
    fe = IngestFrontEnd(
        source="websocket", policy=WEBSOCKET_POLICY, ingest_queue=q, throttle_clock="sensor",
        keyframe_every_n=4, nonkf_min_interval_s=0.5, confidence_threshold=2,
        pose_sink=lambda t, qq, ts, ep, **kw: sink_calls.append([G._r(t.tolist()), G._r(qq.tolist()),
                                                                 (G._r(ts, 3) if kw.get("pose_clock") == "sender" else "server-clock"), ep, G._r(kw)]),
        event_sink=events.append, ledger_sink=poses.append,
    )
    fe.new_session("s1")
    for item in G.ws_stream():
        if item == "session:s2":
            fe.new_session("s2"); continue
        if item == "corrections":
            continue                                          # a text message never reaches the front-end
        data = G._ws_frame(**item)
        rx_seq = fe.next_rx_seq(); ids = [None, None]
        try:
            raw = lens_raw_frame(data, source="websocket", apply_camera_flip=True, hdr_ids=ids)
        except LensFramingError as e:
            fe.reject(RX_MALFORMED, seq=e.seq, ts=e.ts, rx_seq=rx_seq, consume_seq=False); continue
        try:
            pkt = fe.admit(raw, rx_seq=rx_seq)
        except Exception:
            continue                                          # parse_error line written inside admit()
        if pkt is None:
            continue
        packets.append(G._pkt_summary(pkt))
        if q.put(pkt, block=False):
            fe.trace(RX_ENQUEUED, "", pkt=pkt); fe.last_enq_ts_ns = pkt.time.t_sensor_ns
        else:
            fe.trace(RX_DROPPED, RX_QUEUE_FULL, pkt=pkt)
    assert [G._rx_line(e) for e in events] == fixture["receiver"]
    assert [G._pose_line(p) for p in poses] == fixture["pose_ledger"]
    assert sink_calls == fixture["pose_sink"]
    assert packets == fixture["packets"]
    assert (fe.frame_count, fe.frame_epoch, fe.tracking_drops, q.qsize()) == tuple(
        fixture["state"][k] for k in ("frame_count", "frame_epoch", "tracking_drops", "queue"))
