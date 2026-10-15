"""Persistence guards found by the 0.3.0 review (2026-10-14): the RAM cap must never evict a frozen session's
payload before it is on disk, and a DELETE during an in-flight write must not leave a folder behind."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from rtsm.core.sessions import LiveComponents, LiveVizState, SessionRegistry, SessionSnapshot

T0 = 1_700_000_000.0


class Clock:
    def __init__(self) -> None:
        self.t = T0

    def now(self) -> float:
        self.t += 1.0
        return self.t


class FakeWM:
    def __init__(self) -> None:
        self.objects = [{"id": "o1", "xyz_world": np.array([1.0, 0.0, 0.5], np.float32), "confirmed": True, "hits": 3,
                         "stability": 0.8, "label_primary": "cup", "label_scores": {"cup": 0.9}, "emb_mean": [0.1, 0.2],
                         "last_update_frame_id": "ws_1", "image_crops": [b"\xff\xd8jpeg"], "view_bins": {1}}]

    def snapshot_for_session(self) -> SessionSnapshot:
        return SessionSnapshot(objects=[dict(o) for o in self.objects], frame_to_objects={"ws_1": ["o1"]},
                               crops={"o1": [b"\xff\xd8jpeg"]})

    def clear(self):
        return {}


class FakeVectors:
    def export(self):
        return ["o1"], np.full((1, 4), 0.5, dtype=np.float32)

    def clear(self):
        return {}


class FakeViz:
    def freeze_live(self, next_key, **_kw):
        rec = SimpleNamespace(mesh_id="ws_1", timestamp_ns=1_000_000_000, positions=np.ones((10, 3), np.float32),
                              colors=np.full((10, 3), 7, np.uint8), pose=np.eye(4, dtype=np.float32), epoch=0)
        return LiveVizState(records=[rec], pose_history={"ws_1": np.eye(4, dtype=np.float32)}, alias_stamp={"ws_1": 1_000_000_000})

    def set_live_session(self, key, sid):
        pass


def pkt(epoch: int, seq: int = 1) -> SimpleNamespace:
    return SimpleNamespace(frame_epoch=epoch, session_id=f"S{epoch}", time=SimpleNamespace(t_sensor_ns=1_000_000_000 * seq, seq=seq),
                           is_keyframe=True, pose=SimpleNamespace(T_wc=lambda: np.eye(4, dtype=np.float32)), kf_id=None,
                           ingest=SimpleNamespace(rx_seq=seq))


def registry(tmp_path, *, keep_in_memory: int, persist_async: bool = False) -> SessionRegistry:
    cfg = {"sessions": {"dir": str(tmp_path / "sessions"), "persist_on_freeze": True, "persist_crops": True,
                        "keep_in_memory": keep_in_memory, "min_keyframes": 1}}
    return SessionRegistry(cfg, live=LiveComponents(wm=FakeWM(), vectors=FakeVectors(), vis_server=FakeViz()),
                           source="fake", now=Clock().now, persist_async=persist_async)


def _folder_payload(folder):
    clouds = np.load(folder / "clouds.npz")
    crops = sorted(p.name for p in (folder / "crops").rglob("*.jpg")) if (folder / "crops").exists() else []
    return {"clouds": sorted(clouds.files), "vectors": (folder / "vectors.npz").exists(), "crops": crops}


def test_keep_cap_never_evicts_an_unpersisted_payload(tmp_path):
    reg = registry(tmp_path, keep_in_memory=0)
    reg.observe_frame(pkt(0))
    b = reg.observe_frame(pkt(1, seq=2))
    assert b is not None and b.frozen_key
    rec = reg._find(b.frozen_key)
    assert rec.on_disk and rec.persisted and rec.persist_error is None
    payload = _folder_payload(reg.dir / rec.key)
    assert payload["clouds"] == ["colors_ws_1", "positions_ws_1"] and payload["vectors"] and payload["crops"] == ["0.jpg"]
    # with the payload on disk, keep_in_memory 0 evicts it from RAM afterwards
    assert rec.clouds is None and rec.vectors is None and rec.payload_loaded is False


def test_keep_cap_waits_for_the_writer_backlog(tmp_path):
    reg = registry(tmp_path, keep_in_memory=1, persist_async=True)
    queued = []
    reg._schedule = lambda fn: queued.append(fn)              # the writer is "behind": nothing persists until we say so
    reg.observe_frame(pkt(0))
    keys = []
    for e in (1, 2, 3):
        keys.append(reg.observe_frame(pkt(e, seq=e + 1)).frozen_key)
    recs = [reg._find(k) for k in keys]
    assert all(not r.on_disk for r in recs)
    assert all(r.clouds is not None and r.vectors is not None for r in recs), "unpersisted payloads were evicted"
    for fn in queued:                                          # the writer catches up
        fn()
    for r in recs:
        assert r.on_disk and _folder_payload(reg.dir / r.key)["clouds"] == ["colors_ws_1", "positions_ws_1"]
    assert sum(1 for r in recs if r.clouds is not None) <= 1  # the cap applies once the payloads are on disk


def test_delete_during_an_in_flight_write_leaves_no_folder(tmp_path):
    reg = registry(tmp_path, keep_in_memory=3, persist_async=True)
    queued = []
    reg._schedule = lambda fn: queued.append(fn)
    reg.observe_frame(pkt(0))
    key = reg.observe_frame(pkt(1, seq=2)).frozen_key
    assert reg._find(key) is not None and not reg._find(key).on_disk
    assert reg.delete(key) is None                             # accepted while the write is still queued
    assert reg._find(key) is None
    for fn in queued:
        fn()                                                   # the writer finishes after the delete
    assert not (reg.dir / key).exists() and not (reg.dir / (key + ".tmp")).exists()
    fresh = SessionRegistry({"sessions": {"dir": str(reg.dir)}}, source="fake", persist_async=False)
    assert fresh.load_from_disk() == 0
