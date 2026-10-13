"""Stage 3 of the session registry: what the live stores hand to a freeze.

WorkingMemory.snapshot_for_session (copies, the frame index, the crops) and the
last_correction summary; FaissClient.export; KeyframeRegistry session tagging,
per-session listing and detach.
"""
from __future__ import annotations

import numpy as np
import pytest

from rtsm.core.sessions import SessionSnapshot
from rtsm.stores.working_memory import WorkingMemory
from rtsm.visualization.registry import KeyframeRegistry

DIM = 8


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _wm_with_two_objects():
    wm = WorkingMemory({"object": {"promote_hits": 2, "require_view_bins": 1}})
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    crop[:, :, 2] = 200
    a = wm.create_object(np.array([1.0, 0.0, 2.0], np.float32), _unit(np.ones(DIM)), label_topk=[("cup", 0.9)],
                         view_dir_cam=np.array([0.0, 0.0, 1.0], np.float32), crop=crop, frame_id="ws_1")
    b = wm.create_object(np.array([3.0, 0.0, 2.0], np.float32), _unit(np.arange(DIM) + 1.0), label_topk=[("mug", 0.4)],
                         view_dir_cam=np.array([0.0, 0.0, 1.0], np.float32), frame_id="ws_2")
    assert a and b
    return wm, a, b


class TestWorkingMemorySnapshot:
    def test_snapshot_shape_index_and_crops(self):
        wm, a, b = _wm_with_two_objects()
        snap = wm.snapshot_for_session()
        assert isinstance(snap, SessionSnapshot)
        by_id = {o["id"]: o for o in snap.objects}
        assert set(by_id) == {a, b}
        oa = by_id[a]
        assert np.allclose(oa["xyz_world"], [1.0, 0.0, 2.0]) and oa["xyz_world"].dtype == np.float32
        assert oa["label_primary"] == "cup" and oa["label_hint"] == "cup" and oa["label_scores"]["cup"] == pytest.approx(0.9)
        assert oa["confirmed"] is False and oa["hits"] >= 1 and 0.0 <= oa["stability"] <= 1.0
        assert len(oa["emb_mean"]) == DIM and oa["n_crops"] == 1 and oa["last_update_frame_id"] == "ws_1"
        assert by_id[b]["n_crops"] == 0 and by_id[b]["last_update_frame_id"] == "ws_2"
        assert snap.frame_to_objects == {"ws_1": [a], "ws_2": [b]}
        assert list(snap.crops) == [a] and snap.crops[a][0][:2] == b"\xff\xd8"

    def test_snapshot_is_a_copy(self):
        wm, a, _b = _wm_with_two_objects()
        snap = wm.snapshot_for_session()
        oa = next(o for o in snap.objects if o["id"] == a)
        oa["xyz_world"][0] = 99.0
        oa["label_scores"]["cup"] = 0.0
        snap.frame_to_objects["ws_1"].append("ghost")
        live = wm.get(a)
        assert float(live.xyz_world[0]) == 1.0 and live.label_scores["cup"] == pytest.approx(0.9)
        assert wm.snapshot_for_session().frame_to_objects["ws_1"] == [a]

    def test_empty_memory_snapshot(self):
        wm = WorkingMemory({})
        snap = wm.snapshot_for_session()
        assert snap.objects == [] and snap.frame_to_objects == {} and snap.crops == {}

    def test_apply_pose_corrections_records_last_correction(self):
        wm, a, b = _wm_with_two_objects()
        assert wm.last_correction is None
        dR = np.eye(3, dtype=np.float32)
        n = wm.apply_pose_corrections({"ws_1": (np.array([0.0, 0.0, 0.0], np.float32), dR, np.array([0.0, 0.0, 0.6], np.float32))})
        assert n == 2                                       # a directly, b by the nearest-camera fallback
        lc = wm.last_correction
        assert lc["corrected"] == 2 and lc["direct"] == 1 and lc["fallback"] == 1 and lc["n_frames"] == 1
        assert lc["max_shift_m"] == pytest.approx(0.6, abs=1e-5)
        assert sorted(lc["demoted"]) == sorted([a, b])      # 0.6 m is above the 0.30 m default
        assert np.allclose(wm.get(a).xyz_world, [1.0, 0.0, 2.6])


class TestFaissExport:
    @pytest.fixture
    def client(self):
        pytest.importorskip("faiss")
        from rtsm.stores.vectors.faiss_client import FaissClient
        return FaissClient({"vectors": {"enable": True, "backend": "faiss", "dim": DIM}})

    def test_export_empty(self, client):
        ids, embs = client.export()
        assert ids == [] and embs.shape == (0, DIM) and embs.dtype == np.float32

    def test_export_sorted_copies(self, client):
        e_b = _unit(np.arange(DIM) + 1.0)
        e_a = _unit(np.ones(DIM))
        client.upsert_batch([{"object_id": "b", "emb": e_b}, {"object_id": "a", "emb": e_a}])
        ids, embs = client.export()
        assert ids == ["a", "b"] and embs.shape == (2, DIM)
        assert np.allclose(embs[0], e_a) and np.allclose(embs[1], e_b)
        embs[0, 0] = 42.0
        assert np.allclose(client.export()[1][0], e_a)       # a copy, not the stored array
        client.clear()
        assert client.export()[0] == []


class TestRegistrySessions:
    @staticmethod
    def _reg():
        r = KeyframeRegistry()
        pts = np.zeros((2, 3), np.float32)
        cols = np.zeros((2, 3), np.uint8)
        r.register("ws_1", 10, pts, cols, np.eye(4, dtype=np.float32), session_key="A")
        r.register("ws_2", 20, pts, cols, np.eye(4, dtype=np.float32), session_key="A")
        r.register("ws_3", 30, pts, cols, None, session_key="B")
        r.register("kf_9", 40, pts, cols, None)                 # untagged (a bridge thumbnail)
        return r

    def test_default_session_key_is_empty_and_update_keeps_it(self):
        r = self._reg()
        assert r.get("kf_9").session_key == ""
        r.register("ws_1", 11, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8))   # update without a key
        assert r.get("ws_1").session_key == "A" and r.get("ws_1").timestamp_ns == 11
        r.register("ws_1", 12, np.zeros((1, 3), np.float32), np.zeros((1, 3), np.uint8), session_key="C")
        assert r.get("ws_1").session_key == "C"

    def test_get_all_and_count_per_session(self):
        r = self._reg()
        assert sorted(k.mesh_id for k in r.get_all("A")) == ["ws_1", "ws_2"]
        assert r.count("A") == 2 and r.count("B") == 1 and r.count("") == 1 and r.count() == 4
        assert len(r.get_all()) == 4

    def test_detach_removes_and_returns_one_session(self):
        r = self._reg()
        out = r.detach("A")
        assert sorted(k.mesh_id for k in out) == ["ws_1", "ws_2"]
        assert r.count() == 2 and not r.exists("ws_1") and r.exists("ws_3") and r.exists("kf_9")
        assert r.detach("A") == []
        assert r.stats()["keyframes"] == 2
