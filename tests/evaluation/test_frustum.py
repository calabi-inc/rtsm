"""P2 stage C, the pose-math parity test (CLAUDE.md: pose math is tested before
it ships): the vectorised frustum projection must agree with the associator's
own `_project_px` point by point on random poses, and the depth sampling must
use the mask stage's RGB->depth coordinate rule. CPU-only, no models."""
from __future__ import annotations

import numpy as np
import pytest

from rtsm.core.association import _invert_h, _project_px
from rtsm.core.datamodel import PoseStamped
from rtsm.core.frustum import FRUSTUM_MODEL_V1, frustum_view, project_points, sample_depth

INTR = {"fx": 500.0, "fy": 480.0, "cx": 320.0, "cy": 240.0}
HW = (480, 640)


def _random_T_cw(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4).astype(np.float32); q /= np.linalg.norm(q)
    t = rng.uniform(-2.0, 2.0, size=3).astype(np.float32)
    T_wc = PoseStamped(stamp_ns=0, frame_id="", t_wc=t, q_wc_xyzw=q).T_wc()      # float32, like the packet
    return _invert_h(T_wc)                                                        # what the pipeline hands the associator


class TestParityWithTheAssociator:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_poses_agree_point_by_point(self, seed):
        rng = np.random.default_rng(seed)
        T_cw = _random_T_cw(rng)
        P = rng.uniform(-4.0, 4.0, size=(200, 3)).astype(np.float32)
        u, v, z, inside = project_points(P, T_cw, INTR, HW)
        n_behind = 0
        for i in range(P.shape[0]):
            ref = _project_px(P[i], T_cw, INTR)
            if ref is None:                                   # behind the camera for the associator
                n_behind += 1
                assert not np.isfinite(u[i]) and not np.isfinite(v[i]) and z[i] <= 1e-6 and not inside[i]
            else:
                assert np.isfinite(u[i]) and np.isfinite(v[i])
                # both sides carry the packet's float32 pose; the residual is float32 rounding, not geometry
                assert abs(u[i] - ref[0]) < 5e-3 and abs(v[i] - ref[1]) < 5e-3, (i, u[i], v[i], ref)
                assert inside[i] == (z[i] > 0.05 and 0 <= u[i] < HW[1] and 0 <= v[i] < HW[0])
        assert 0 < n_behind < 200                             # the random pose splits the cloud

    def test_identity_pose_known_numbers(self):
        u, v, z, inside = project_points([[0.5, 0.25, 2.0]], np.eye(4), {"fx": 10, "fy": 10, "cx": 8, "cy": 8}, (16, 16))
        assert u[0] == pytest.approx(10.5) and v[0] == pytest.approx(9.25) and z[0] == 2.0 and inside[0]

    def test_edges_and_near_plane(self):
        intr = {"fx": 10, "fy": 10, "cx": 8, "cy": 8}
        pts = [[0.0, 0.0, -1.0],        # behind: NaN pixel, not inside
               [0.0, 0.0, 0.01],        # in front but closer than z_min: pixel finite, not inside
               [0.8, 0.0, 1.0],         # u == 16 == W: outside (half-open interval)
               [0.75, 0.0, 1.0],        # u == 15.5: inside
               [-0.8, 0.0, 1.0],        # u == 0: inside (left edge is closed)
               [0.0, -0.81, 1.0]]       # v < 0: outside
        u, v, z, inside = project_points(pts, np.eye(4), intr, (16, 16))
        assert not np.isfinite(u[0]) and not inside[0]
        assert np.isfinite(u[1]) and not inside[1]
        assert u[2] == pytest.approx(16.0) and not inside[2]
        assert inside[3] and inside[4] and not inside[5]


class TestDepthSampling:
    def test_uses_the_mask_stages_coordinate_rule_and_ignores_nans(self):
        depth = np.full((6, 8), np.nan, dtype=np.float32)            # depth 6x8 under a 12x16 RGB (session1-like ratio)
        depth[3, 4] = 1.234
        # RGB pixel (u=9.0, v=7.0) -> depth (y = 7*0.5 = 3, x = 9*0.5 = 4)
        got = sample_depth(depth, [9.0, 1.0, np.nan], [7.0, 1.0, 2.0], (12, 16), window=3)
        assert got[0] == pytest.approx(1.234)
        assert np.isnan(got[1]) and np.isnan(got[2])                  # nothing finite around (0,0); NaN pixel

    def test_window_median_and_zero_depth_is_invalid(self):
        depth = np.zeros((6, 8), dtype=np.float32)
        depth[2:5, 3:6] = [[2.0, 2.1, 0.0], [2.2, 0.0, 2.3], [2.4, 2.5, 2.6]]
        got = sample_depth(depth, [9.0], [7.0], (12, 16), window=3)  # depth (3, 4): 3x3 window rows 2..4, cols 3..5
        assert got[0] == pytest.approx(np.median([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6]))

    def test_no_depth_map(self):
        assert np.isnan(sample_depth(None, [1.0], [1.0], (12, 16))).all()


class _O:
    def __init__(self, oid, xyz, confirmed=True, hits=3, stability=0.9, label="mug"):
        self.id = oid; self.xyz_world = np.asarray(xyz, dtype=np.float32); self.confirmed = confirmed
        self.hits = hits; self.stability = stability; self.label_primary = label


class TestFrustumView:
    def test_lists_only_in_frustum_objects_with_both_depths(self):
        intr = {"fx": 10, "fy": 10, "cx": 8, "cy": 6}
        depth = np.full((6, 8), np.nan, dtype=np.float32)
        depth[3, 4] = 2.5                                             # under RGB pixel (u ~ 9, v ~ 7)
        objs = [_O("front", [0.1, 0.1, 1.0]),                       # u = 9, v = 7 -> inside, observed 2.5 vs expected 1.0
                _O("behind", [0.0, 0.0, -1.0]),
                _O("outside", [5.0, 0.0, 1.0])]                     # u = 58 > W
        entries, n = frustum_view(objs, np.eye(4), intr, (12, 16), depth)
        assert n == 3 and [e["id"] for e in entries] == ["front"]
        e = entries[0]
        assert e["expected_depth"] == pytest.approx(1.0) and e["observed_depth"] == pytest.approx(2.5)
        assert (e["u"], e["v"], e["confirmed"], e["hits"], e["stability"], e["label_primary"]) == (9.0, 7.0, True, 3, 0.9, "mug")

    def test_empty_and_no_depth(self):
        assert frustum_view([], np.eye(4), INTR, HW) == ([], 0)
        entries, n = frustum_view([_O("a", [0.0, 0.0, 2.0])], np.eye(4), INTR, HW, None)
        assert n == 1 and entries[0]["observed_depth"] is None and entries[0]["expected_depth"] == 2.0
        assert FRUSTUM_MODEL_V1 == "v1_occlusion_agnostic"
