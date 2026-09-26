"""
A small TF buffer for bag sources (Gate 4.5 plan, P3 task 1): static and timed
transforms per (parent, child) hop, chain discovery from a camera frame up to
a world frame, and a lookup at an arbitrary stamp that interpolates each moving
hop (lerp on translation, shortest-path slerp on rotation) between its
bracketing samples. NumPy only.

Conventions (CLAUDE.md pose-math rule: every step is pinned by
``tests/test_tf_buffer.py`` against analytic answers):
- a hop ``(parent, child)`` stores ``T_parent_child`` -- the child's pose in the
  parent frame, as a ROS ``TransformStamped`` does;
- ``lookup(world, camera, t)`` returns ``T_world_camera`` = the product of the
  hops from the world down to the camera, i.e. the camera pose in the world
  frame -- what ``PoseStamped(t_wc, q_wc_xyzw)`` carries;
- frame ids are normalised by stripping a leading ``/`` (ROS 1 style), so
  ``/world`` and ``world`` are the same frame;
- quaternions are ``[x, y, z, w]``.

The buffer is built in one pass over a bag's ``/tf`` + ``/tf_static`` (and
optionally an odometry topic) before frames are read; a streaming variant with
lookahead is a later concern (live ROS sources).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rtsm.utils.transforms import rotmat_to_quat_xyzw


class TfLookupError(LookupError):
    """``reason`` is one of no_chain | no_samples | extrapolation."""

    def __init__(self, reason: str, msg: str) -> None:
        super().__init__(msg)
        self.reason = reason


def norm_frame(frame_id: Optional[str]) -> str:
    return (frame_id or "").strip().lstrip("/")


# ───────────────────────────── quaternion helpers ─────────────────────────────

def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0.0:
        raise ValueError("zero quaternion")
    return q / n


def quat_to_rotmat(q) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_slerp(q0, q1, alpha: float) -> np.ndarray:
    """Shortest-path spherical interpolation, ``alpha`` in [0, 1]."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:                                        # q and -q are the same rotation: take the short arc
        q1 = -q1
        dot = -dot
    if dot > 0.9995:                                     # nearly parallel: lerp + renormalise
        return quat_normalize(q0 + alpha * (q1 - q0))
    theta0 = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - alpha) * theta0) / np.sin(theta0)
    s1 = np.sin(alpha * theta0) / np.sin(theta0)
    return quat_normalize(s0 * q0 + s1 * q1)


def make_T(t, q) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rotmat(q)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def split_T(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(t float32 (3,), q_xyzw float32 (4,)) from a 4x4."""
    t = np.asarray(T[:3, 3], dtype=np.float32)
    q = rotmat_to_quat_xyzw(np.asarray(T[:3, :3], dtype=np.float32))
    return t, np.asarray(q, dtype=np.float32)


# ───────────────────────────── the buffer ─────────────────────────────

@dataclass
class _Hop:
    static: Optional[Tuple[np.ndarray, np.ndarray]] = None      # (t, q) when static
    stamps: List[int] = None                                    # sorted ns
    ts: List[np.ndarray] = None
    qs: List[np.ndarray] = None
    constant: Optional[bool] = None                             # timed samples that never change (a re-published calibration)

    def __post_init__(self) -> None:
        self.stamps = [] if self.stamps is None else self.stamps
        self.ts = [] if self.ts is None else self.ts
        self.qs = [] if self.qs is None else self.qs

    def is_constant(self) -> bool:
        """ROS 1 systems re-publish static calibration transforms on /tf as timed
        samples (TUM: the kinect -> optical-frame chain). Such a hop is constant:
        clamping to it is exact at any stamp, so the extrapolation limit does not apply."""
        if self.constant is None:
            if len(self.stamps) < 2:
                self.constant = False
            else:
                t0, q0 = self.ts[0], self.qs[0]
                self.constant = all(np.allclose(t, t0, atol=1e-9) and (np.allclose(q, q0, atol=1e-9) or np.allclose(q, -q0, atol=1e-9))
                                    for t, q in zip(self.ts, self.qs))
        return self.constant


class TfBuffer:
    def __init__(self, *, extrapolation_s: float = 0.05) -> None:
        self._hops: Dict[Tuple[str, str], _Hop] = {}
        self._parents: Dict[str, List[str]] = {}
        self.extrapolation_ns = int(float(extrapolation_s) * 1e9)
        self.n_static = 0
        self.n_timed = 0
        self._sorted = True

    # ── building ──

    def add(self, parent: str, child: str, t_ns: int, t, q, *, static: bool = False) -> None:
        """One ``TransformStamped``: ``T_parent_child`` at ``t_ns`` (ignored when static)."""
        p, c = norm_frame(parent), norm_frame(child)
        if not p or not c or p == c:
            raise ValueError(f"bad transform frames {parent!r} -> {child!r}")
        key = (p, c)
        hop = self._hops.get(key)
        if hop is None:
            hop = self._hops[key] = _Hop()
            self._parents.setdefault(c, []).append(p)
        tt = np.asarray(t, dtype=np.float64).reshape(3)
        qq = np.asarray(q, dtype=np.float64).reshape(4)          # stored as given (bit-exact read-back); normalised where used
        if float(np.linalg.norm(qq)) == 0.0:
            raise ValueError(f"zero quaternion on {parent!r} -> {child!r}")
        if static:
            hop.static = (tt, qq)
            self.n_static += 1
            return
        if hop.stamps and int(t_ns) < hop.stamps[-1]:
            self._sorted = False
        hop.constant = None
        hop.stamps.append(int(t_ns))
        hop.ts.append(tt)
        hop.qs.append(qq)
        self.n_timed += 1

    def _ensure_sorted(self) -> None:
        if self._sorted:
            return
        for hop in self._hops.values():
            if hop.stamps and any(b < a for a, b in zip(hop.stamps, hop.stamps[1:])):
                order = sorted(range(len(hop.stamps)), key=hop.stamps.__getitem__)
                hop.stamps = [hop.stamps[i] for i in order]
                hop.ts = [hop.ts[i] for i in order]
                hop.qs = [hop.qs[i] for i in order]
        self._sorted = True

    # ── graph ──

    def frames(self) -> List[str]:
        s = set()
        for p, c in self._hops:
            s.add(p); s.add(c)
        return sorted(s)

    def hops(self) -> List[Tuple[str, str, bool]]:
        return sorted((p, c, hop.static is not None and not hop.stamps) for (p, c), hop in self._hops.items())

    def roots(self) -> List[str]:
        """Frames that are never a child."""
        children = {c for _p, c in self._hops}
        return sorted({p for p, _c in self._hops} - children)

    def chain(self, root: str, source: str) -> List[Tuple[str, str]]:
        """Hops from ``root`` down to ``source`` (parent pointers walked up
        from the source; the first parent wins when a frame has several).
        Raises TfLookupError('no_chain')."""
        root, source = norm_frame(root), norm_frame(source)
        path: List[Tuple[str, str]] = []
        cur = source
        seen = {cur}
        while cur != root:
            parents = self._parents.get(cur) or []
            if not parents:
                raise TfLookupError("no_chain", f"no TF chain from {source!r} up to {root!r} (stuck at {cur!r}; roots: {self.roots()})")
            nxt = parents[0]
            if nxt in seen:
                raise TfLookupError("no_chain", f"TF cycle at {nxt!r}")
            path.append((nxt, cur))
            seen.add(nxt)
            cur = nxt
        path.reverse()
        return path

    # ── lookup ──

    def _hop_at(self, key: Tuple[str, str], t_ns: int) -> np.ndarray:
        hop = self._hops[key]
        if not hop.stamps:
            if hop.static is None:
                raise TfLookupError("no_samples", f"hop {key} has no samples")
            return make_T(*hop.static)
        stamps = hop.stamps
        i = bisect.bisect_left(stamps, t_ns)
        if i < len(stamps) and stamps[i] == t_ns:
            return make_T(hop.ts[i], hop.qs[i])
        if i == 0 or i == len(stamps):
            j = 0 if i == 0 else len(stamps) - 1
            if abs(int(t_ns) - stamps[j]) > self.extrapolation_ns and not hop.is_constant():
                raise TfLookupError("extrapolation", f"hop {key}: {t_ns} is {abs(t_ns - stamps[j]) / 1e6:.1f} ms outside its samples")
            return make_T(hop.ts[j], hop.qs[j])
        t0, t1 = stamps[i - 1], stamps[i]
        alpha = (int(t_ns) - t0) / float(t1 - t0)
        t = hop.ts[i - 1] + alpha * (hop.ts[i] - hop.ts[i - 1])
        q = quat_slerp(hop.qs[i - 1], hop.qs[i], alpha)
        return make_T(t, q)

    def lookup_T(self, root: str, source: str, t_ns: int) -> np.ndarray:
        """``T_root_source`` at ``t_ns`` as a 4x4 (float64)."""
        self._ensure_sorted()
        T = np.eye(4, dtype=np.float64)
        for key in self.chain(root, source):
            T = T @ self._hop_at(key, int(t_ns))
        return T

    def lookup(self, root: str, source: str, t_ns: int) -> Tuple[np.ndarray, np.ndarray]:
        """(t_wc float32 (3,), q_wc_xyzw float32 (4,)): the ``source`` frame's pose in ``root``.
        A single-hop chain with a sample exactly at ``t_ns`` returns that sample's
        translation and quaternion as stored (no 4x4 round trip), so a bag that
        carries the receiver's own pose gives it back bit for bit."""
        self._ensure_sorted()
        chain = self.chain(root, source)
        if len(chain) == 1:
            hop = self._hops[chain[0]]
            if hop.stamps:
                i = bisect.bisect_left(hop.stamps, int(t_ns))
                if i < len(hop.stamps) and hop.stamps[i] == int(t_ns):
                    return np.asarray(hop.ts[i], dtype=np.float32), np.asarray(hop.qs[i], dtype=np.float32)
            elif hop.static is not None:
                return np.asarray(hop.static[0], dtype=np.float32), np.asarray(hop.static[1], dtype=np.float32)
        return split_T(self.lookup_T(root, source, t_ns))

    def span(self, key: Tuple[str, str]) -> Optional[Tuple[int, int]]:
        hop = self._hops.get((norm_frame(key[0]), norm_frame(key[1])))
        if hop is None or not hop.stamps:
            return None
        self._ensure_sorted()
        return hop.stamps[0], hop.stamps[-1]


def odometry_buffer(samples: Sequence[Tuple[int, Sequence[float], Sequence[float]]], *, world: str = "odom",
                    body: str = "base_link", extrinsic: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                    camera: str = "camera", extrapolation_s: float = 0.05) -> TfBuffer:
    """A TfBuffer from an odometry / PoseStamped track: one moving hop
    ``world -> body`` plus, when given, a static ``body -> camera`` extrinsic.
    Lookups then interpolate exactly like a TF hop."""
    buf = TfBuffer(extrapolation_s=extrapolation_s)
    for t_ns, t, q in samples:
        buf.add(world, body, int(t_ns), t, q)
    if extrinsic is not None:
        buf.add(body, camera, 0, extrinsic[0], extrinsic[1], static=True)
    return buf
