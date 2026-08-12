"""
Thin RTSM REST customer — the agent's ONLY window into RTSM.

Boundary rule (locked): the agent consumes RTSM through its public REST API
exactly like any external user would. No rtsm imports, no internals.

Endpoints used (shapes verified against rtsm/api/server.py):
  GET /healthz          -> {"status": "ok"}
  GET /stats            -> {objects, confirmed, avg_hits, upserts_total,
                            robot_pose: {xyz, quaternion_xyzw, timestamp} | None}
  GET /search/semantic  -> {query, robot_pose, results: [{id, score, confirmed,
                            stability, xyz_world, snapshot_b64?}]}

`robot_pose` updates at WebSocket receive time (~5 Hz input rate, PR #20);
`PoseSample.fetched_at_mono` is stamped locally so the monitor can measure
staleness on OUR clock and detect a frozen feed by a non-advancing
`timestamp` across polls (never by comparing sensor time to wall time).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

_HEADERS = {"Connection": "close"}


@dataclass(frozen=True)
class PoseSample:
    xyz: List[float]                 # ARKit world, Y up
    quaternion_xyzw: List[float]     # camera orientation, OpenCV convention
    timestamp: float                 # sender wall clock (FramePacket)
    fetched_at_mono: float           # OUR time.monotonic() at response
    # RTSM receiver's world-frame discontinuity counter (bumps when the
    # sender starts a new streaming session — poses across a bump must not
    # be compared in one frame). None = server predates the field, or the
    # pose came from an epoch-less path (ZMQ/replay, post-/reset transient);
    # the monitor's equality gate fires only when BOTH sides are ints.
    frame_epoch: Optional[int] = None


@dataclass(frozen=True)
class SemanticHit:
    id: str
    score: float
    confirmed: bool
    stability: float
    xyz_world: Optional[List[float]]
    # Server wall clock of the object's last observation (None on servers
    # predating the field). The baseline's freshness gate compares this to
    # the agent's own time.time() — valid because RTSM and the agent run
    # on the SAME machine (locked demo topology).
    last_seen_wall_utc: Optional[float] = None


@dataclass(frozen=True)
class SemanticResult:
    query: str
    robot_pose: Optional[PoseSample]
    results: List[SemanticHit]


class RtsmClient:
    def __init__(self, base_url: str, timeout_s: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    # ── low-level ────────────────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        r = requests.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout_s,
            headers=_HEADERS,
        )
        r.raise_for_status()
        return r.json()

    @staticmethod
    def _parse_pose(raw: Optional[Dict[str, Any]]) -> Optional[PoseSample]:
        if not raw:
            return None
        xyz = raw.get("xyz")
        quat = raw.get("quaternion_xyzw")
        ts = raw.get("timestamp")
        if xyz is None or quat is None or ts is None:
            return None
        try:
            xyz_f = [float(v) for v in xyz]
            quat_f = [float(v) for v in quat]
            ts_f = float(ts)
        except (TypeError, ValueError):
            return None
        # A malformed pose is treated as NO pose — it flows through the
        # monitor's bounded stale-hold path instead of ever reaching the
        # geometry with a quaternion that can't rotate anything.
        if len(xyz_f) != 3 or len(quat_f) != 4:
            return None
        if sum(v * v for v in quat_f) < 1e-12:       # zero-norm quaternion
            return None
        epoch = raw.get("frame_epoch")   # tolerate missing key AND null
        return PoseSample(
            xyz=xyz_f,
            quaternion_xyzw=quat_f,
            timestamp=ts_f,
            fetched_at_mono=time.monotonic(),
            frame_epoch=int(epoch) if epoch is not None else None,
        )

    # ── public API ───────────────────────────────────────────────────────

    def healthz(self) -> bool:
        """True iff RTSM answers /healthz with status ok. Never raises."""
        try:
            return self._get("/healthz").get("status") == "ok"
        except requests.RequestException:
            return False

    def stats(self) -> Dict[str, Any]:
        """Raw /stats payload (object counts + robot_pose). Raises on HTTP error."""
        return self._get("/stats")

    def object_count(self) -> int:
        return int(self.stats().get("objects", 0))

    def get_robot_pose(self) -> Optional[PoseSample]:
        """Latest robot pose via /stats, or None if no frames received yet."""
        return self._parse_pose(self.stats().get("robot_pose"))

    def get_object_label(self, object_id: str) -> Optional[str]:
        """Best-effort label for one object via GET /objects/{id}
        (label_primary — the semantic search response carries no labels).
        Returns None on any failure; the planner treats labels as optional."""
        try:
            detail = self._get(f"/objects/{object_id}")
            label = detail.get("label_primary")
            return str(label) if label else None
        except (requests.RequestException, ValueError):
            return None

    def get_object_snapshot_b64(self, object_id: str) -> Optional[str]:
        """Base64 JPEG of the object's most recent observation crop via
        GET /objects/{id}/snapshots/0/image, or None on any failure —
        snapshots are best-effort evidence, never a blocker. Added after
        t20260811-200758-001: the captioner labeled the teddy bear 'audio'
        and the label-only selector rejected the RIGHT object mid-search;
        the crop is the ground truth the labels approximate."""
        try:
            r = requests.get(
                f"{self.base_url}/objects/{object_id}/snapshots/0/image",
                timeout=self.timeout_s,
            )
            r.raise_for_status()
            import base64
            return base64.standard_b64encode(r.content).decode("ascii")
        except (requests.RequestException, ValueError):
            return None

    def semantic_query(self, query: str, top_k: int = 5) -> SemanticResult:
        """One /search/semantic call — carries BOTH results and robot_pose,
        which is the client-side MemorySlice assembly (one atomic snapshot)."""
        data = self._get("/search/semantic", params={"query": query, "top_k": top_k})
        hits = [
            SemanticHit(
                id=str(e.get("id")),
                score=float(e.get("score", 0.0)),
                confirmed=bool(e.get("confirmed", False)),
                stability=float(e.get("stability", 0.0)),
                xyz_world=(
                    [float(v) for v in e["xyz_world"]]
                    if e.get("xyz_world") is not None
                    else None
                ),
                last_seen_wall_utc=(
                    float(e["last_seen_wall_utc"])
                    if e.get("last_seen_wall_utc") is not None
                    else None
                ),
            )
            for e in data.get("results", [])
        ]
        return SemanticResult(
            query=str(data.get("query", query)),
            robot_pose=self._parse_pose(data.get("robot_pose")),
            results=hits,
        )
