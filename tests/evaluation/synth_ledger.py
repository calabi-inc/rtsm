"""Synthetic ledger rows (P2): dicts with exactly the schema-1 keys the writers
produce, for reader / rollup tests that must not depend on a receiver. Not a
test module (no test_ prefix); P3's metric tests reuse it.

    rows = meta_row() + pose_rows(300, hz=30, gaps=[(100, 2.0)], limited=[(150, 5)],
                                  jumps=[(220, 2.0)], conf2=0.6)
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from rtsm.evaluation.event_log import LEDGER_SCHEMA, SCHEMA_VERSION, TS_NORMAL


def meta_row(*, ledgers: bool = True, ingest_clock: str = "sensor", ingest_policy: str = "lossless") -> List[dict]:
    return [{
        "kind": "meta", "schema_version": SCHEMA_VERSION, "created_wall_utc_s": 1.7e9, "created_mono_s": 100.0,
        "pid": 1, "ingest_clock": ingest_clock, "ingest_policy": ingest_policy,
        "ledgers": ({"enabled": True, "schema": LEDGER_SCHEMA, "format": "jsonl"} if ledgers else {"enabled": False}),
    }]


def pose_rows(n: int, *, hz: float = 30.0, source: str = "replay", epoch: int = 1, t0_ns: int = 1_000_000_000,
              speed_mps: float = 0.2, gaps: Sequence[Tuple[int, float]] = (), limited: Sequence[Tuple[int, int]] = (),
              jumps: Sequence[Tuple[int, float]] = (), conf2: Optional[float] = 0.5, depth_valid: float = 0.9,
              rx_seq0: int = 1, conf_px: int = 256 * 192) -> List[dict]:
    """``n`` pose lines of a camera moving along +x at ``speed_mps`` at ``hz``.

    gaps    [(index, seconds)]: the interval BEFORE row ``index`` is ``seconds``
            instead of 1/hz (a stall in the stream)
    limited [(start, length)]: rows start..start+length-1 have
            tracking_state "limited" (no depth statistics, pose still present)
    jumps   [(index, metres)]: row ``index`` teleports +``metres`` along x
            (and the rows after it keep the offset)
    conf2   share of confidence-2 pixels on every line (None = no map)
    """
    limited_idx = set()
    for start, length in limited:
        limited_idx.update(range(start, start + length))
    gap_at = {i: s for i, s in gaps}
    jump_at = {i: m for i, m in jumps}
    rows: List[dict] = []
    t_ns = t0_ns
    x = 0.0
    dt = 1.0 / hz
    for i in range(n):
        if i > 0:
            step = gap_at.get(i, dt)
            t_ns += int(round(step * 1e9))
            x += speed_mps * step
        if i in jump_at:
            x += jump_at[i]
        lim = i in limited_idx
        hist = None
        if conf2 is not None and not lim:
            n2 = int(round(conf2 * conf_px))
            n1 = int(round((1.0 - conf2) * conf_px * 0.6))
            hist = [conf_px - n1 - n2, n1, n2]
        rows.append({
            "kind": "pose", "timestamp": 100.0 + i * dt, "source": source, "rx_seq": rx_seq0 + i,
            "frame_seq": i + 1, "t_sensor_ns": t_ns, "t_wall_utc_s": 1.7e9 + i * dt, "pose_clock": "sender",
            "epoch": epoch, "tracking_state": ("limited" if lim else TS_NORMAL), "mailbox_write": (not lim),
            "t_wc": [round(x, 6), 0.0, 1.5], "q_wc_xyzw": [0.0, 0.0, 0.0, 1.0], "pose_error": None,
            "depth_valid_frac": (None if lim else depth_valid), "conf_hist": hist,
        })
    return rows


def receiver_rows_for(pose: Iterable[dict], *, keyframe_every_n: int = 30, throttle_every: int = 2) -> List[dict]:
    """A receiver line per pose line with the same rx_seq: keyframes and every
    ``throttle_every``-th non-keyframe are enqueued, the others throttled."""
    out: List[dict] = []
    for k, p in enumerate(pose, 1):
        is_kf = (k == 1 or k % keyframe_every_n == 0)
        throttled = (not is_kf) and (k % throttle_every != 0)
        out.append({
            "kind": "receiver", "timestamp": p["timestamp"], "source": p["source"],
            "decision": ("dropped" if throttled else "enqueued"), "reason": ("throttle" if throttled else ""),
            "frame_seq": p["frame_seq"], "t_sensor_ns": p["t_sensor_ns"], "is_keyframe": is_kf,
            "frame_count": k, "queue_depth": 0, "depth_valid_frac": p["depth_valid_frac"],
            "lane": (None if throttled else ("keyframe" if is_kf else "fifo")), "rx_seq": p["rx_seq"],
        })
    return out
