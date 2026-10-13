#!/usr/bin/env python
"""Clone a Lens recording and give its later parts new session ids.

The clone replays as one stream whose later parts arrive under other session
ids, exactly as a device that reconnected with a new session would: the
replay source re-announces the session at each split (a ``hello`` text message
at that point of the timeline), the ingest front-end bumps the frame epoch,
and the session registry freezes the part before and opens the next
(`GET /sessions` then lists every part; the dashboard's picker shows them all,
each with its share of the map).

    python scripts/split_recording_session.py recordings/session1 recordings/session1_split
    python scripts/split_recording_session.py SRC DST --at 0.5 --session-id 7B2C...   # fraction of the frames
    python scripts/split_recording_session.py SRC DST --at 0.33 --at 0.66            # three sessions

``messages.bin`` (the frames, gigabytes) is hard-linked, not copied, when the
file system allows it; ``index.jsonl`` and ``meta.json`` are copied with a
``split`` note; ``text_messages.jsonl`` gets the hello (existing text messages
are kept in order).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def split_recording(src: Path, dst: Path, *, at: Any = 0.5, session_id: Any = None,
                    device_name: Optional[str] = None) -> Dict[str, Any]:
    """Write the clone; returns a summary dict (also written into DST/meta.json under "split").

    ``at`` is one fraction or a list of fractions (strictly increasing, each in (0, 1)); ``session_id``
    one id or a list of the same length (missing ones become fresh UUIDs)."""
    src = Path(src)
    dst = Path(dst)
    if not (src / "index.jsonl").is_file() or not (src / "messages.bin").is_file():
        raise FileNotFoundError(f"{src} is not a recording (index.jsonl + messages.bin)")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty")
    ats: List[float] = [float(a) for a in (at if isinstance(at, (list, tuple)) else [at])]
    if not ats or any(not 0.0 < a < 1.0 for a in ats) or any(b <= a for a, b in zip(ats, ats[1:])):
        raise ValueError("--at must be fractions strictly between 0 and 1, in increasing order")
    sids_in: List[Optional[str]] = list(session_id) if isinstance(session_id, (list, tuple)) else [session_id]
    if len(sids_in) < len(ats):
        sids_in += [None] * (len(ats) - len(sids_in))
    if len(sids_in) > len(ats):
        raise ValueError("more --session-id values than --at values")
    index = _read_jsonl(src / "index.jsonl")
    if len(index) < len(ats) + 1:
        raise ValueError("the recording needs at least one frame per part to split")
    positions: List[int] = []
    for a in ats:
        pos = max(1, min(len(index) - 1, int(round(len(index) * a))))
        if positions and pos <= positions[-1]:
            pos = positions[-1] + 1
            if pos >= len(index):
                raise ValueError("split points fall on the same frame; use fewer or more spread out --at values")
        positions.append(pos)

    meta: Dict[str, Any] = {}
    if (src / "meta.json").is_file():
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    first_sid = meta.get("session_id") or (meta.get("hello") or {}).get("session_id") or src.name
    proto = int((meta.get("hello") or {}).get("protocol_version", 1))
    dev = device_name or meta.get("device_name") or "split-clone"
    splits: List[Dict[str, Any]] = []
    for i, pos in enumerate(positions):
        entry = index[pos]
        sid = sids_in[i] or str(uuid.uuid4()).upper()
        splits.append({"index": pos, "seq": int(entry.get("seq", pos)), "t_mono_s": round(float(entry["t_mono_s"]) - 1e-6, 6),
                       "session_id": sid,
                       "hello": {"type": "hello", "protocol_version": proto, "session_id": sid, "device_name": dev}})

    dst.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src / "messages.bin", dst / "messages.bin")
        linked = True
    except OSError:
        shutil.copy2(src / "messages.bin", dst / "messages.bin")
        linked = False
    shutil.copy2(src / "index.jsonl", dst / "index.jsonl")
    texts = _read_jsonl(src / "text_messages.jsonl")
    for sp in splits:
        texts.append({"t_mono_s": sp["t_mono_s"], "after_seq": sp["seq"], "payload": json.dumps(sp["hello"])})
    texts.sort(key=lambda e: float(e["t_mono_s"]))
    with open(dst / "text_messages.jsonl", "w", encoding="utf-8") as fh:
        for e in texts:
            fh.write(json.dumps(e) + "\n")
    bounds = [0] + positions + [len(index)]
    part_ids = [first_sid] + [sp["session_id"] for sp in splits]
    parts = [{"session_id": part_ids[i], "frames": bounds[i + 1] - bounds[i], "first_index": bounds[i]}
             for i in range(len(bounds) - 1)]
    first = splits[0]
    summary = {"from": str(src), "at": ats if len(ats) > 1 else ats[0], "n_sessions": len(parts), "parts": parts,
               "splits": [{k: v for k, v in sp.items() if k != "hello"} for sp in splits],
               # the two-way names, kept for the first split
               "split_index": first["index"], "split_seq": first["seq"], "t_mono_s": first["t_mono_s"],
               "frames_first_half": first["index"], "frames_second_half": len(index) - first["index"],
               "first_session_id": first_sid, "second_session_id": first["session_id"],
               "messages_bin": "hard link" if linked else "copy"}
    meta["split"] = summary
    (dst / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="the recording to clone (a folder with index.jsonl and messages.bin)")
    ap.add_argument("dst", help="the clone to write (must not exist or be empty)")
    ap.add_argument("--at", type=float, action="append", default=None,
                    help="fraction of the frames before a split; repeat for several sessions (default 0.5)")
    ap.add_argument("--session-id", action="append", default=None,
                    help="session id for the part after a split, in --at order (default: new UUIDs)")
    ap.add_argument("--device-name", default=None, help="the device name in the hello (default: the recording's)")
    args = ap.parse_args(argv)
    try:
        summary = split_recording(Path(args.src), Path(args.dst), at=(args.at or [0.5]), session_id=(args.session_id or []),
                                  device_name=args.device_name)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    print(f"\nreplay it:  python -m rtsm --viz --replay {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
