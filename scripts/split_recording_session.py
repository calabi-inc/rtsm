#!/usr/bin/env python
"""Clone a Lens recording and give its later part a new session id.

The clone replays as one stream whose second half arrives under another
session id, exactly as a device that reconnected with a new session would:
the replay source re-announces the session at the split (a ``hello`` text
message at that point of the timeline), the ingest front-end bumps the frame
epoch, and the session registry freezes the first half and opens the second
(`GET /sessions` then lists two sessions; the dashboard's picker shows both,
each with its half of the map).

    python scripts/split_recording_session.py recordings/session1 recordings/session1_split
    python scripts/split_recording_session.py SRC DST --at 0.5 --session-id 7B2C...   # fraction of the frames

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


def split_recording(src: Path, dst: Path, *, at: float = 0.5, session_id: Optional[str] = None,
                    device_name: Optional[str] = None) -> Dict[str, Any]:
    """Write the clone; returns a summary dict (also written into DST/meta.json under "split")."""
    src = Path(src)
    dst = Path(dst)
    if not (src / "index.jsonl").is_file() or not (src / "messages.bin").is_file():
        raise FileNotFoundError(f"{src} is not a recording (index.jsonl + messages.bin)")
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"{dst} exists and is not empty")
    if not 0.0 < at < 1.0:
        raise ValueError("--at must be a fraction strictly between 0 and 1")
    index = _read_jsonl(src / "index.jsonl")
    if len(index) < 2:
        raise ValueError("the recording needs at least two frames to split")
    pos = max(1, min(len(index) - 1, int(round(len(index) * at))))
    split_entry = index[pos]
    t_split = float(split_entry["t_mono_s"]) - 1e-6            # sorts just before the first frame of the second half
    seq_split = int(split_entry.get("seq", pos))
    new_sid = session_id or str(uuid.uuid4()).upper()

    meta: Dict[str, Any] = {}
    if (src / "meta.json").is_file():
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    first_sid = meta.get("session_id") or (meta.get("hello") or {}).get("session_id") or src.name
    hello = {"type": "hello", "protocol_version": int((meta.get("hello") or {}).get("protocol_version", 1)),
             "session_id": new_sid, "device_name": device_name or meta.get("device_name") or "split-clone"}

    dst.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src / "messages.bin", dst / "messages.bin")
        linked = True
    except OSError:
        shutil.copy2(src / "messages.bin", dst / "messages.bin")
        linked = False
    shutil.copy2(src / "index.jsonl", dst / "index.jsonl")
    texts = _read_jsonl(src / "text_messages.jsonl")
    texts.append({"t_mono_s": round(t_split, 6), "after_seq": seq_split, "payload": json.dumps(hello)})
    texts.sort(key=lambda e: float(e["t_mono_s"]))
    with open(dst / "text_messages.jsonl", "w", encoding="utf-8") as fh:
        for e in texts:
            fh.write(json.dumps(e) + "\n")
    summary = {"from": str(src), "at": at, "split_index": pos, "split_seq": seq_split, "t_mono_s": round(t_split, 6),
               "frames_first_half": pos, "frames_second_half": len(index) - pos,
               "first_session_id": first_sid, "second_session_id": new_sid, "messages_bin": "hard link" if linked else "copy"}
    meta["split"] = summary
    (dst / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="the recording to clone (a folder with index.jsonl and messages.bin)")
    ap.add_argument("dst", help="the clone to write (must not exist or be empty)")
    ap.add_argument("--at", type=float, default=0.5, help="fraction of the frames before the split (default 0.5)")
    ap.add_argument("--session-id", default=None, help="the second half's session id (default: a new UUID)")
    ap.add_argument("--device-name", default=None, help="the device name in the hello (default: the recording's)")
    args = ap.parse_args(argv)
    try:
        summary = split_recording(Path(args.src), Path(args.dst), at=args.at, session_id=args.session_id,
                                  device_name=args.device_name)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    print(f"\nreplay it:  python -m rtsm --viz --replay {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
