"""Multiset identity for benchmark_datasheet raw JSONs -- the G0 recipe.

Same recipe as eval/baselines/2026-09-pre-reconcile/README.md and the bisect
wrapper (scratchpad/sweep_dual.py): the multiset of
(label_primary, xyz_world rounded to 1 mm, hits, confirmed) over the /objects
page the harness saved, hashed as sha256(json.dumps(sorted(...)))[:16].

Usage:
    python tools/multiset_sha.py reports/datasheet_raw_dual.json [more.json ...]

Prints objects/confirmed/frame_count, the first-100-page sha (comparable to the
recorded anchors), the full-page sha when the harness recorded `objects_full`
(post-reconcile harness only; NOT comparable to the old anchors), the gate
counters (frame_rejections / pose_conversion_failures) wherever they appear in
the JSON, and which known anchor the page sha matches.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter

ANCHORS = {
    # pre-120fe7c lineage (main 73aa8e2's perception code)
    "b71b98ca1fc2bf0d": "PRE-flip @53 frames: 107/70 (main 73aa8e2)",
    "935bc8eb254c4dfe": "PRE-flip @54 frames: 111/74 (base 96e71eb, Apr/Jun main)",
    # post-120fe7c lineage (demo2 f7a0880's / merged tree's perception code)
    "1994e0fe5dd6167c": "POST-flip @53 frames: 115/66 (demo2 f7a0880 runs 1,3)",
    "72fd5c7f0475da90": "POST-flip @54 frames: 116/68 (demo2 f7a0880 run 2, 120fe7c)",
}


def multiset(objs):
    return Counter(
        (
            o.get("label_primary"),
            tuple(round(float(v), 3) for v in (o.get("xyz_world") or [])),
            int(o.get("hits") or 0),
            bool(o.get("confirmed")),
        )
        for o in objs
    )


def ms_sha(ms):
    return hashlib.sha256(
        json.dumps(sorted(map(list, ms.elements())), default=str).encode()
    ).hexdigest()[:16]


def find_counters(node, path="", out=None):
    out = {} if out is None else out
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}" if path else k
            if k in ("frame_rejections", "pose_conversion_failures", "gate_rejections"):
                out[p] = v
            find_counters(v, p, out)
    elif isinstance(node, list):
        for i, v in enumerate(node[:3]):
            find_counters(v, f"{path}[{i}]", out)
    return out


def main(paths):
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        if "error" in d:
            print(f"{p}: ERROR {d['error']}")
            continue
        wm = d.get("working_memory") or {}
        lat = d.get("latency") or {}
        page = (d.get("objects") or {}).get("objects") or []
        sha = ms_sha(multiset(page))
        full = (d.get("objects_full") or {}).get("objects") or []
        full_sha = ms_sha(multiset(full)) if full else None
        print(f"{p}")
        print(f"  backend={d.get('backend')} extra_args={d.get('rtsm_extra_args')}")
        print(f"  objects={wm.get('objects')} confirmed={wm.get('confirmed')} "
              f"upserts_total={wm.get('upserts_total')} frame_count={lat.get('frame_count')}")
        print(f"  page multiset sha ({len(page)} objs) = {sha}  -> {ANCHORS.get(sha, 'no known anchor')}")
        if full_sha:
            print(f"  full multiset sha ({len(full)} objs) = {full_sha}  (new-anchor only)")
        print(f"  counters: {find_counters(d) or '(none found in JSON)'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1:])
