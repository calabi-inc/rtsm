"""COCO run-length encoding of a binary mask, in pure numpy: the compressed string form pycocotools writes (``counts`` as a
string, ``size`` as ``[h, w]``), plus the decoder the readers need. Runs are taken over the mask in column-major (Fortran)
order, the first run counting zeros (possibly zero of them); from the third count on each count is delta-coded against the
one two back, and every count is written as 5-bit groups, low group first, with bit 0x20 as the continuation flag and the
characters offset by 48 (``rleToString`` / ``rleFrString`` in the COCO API). An ``rle`` block decodes with pycocotools,
FiftyOne or this module alike.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def mask_to_counts(mask: Any) -> List[int]:
    """Run lengths of ``mask`` (2-D, anything truthy is foreground) in Fortran order, zeros first."""
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"a mask is 2-D; got shape {m.shape}")
    flat = m.flatten(order="F").astype(np.int8)
    n = int(flat.size)
    if n == 0:
        return [0]
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], change, [n]))
    runs = [int(r) for r in np.diff(bounds)]
    return ([0] + runs) if flat[0] else runs


def counts_to_string(counts: Sequence[int]) -> str:
    out: List[str] = []
    for i, c in enumerate(counts):
        x = int(c)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            ch = x & 0x1F
            x >>= 5
            more = (x != -1) if (ch & 0x10) else (x != 0)
            if more:
                ch |= 0x20
            out.append(chr(ch + 48))
    return "".join(out)


def string_to_counts(s: str) -> List[int]:
    counts: List[int] = []
    i, n = 0, len(s)
    while i < n:
        x, k, more = 0, 0, True
        while more:
            if i >= n:
                raise ValueError("truncated RLE string")
            c = ord(s[i]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            i += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(int(x))
    return counts


def counts_to_mask(counts: Sequence[int], size: Sequence[int]) -> np.ndarray:
    h, w = int(size[0]), int(size[1])
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        c = int(c)
        if c < 0 or pos + c > h * w:
            raise ValueError("run lengths do not fit the mask size")
        if val and c:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    if pos != h * w:
        raise ValueError("run lengths do not cover the mask size")
    return flat.reshape((h, w), order="F")


def mask_to_rle(mask: Any) -> Dict[str, Any]:
    """``{"size": [h, w], "counts": "<compressed>"}`` for a 2-D mask (a numpy array or a torch tensor)."""
    if hasattr(mask, "detach"):                                   # a torch tensor
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask).astype(bool)
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts_to_string(mask_to_counts(m))}


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """The boolean mask of an ``rle`` block (compressed string counts, or a list of counts)."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = string_to_counts(counts)
    return counts_to_mask(counts, rle["size"])
