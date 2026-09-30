"""Guards for tests that need the reference recording on disk.

``recordings/session1/messages.bin`` is a Git LFS object (1.03 GB). A checkout
without LFS (CI, a fresh clone before ``git lfs pull``) leaves a ~130-byte
pointer file in its place, so ``Path.is_file()`` is True while the bytes are
garbage. ``real_recording`` says whether the file is the actual recording.
"""
from __future__ import annotations

from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com"
MIN_REAL_BYTES = 1 << 20      # a real Lens recording is far above 1 MiB


def is_lfs_pointer(path: Path) -> bool:
    try:
        with Path(path).open("rb") as fh:
            return fh.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except OSError:
        return False


def real_recording(path: Path) -> bool:
    """True iff ``path`` exists, is larger than 1 MiB and is not an LFS pointer."""
    p = Path(path)
    try:
        if not p.is_file() or p.stat().st_size < MIN_REAL_BYTES:
            return False
    except OSError:
        return False
    return not is_lfs_pointer(p)
