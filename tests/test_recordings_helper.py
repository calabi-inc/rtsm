"""The LFS-pointer guard the session1-dependent tests skip on (P3 task 4, CI)."""
from __future__ import annotations

from pathlib import Path

from _recordings import MIN_REAL_BYTES, is_lfs_pointer, real_recording


def test_pointer_small_and_missing_files_are_not_recordings(tmp_path: Path):
    pointer = tmp_path / "messages.bin"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 1030946377\n")
    assert is_lfs_pointer(pointer) and not real_recording(pointer)
    small = tmp_path / "small.bin"
    small.write_bytes(b"\x00" * 1024)
    assert not is_lfs_pointer(small) and not real_recording(small)
    assert not real_recording(tmp_path / "absent.bin")
    assert not real_recording(tmp_path)                       # a directory


def test_large_binary_is_a_recording(tmp_path: Path):
    big = tmp_path / "messages.bin"
    with big.open("wb") as fh:
        fh.truncate(MIN_REAL_BYTES + 1)
    assert real_recording(big) and not is_lfs_pointer(big)
