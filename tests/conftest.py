"""Test-suite root: puts ``tests/`` on the import path so test modules in
subdirectories can import the shared helpers (``_recordings``) the same way
the top-level ones do. pytest already inserts each test file's own directory."""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
