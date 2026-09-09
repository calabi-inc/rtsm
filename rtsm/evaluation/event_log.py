"""
Phase 0 diagnostic event log: append-only JSONL writer.

One line per processed frame. Off by default; opt in via cfg.diagnostics.enabled.
Each pipeline run gets a fresh, auto-timestamped file (no appending across runs).

Path resolution rules:
  - None / unset  -> eval_output/<YYYYMMDD_HHMMSS>/events.jsonl  (per-run, auto)
  - "foo/bar/"    -> foo/bar/<YYYYMMDD_HHMMSS>/events.jsonl     (per-run inside dir)
  - "foo/bar.jsonl" -> exact path (overwrites with warning if exists)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameEvent:
    """One JSONL line. Numbers in milliseconds where named *_ms."""
    timestamp: float
    frame_seq: int
    is_keyframe: bool
    n_masks_raw: int
    filter: Dict[str, Any] = field(default_factory=dict)
    scoring: Dict[str, Any] = field(default_factory=dict)
    n_matched: int = 0
    n_created: int = 0
    n_objects_confirmed: int = 0
    timing_ms: Dict[str, float] = field(default_factory=dict)


def _json_default(o: Any) -> Any:
    """JSON encoder fallback for numpy scalars / arrays."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _resolve_path(configured: Optional[str], repo_root: Path) -> Path:
    """See module docstring for rules."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if configured is None or str(configured).strip() == "":
        return repo_root / "eval_output" / ts / "events.jsonl"

    p = Path(configured)
    if not p.is_absolute():
        p = repo_root / p

    if str(configured).endswith("/") or str(configured).endswith("\\") or p.is_dir():
        return p / ts / "events.jsonl"

    return p  # explicit file path


class EventLogWriter:
    """Append-only JSONL writer for per-frame diagnostic events.

    When `enabled=False`, all calls are zero-cost no-ops.
    """

    def __init__(self, enabled: bool, configured_path: Optional[str], repo_root: Optional[Path] = None):
        self._enabled = bool(enabled)
        self._fh = None
        self._path: Optional[Path] = None
        if not self._enabled:
            return

        root = repo_root if repo_root is not None else Path.cwd()
        self._path = _resolve_path(configured_path, root)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._path.exists():
            logger.warning(f"event_log: overwriting existing file {self._path}")

        # WRITE mode (truncate) — each run starts fresh. Line-buffered so a
        # crash mid-run still preserves prior frames.
        self._fh = self._path.open("w", encoding="utf-8", buffering=1)
        logger.info(f"event_log: writing diagnostics to {self._path}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def write(self, event: FrameEvent) -> None:
        if not self._enabled or self._fh is None:
            return
        self._fh.write(json.dumps(asdict(event), default=_json_default) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "EventLogWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def summarize_sources(entries) -> Dict[str, int]:
    """Helper: count entries by confirmation_source."""
    counts: Dict[str, int] = {"dual": 0, "fastsam_only": 0, "yoloe_only": 0, "none": 0}
    for e in entries:
        src = e.confirmation_source or "none"
        counts[src] = counts.get(src, 0) + 1
    return counts
