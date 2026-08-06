"""
Static frontend directory discovery with asset validation.

The dashboard frontend is a Vite build: index.html referencing hashed
bundles under assets/. A stale dev build (demo/dist/ left over with an old
index.html whose hashed assets no longer exist) can shadow the good
packaged bundle — the server then delivers a shell whose JS/CSS 404,
and the dashboard loads forever. Candidates are therefore validated
before being accepted: every assets/ file referenced by index.html must
exist on disk, otherwise the candidate is skipped with a loud warning.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Asset references in a built index.html, e.g. src="/assets/index-DP70BXmG.js"
_ASSET_RE = re.compile(r'(?:src|href)="/?(assets/[^"]+)"')


def _validate(candidate: Path, label: str) -> list[str] | None:
    """Return the asset paths referenced by candidate's index.html if they all
    exist on disk, else None (candidate unusable)."""
    index_html = candidate / "index.html"
    if not candidate.is_dir() or not index_html.is_file():
        return None
    try:
        html = index_html.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Static dir %s (%s): cannot read index.html (%s) — skipping",
                       candidate.resolve(), label, e)
        return None
    assets = _ASSET_RE.findall(html)
    missing = [a for a in assets if not (candidate / a).is_file()]
    if missing:
        logger.warning(
            "STALE static dir %s (%s): index.html references missing files %s — "
            "skipping it. Rebuild (npm run build) or delete the directory.",
            candidate.resolve(), label, ", ".join(missing),
        )
        return None
    return assets


def find_static_dir(dev_dist: Path | None = None, pkg_static: Path | None = None) -> str | None:
    """Locate the built frontend static directory.

    Search order (dev build wins so you never need to copy):
      1. demo/dist/ (dev, after npm run build)
      2. rtsm/static/ (packaged release)

    A candidate is accepted only if every assets/ file its index.html
    references exists on disk; broken candidates fall through with a warning.
    """
    if dev_dist is None:
        dev_dist = Path("demo/dist")
    if pkg_static is None:
        pkg_static = Path(__file__).resolve().parent.parent / "static"

    for candidate, label in ((dev_dist, "dev build"), (pkg_static, "packaged")):
        assets = _validate(candidate, label)
        if assets is None:
            continue
        resolved = candidate.resolve()
        logger.info("Serving frontend from %s (%s; assets: %s)",
                    resolved, label,
                    ", ".join(a.rsplit("/", 1)[-1] for a in assets) or "none referenced")
        return str(resolved)
    return None
