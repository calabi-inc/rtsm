"""MkDocs hook: the release number on the docs home page follows the tag, not a hand-edited string.

``{{ rtsm_version }}`` in any page is replaced at build time by the version resolved here, in this order:

1. ``RTSM_DOCS_VERSION`` in the environment: the release workflow sets it to the tag it has just published.
2. The highest stable ``v<major>.<minor>.<patch>`` tag reachable from the built commit (the docs workflow fetches the
   tags). Pre-release tags are skipped: ``pip install rtsm`` does not install them, and the page says "on PyPI".
3. The version in ``pyproject.toml``: a checkout without tags (a shallow clone, an archive, a fresh fork).

A local ``mkdocs serve`` goes through the same resolution; the chosen version and its source are logged once.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "{{ rtsm_version }}"
ENV_VAR = "RTSM_DOCS_VERSION"
STABLE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

log = logging.getLogger("mkdocs.hooks.version")


def _from_env(env: Mapping[str, str]) -> Optional[str]:
    v = (env.get(ENV_VAR) or "").strip()
    v = v[1:] if v.startswith("v") else v
    return v or None


def _from_git(root: Path) -> Optional[str]:
    """The highest stable ``v*`` tag that is an ancestor of HEAD, or None (no git, no history, no tags)."""
    try:
        out = subprocess.run(["git", "tag", "--merged", "HEAD", "--list", "v*"], cwd=root, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    best: Optional[Tuple[int, int, int]] = None
    for line in out.stdout.splitlines():
        m = STABLE_TAG.match(line.strip())
        if m:
            key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if best is None or key > best:
                best = key
    return ".".join(str(x) for x in best) if best else None


def _from_pyproject(root: Path) -> Optional[str]:
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def resolve_version(env: Optional[Mapping[str, str]] = None, root: Path = ROOT) -> Tuple[str, str]:
    """(version, source) with source one of ``env``, ``tag``, ``pyproject``; ``unknown`` when none of the three answers."""
    env = os.environ if env is None else env
    for source, fn in (("env", lambda: _from_env(env)), ("tag", lambda: _from_git(root)), ("pyproject", lambda: _from_pyproject(root))):
        v = fn()
        if v:
            return v, source
    return "unknown", "unknown"


def on_config(config):
    version, source = resolve_version()
    config.extra["rtsm_version"] = version
    log.info("release version on the pages: %s (from %s)", version, source)
    return config


def on_page_markdown(markdown: str, page, config, files) -> str:
    return markdown.replace(PLACEHOLDER, config.extra.get("rtsm_version", "unknown"))
