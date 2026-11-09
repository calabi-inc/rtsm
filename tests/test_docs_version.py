"""The docs home page states the release that is on PyPI; the number comes from the tag through the MkDocs hook, never from the page."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import rtsm

REPO = Path(__file__).resolve().parents[1]
HOOK = REPO / "docs" / "hooks" / "version.py"


def _hook():
    spec = importlib.util.spec_from_file_location("docs_version_hook", HOOK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_home_page_carries_the_placeholder_and_no_hand_written_release_number():
    text = (REPO / "docs" / "index.md").read_text(encoding="utf-8")
    assert text.count("{{ rtsm_version }}") == 2                       # the pill and the meta strip
    hero = text.split("## What RTSM is")[0]
    assert not re.search(r"\b(RTSM|rtsm)\s+\d+\.\d+\.\d+", hero)       # the 0.2.0 that went stale


def test_hook_is_registered_and_the_hook_directory_stays_out_of_the_site():
    cfg = (REPO / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^hooks:\s*\n\s+- docs/hooks/version\.py", cfg, re.M)
    assert re.search(r"^exclude_docs:.*\n\s+hooks/", cfg, re.M)


def test_resolution_order_env_then_tag_then_pyproject(tmp_path, monkeypatch):
    h = _hook()
    # 1. the release workflow's variable wins and a leading v is dropped
    assert h.resolve_version({"RTSM_DOCS_VERSION": "v9.9.9"}, root=REPO) == ("9.9.9", "env")
    assert h.resolve_version({"RTSM_DOCS_VERSION": " 1.2.3 "}, root=REPO) == ("1.2.3", "env")
    # 2. on this checkout the tag answer is a stable version or nothing (a shallow CI clone carries no tags)
    tag = h._from_git(REPO)
    assert tag is None or re.fullmatch(r"\d+\.\d+\.\d+", tag)
    # 3. without git and without the variable, pyproject answers, and it agrees with the package
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rtsm"\nversion = "0.0.7"\n', encoding="utf-8")
    assert h.resolve_version({}, root=tmp_path) == ("0.0.7", "pyproject")
    assert h._from_pyproject(REPO) == rtsm.__version__
    # nothing at all: a visible marker rather than an exception
    assert h.resolve_version({}, root=tmp_path / "nowhere") == ("unknown", "unknown")


def test_stable_tags_only_and_the_highest_wins(monkeypatch):
    h = _hook()

    def fake_run(*a, **k):
        return SimpleNamespace(returncode=0, stdout="v0.3.1\nv0.10.0\nv0.4.0rc1\nv0.9.2\nnot-a-tag\n")

    monkeypatch.setattr(h.subprocess, "run", fake_run)
    assert h._from_git(REPO) == "0.10.0"                                # numeric order, not string order; rc skipped


def test_page_markdown_substitution():
    h = _hook()
    config = SimpleNamespace(extra={})
    h.resolve_version = lambda: ("4.5.6", "env")                         # the resolved value lands in config.extra
    h.on_config(config)
    assert config.extra["rtsm_version"] == "4.5.6"
    out = h.on_page_markdown("<strong>RTSM {{ rtsm_version }}</strong> and rtsm {{ rtsm_version }}", None, config, None)
    assert out == "<strong>RTSM 4.5.6</strong> and rtsm 4.5.6"
