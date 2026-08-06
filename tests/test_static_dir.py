"""
Tests for static frontend directory discovery (rtsm.utils.static_dir).

Guards against the 2026-08-06 stale-dist incident: a leftover demo/dist/
whose old index.html referenced hashed assets that no longer existed
shadowed the good packaged bundle, so the dashboard served a shell that
404'd on every JS/CSS file (infinite loading). find_static_dir() must
reject a candidate whose index.html references missing assets/ files,
warn loudly, and fall through to the next candidate — while keeping the
dev-build-wins search order and logging which dir it ends up serving.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rtsm.utils.static_dir import find_static_dir

JS = "index-AAAA1111.js"
CSS = "index-BBBB2222.css"


def _make_frontend(root: Path, *, with_assets: bool = True) -> Path:
    """Create a Vite-style build: index.html referencing hashed assets."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(
        "<!doctype html>\n<html>\n<head>\n"
        f'<script type="module" crossorigin src="/assets/{JS}"></script>\n'
        f'<link rel="stylesheet" crossorigin href="/assets/{CSS}">\n'
        '</head>\n<body><div id="app"></div></body>\n</html>\n',
        encoding="utf-8",
    )
    (root / "assets").mkdir(exist_ok=True)
    if with_assets:
        (root / "assets" / JS).write_text("// js", encoding="utf-8")
        (root / "assets" / CSS).write_text("/* css */", encoding="utf-8")
    return root


class TestFindStaticDir:
    def test_broken_dev_falls_through_to_packaged(self, tmp_path, caplog):
        # The incident: dev dist's index.html references deleted assets
        dev = _make_frontend(tmp_path / "demo" / "dist", with_assets=False)
        pkg = _make_frontend(tmp_path / "static")
        with caplog.at_level(logging.WARNING, logger="rtsm.utils.static_dir"):
            result = find_static_dir(dev_dist=dev, pkg_static=pkg)
        assert result == str(pkg.resolve())
        warnings = "\n".join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert str(dev.resolve()) in warnings
        assert JS in warnings and CSS in warnings

    def test_valid_dev_wins_over_packaged(self, tmp_path):
        dev = _make_frontend(tmp_path / "demo" / "dist")
        pkg = _make_frontend(tmp_path / "static")
        assert find_static_dir(dev_dist=dev, pkg_static=pkg) == str(dev.resolve())

    def test_served_dir_logged_with_asset_hashes(self, tmp_path, caplog):
        pkg = _make_frontend(tmp_path / "static")
        with caplog.at_level(logging.INFO, logger="rtsm.utils.static_dir"):
            find_static_dir(dev_dist=tmp_path / "no-such-dir", pkg_static=pkg)
        info = "\n".join(r.getMessage() for r in caplog.records)
        assert str(pkg.resolve()) in info
        assert JS in info and CSS in info

    def test_dir_without_index_falls_through(self, tmp_path):
        dev = tmp_path / "demo" / "dist"
        dev.mkdir(parents=True)  # dir exists, no index.html
        pkg = _make_frontend(tmp_path / "static")
        assert find_static_dir(dev_dist=dev, pkg_static=pkg) == str(pkg.resolve())

    def test_no_valid_candidate_returns_none(self, tmp_path):
        dev = _make_frontend(tmp_path / "demo" / "dist", with_assets=False)
        assert find_static_dir(dev_dist=dev, pkg_static=tmp_path / "static") is None

    def test_index_without_asset_refs_is_accepted(self, tmp_path):
        pkg = tmp_path / "static"
        pkg.mkdir()
        (pkg / "index.html").write_text(
            "<!doctype html><html><body>inline build</body></html>", encoding="utf-8"
        )
        result = find_static_dir(dev_dist=tmp_path / "no-such-dir", pkg_static=pkg)
        assert result == str(pkg.resolve())
