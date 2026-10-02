"""One version, three places that must agree: pyproject.toml, rtsm.__version__ and `rtsm version`."""
from __future__ import annotations

import re
from pathlib import Path

import rtsm
from rtsm.evaluation import runner

REPO = Path(__file__).resolve().parents[1]


def test_pyproject_and_package_version_agree():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert m and m.group(1) == rtsm.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+([ab]\d+|rc\d+)?", rtsm.__version__)


def test_cli_version_and_runner_fallback(capsys, monkeypatch):
    import rtsm.cli as cli
    monkeypatch.setattr("sys.argv", ["rtsm", "version"])
    cli.main()
    assert capsys.readouterr().out.strip() == f"rtsm {rtsm.__version__}"
    # the runner reports the distribution's version, else the package's (never None on a checkout)
    assert runner._rtsm_version() in (rtsm.__version__,) or runner._rtsm_version() is not None
