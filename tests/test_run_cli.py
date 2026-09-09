"""`python -m rtsm` runner CLI surface (argparse only; no models, no server).

Kept as a subprocess test on purpose: rtsm.run builds its parser inside
main(), and importing the module pulls the runtime stack.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _help() -> str:
    r = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "rtsm", "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120, check=False,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_runner_help_lists_no_viz_and_config_flags():
    out = _help()
    # --no-viz: headless runs (benchmarks, eval, CI) must be able to skip the
    # visualization server and its browser auto-open without editing the yaml.
    assert "--no-viz" in out
    # PR #25's config surface must survive any runner refactor.
    for flag in ("--config", "--profile", "--set", "--replay", "--replay-speed"):
        assert flag in out, flag
