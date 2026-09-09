"""Freeze guard for the sealed E1 paper tree.

``examples/rc_car_agent/paper/**`` was sealed at the demo2 tip ``f7a0880``
(E1 protocol, campaign runbook, 121 per-trial JSONLs, the frozen
``results.json`` and the analysis / figure scripts).  This test pins the
*committed* git tree object of that directory, so any change inside it --
edit, addition, deletion, rename or mode flip -- fails the suite until
``FROZEN_TREE_OID`` is updated on purpose, in the same commit, with the
paper-side justification.

Why a tree OID rather than per-file hashes
------------------------------------------
* one constant covers all 134 files, their names and their modes;
* it is read through git plumbing (``git rev-parse HEAD:<path>``), i.e. from
  the committed content, so working-tree line endings (the dev box has
  ``core.autocrlf=true``), editors and untracked scratch files cannot raise
  false alarms.  All 134 sealed blobs are LF-only text (verified at f7a0880),
  so a no-op ``git add`` under autocrlf cannot silently rewrite them either;
* ``.gitattributes`` only touches ``recordings/**/*.bin`` on both sides, so no
  filter is involved for this subtree.

Skips (does not fail) only when this file is *not* inside a git checkout at
all (sdist / pip-installed copy: no ``.git`` next to ``tests/``).  Inside a
checkout every other problem -- git missing from PATH, ``rev-parse`` refusing
(dubious ownership), a foreign toplevel, a non-SHA-1 object format -- is a
FAILURE, so a gate run can never report green because the check was skipped.
Fails when the path is missing from HEAD -- that is exactly the deletion this
guard exists to catch.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PAPER_PATH = "examples/rc_car_agent/paper"
FROZEN_AT_COMMIT = "f7a0880"
# git rev-parse f7a0880:examples/rc_car_agent/paper
FROZEN_TREE_OID = "81fec2ce91883e87e09cc774565ba520e925ba95"

# tests/ sits directly under the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run git against REPO_ROOT (never the caller's cwd) and capture text."""
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _problem(msg: str) -> None:
    """Skip outside a checkout, FAIL inside one (a skipped gate is not green)."""
    if (REPO_ROOT / ".git").exists():
        pytest.fail(f"frozen-tree check cannot run inside the checkout: {msg}")
    pytest.skip(msg)


@pytest.fixture(scope="module")
def git_checkout() -> Path:
    """Resolve the RTSM git checkout; skip only when there is none."""
    if shutil.which("git") is None:
        _problem("git executable not on PATH; the frozen-tree check needs git plumbing")

    top = _git("rev-parse", "--show-toplevel")
    if top.returncode != 0:
        # 'not a git repository', 'dubious ownership', shallow oddities: all
        # mean we cannot read committed trees, so we cannot judge the freeze.
        _problem(f"not inside a git checkout: {top.stderr.strip() or 'git rev-parse failed'}")

    toplevel = Path(top.stdout.strip()).resolve()
    if toplevel != REPO_ROOT:
        _problem(
            f"test file lives in {REPO_ROOT} but git toplevel is {toplevel}; "
            "not the RTSM checkout, refusing to judge the freeze"
        )

    fmt = _git("rev-parse", "--show-object-format")
    if fmt.returncode == 0 and fmt.stdout.strip() != "sha1":
        _problem(f"repository object format is {fmt.stdout.strip()!r}; pinned OID is SHA-1")

    return toplevel


def test_paper_tree_is_frozen(git_checkout: Path) -> None:
    """HEAD:examples/rc_car_agent/paper must be byte-identical to the seal."""
    head = _git("rev-parse", "--verify", "-q", f"HEAD:{PAPER_PATH}")
    assert head.returncode == 0, (
        f"{PAPER_PATH} is missing from HEAD's tree. The sealed E1 paper tree "
        f"(demo2 {FROZEN_AT_COMMIT}, tree {FROZEN_TREE_OID}) must stay in the repo; "
        f"git said: {head.stderr.strip() or '(no stderr)'}"
    )
    actual = head.stdout.strip()
    if actual == FROZEN_TREE_OID:
        return

    # Explain *what* moved so the failure is actionable.
    changes = _git("diff-tree", "-r", "--name-status", FROZEN_TREE_OID, actual)
    detail = changes.stdout.strip() if changes.returncode == 0 else changes.stderr.strip()
    pytest.fail(
        f"{PAPER_PATH} changed since the seal at demo2 {FROZEN_AT_COMMIT}.\n"
        f"  expected tree {FROZEN_TREE_OID}\n"
        f"  HEAD has      {actual}\n"
        f"Changed entries (status\\tpath):\n{detail or '(git diff-tree gave no detail)'}\n"
        "The paper tree is frozen for the E1 write-up. If this change is intentional, "
        "update FROZEN_TREE_OID in this file in the same commit and record why in the "
        "paper changelog; otherwise revert the paper/** edits."
    )


def test_paper_tree_has_no_uncommitted_tracked_changes(git_checkout: Path) -> None:
    """Gate hygiene: no modified / staged / deleted *tracked* paper files.

    Untracked files are deliberately ignored (``--untracked-files=no``): they do
    not alter the committed tree, and the figure scripts write outside paper/.
    With core.autocrlf=true and LF-only blobs, CRLF working copies compare
    clean, so this cannot trip on line endings alone.
    """
    status = _git("status", "--porcelain", "--untracked-files=no", "--", PAPER_PATH)
    assert status.returncode == 0, f"git status failed: {status.stderr.strip()}"
    dirty = status.stdout.strip()
    assert dirty == "", (
        f"tracked files under {PAPER_PATH} have uncommitted changes:\n{dirty}\n"
        "The paper tree is frozen; revert these before the gate."
    )
