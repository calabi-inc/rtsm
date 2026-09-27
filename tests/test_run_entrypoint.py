"""`python -m rtsm` (rtsm.run.main) startup block, on CPU (P1 task 6; the second
half of task 4's lesson: every runner entry point needs a CPU test that
executes its startup block).

The ingest validation (clock, `LaneConfig`, `robot_pose.stale_after_s`) now
sits ABOVE the GPU-dependency check, so a bad value exits through
`parser.error` on any machine, before the check and long before a model
loads. `main()` takes `argv` (the console entry still calls it bare). The
deprecated `io.websocket.*` paths are aliased on this path too.
"""
from __future__ import annotations

import dis
import logging
import sys
import warnings

import pytest

import rtsm.run as run


@pytest.fixture
def no_models(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model loading must not be reached when the config is invalid")
    # run.py binds get_segmenter at module import; patch the module attribute
    # (patching rtsm.models.segmentation, as the demo test does, would be inert here).
    monkeypatch.setattr(run, "get_segmenter", boom)


@pytest.mark.parametrize("bad", [
    "ingest.policy=newest",
    "ingest.keyframe_lane_depth=0",
    "ingest.keyframe_every_n=0",
    "ingest.pair_window_fps=0",
    "robot_pose.stale_after_s=0",
    "diagnostics.ledgers=1",              # P2: must be a bool
    "diagnostics.ledger_format=csv",      # P2: jsonl | parquet
])
def test_bad_ingest_or_pose_config_exits_before_the_gpu_check(no_models, bad, tmp_path, capsys):
    with pytest.raises(SystemExit) as ex:
        run.main(["--replay", str(tmp_path), "--set", bad])
    assert ex.value.code == 2                                    # argparse parser.error
    captured = capsys.readouterr()                               # read ONCE (the buffers drain)
    assert "ingest" in captured.err or "robot_pose" in captured.err or "diagnostics" in captured.err
    assert "GPU dependencies" not in captured.out


def test_gpu_check_runs_after_validation_at_runtime(no_models, tmp_path, monkeypatch, capsys):
    """Runtime proof of the reorder on any box: with the GPU flag forced off, a
    VALID config reaches the check and returns (no model touched), a BAD one
    exits through parser.error before the check prints anything."""
    monkeypatch.setattr(run, "_GPU_AVAILABLE", False)
    monkeypatch.setattr(run, "_GPU_IMPORT_ERROR", "simulated for the test")
    assert run.main(["--replay", str(tmp_path)]) is None
    assert "GPU dependencies" in capsys.readouterr().out
    with pytest.raises(SystemExit) as ex:
        run.main(["--replay", str(tmp_path), "--set", "ingest.policy=newest"])
    captured = capsys.readouterr()
    assert ex.value.code == 2 and "ingest" in captured.err and "GPU dependencies" not in captured.out


def test_unknown_ingest_source_exits_before_the_gpu_check(no_models, capsys):
    """P3 task 0.5: io.receiver must name a registered source (built-in or an
    `rtsm.sources` entry point); a typo exits with the known names above the
    GPU check, no model touched. --replay is exempt (it always replays)."""
    with pytest.raises(SystemExit) as ex:
        run.main(["--set", "io.receiver=nosuch"])
    assert ex.value.code == 2
    err = capsys.readouterr().err
    assert "nosuch" in err and "websocket" in err and "zeromq" in err and "replay" in err


def test_deprecated_websocket_path_is_aliased_on_the_runner_path(no_models, tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="rtsm.cfg"), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(SystemExit) as ex:
            run.main(["--replay", str(tmp_path),
                      "--set", "io.websocket.keyframe_every_n=5",
                      "--set", "ingest.policy=newest"])
    assert ex.value.code == 2
    assert any(issubclass(x.category, DeprecationWarning) and "io.websocket.keyframe_every_n" in str(x.message) for x in w)
    assert any("io.websocket.keyframe_every_n" in r.getMessage() and "ingest.keyframe_every_n" in r.getMessage()
               for r in caplog.records)


def test_main_without_argv_reads_sys_argv(no_models, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rtsm", "--replay", str(tmp_path), "--set", "ingest.policy=newest"])
    with pytest.raises(SystemExit) as ex:
        run.main()
    assert ex.value.code == 2 and "ingest" in capsys.readouterr().err


def test_demo_dispatch_works_through_argv(capsys):
    with pytest.raises(SystemExit) as ex:
        run.main(["demo", "--help"])
    assert ex.value.code == 0
    assert "--no-viz" in capsys.readouterr().out


def test_validation_precedes_the_gpu_check():
    """Bytecode pin of the reorder: the first LaneConfig load comes before the
    first _GPU_AVAILABLE load in main()."""
    first = {}
    for ins in dis.get_instructions(run.main.__code__):
        if ins.opname.startswith("LOAD_") and ins.argval in ("LaneConfig", "_GPU_AVAILABLE", "resolve_clock_mode") \
                and ins.argval not in first:
            first[ins.argval] = ins.offset
    assert {"LaneConfig", "_GPU_AVAILABLE", "resolve_clock_mode"} <= set(first), first
    assert first["resolve_clock_mode"] < first["_GPU_AVAILABLE"]
    assert first["LaneConfig"] < first["_GPU_AVAILABLE"]


def test_viz_and_no_viz_are_mutually_exclusive(no_models, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run.main(["--replay", str(tmp_path), "--viz", "--no-viz"])
    assert "not allowed with argument" in capsys.readouterr().err
