"""`rtsm demo` startup validation runs BEFORE any model loads and its imports
are bound before their first use (a function-local import placed after the
validation block left `LaneConfig` an unbound local: `rtsm demo` crashed at
startup on main between PR #33 and P1 task 4). CPU-only: a deliberately bad
config value must exit through parser.error, which proves the block ran with
its names bound and never reached the segmentation loader.
"""
from __future__ import annotations

import pytest

import rtsm.demo as demo


@pytest.fixture
def no_models(monkeypatch, tmp_path):
    monkeypatch.setattr(demo, "_find_demo_data", lambda: str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("model loading must not be reached when the config is invalid")
    import rtsm.models.segmentation as seg
    monkeypatch.setattr(seg, "get_segmenter", boom)


@pytest.mark.parametrize("bad", [
    "robot_pose.stale_after_s=0",
    "ingest.policy=newest",
    "ingest.keyframe_lane_depth=0",
])
def test_bad_ingest_or_pose_config_exits_before_model_load(no_models, bad, capsys):
    with pytest.raises(SystemExit) as ex:
        demo.run_demo(["--no-viz", "--set", bad])
    assert ex.value.code == 2                                    # argparse parser.error
    err = capsys.readouterr().err
    assert "robot_pose" in err or "ingest" in err


def test_validation_names_are_bound_before_use():
    """Static pin: the two validators must not be function-local names that are
    assigned only after the validation block (Python would compile the early
    reference as an unbound local)."""
    import dis
    code = demo.run_demo.__code__
    first_load, first_store = {}, {}
    for ins in dis.get_instructions(code):
        if ins.argval in ("LaneConfig", "resolve_pose_stale_after_s"):
            if ins.opname.startswith("LOAD_") and ins.argval not in first_load:
                first_load[ins.argval] = ins.offset
            if ins.opname.startswith("STORE_") and ins.argval not in first_store:
                first_store[ins.argval] = ins.offset
    for name in ("LaneConfig", "resolve_pose_stale_after_s"):
        assert name in first_load, name
        if name in first_store:                                   # function-local import: must precede the load
            assert first_store[name] < first_load[name], f"{name} is used before its import binds it"
