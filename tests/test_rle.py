"""COCO RLE masks (B4): the pure-numpy encoder round-trips with its decoder (and with pycocotools when it is installed), and the
obs line carries `mask.rle` only under diagnostics.save_masks."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtsm.core.datamodel import FramePacket, IngestMeta, PoseStamped, TimeBundle
from rtsm.core.pipeline import Pipeline
from rtsm.evaluation.event_log import LEDGER_SCHEMA, EventLogWriter
from rtsm.io.ingest_queue import IngestQueue
from rtsm.utils.mask_staging import MaskStats
from rtsm.utils.rle import counts_to_mask, counts_to_string, mask_to_counts, mask_to_rle, rle_to_mask, string_to_counts


def _masks():
    rng = np.random.default_rng(7)
    yield np.zeros((6, 9), dtype=bool)
    yield np.ones((6, 9), dtype=bool)
    m = np.zeros((5, 7), dtype=bool); m[1:4, 2:5] = True; yield m
    yield rng.random((48, 64)) > 0.7
    yield rng.random((480, 640)) > 0.5
    m = np.zeros((3, 4), dtype=bool); m[0, 0] = True; m[2, 3] = True; yield m
    yield np.array([[1, 0], [0, 1]], dtype=bool)


def test_counts_are_fortran_runs_zeros_first():
    m = np.array([[1, 0], [0, 1]], dtype=bool)                   # Fortran flat: 1 0 0 1
    assert mask_to_counts(m) == [0, 1, 2, 1]
    assert mask_to_counts(np.zeros((2, 3), dtype=bool)) == [6] and mask_to_counts(np.ones((2, 3), dtype=bool)) == [0, 6]
    assert counts_to_mask([0, 1, 2, 1], (2, 2)).tolist() == m.tolist()


def test_round_trip_through_the_compressed_string():
    for m in _masks():
        rle = mask_to_rle(m)
        assert rle["size"] == [m.shape[0], m.shape[1]] and isinstance(rle["counts"], str)
        back = rle_to_mask(rle)
        assert back.dtype == bool and back.shape == m.shape and np.array_equal(back, m)
        assert string_to_counts(rle["counts"]) == mask_to_counts(m)
    # a torch tensor encodes the same as its numpy view; a list of counts decodes too
    t = torch.zeros(5, 7, dtype=torch.bool); t[1:4, 2:5] = True
    assert mask_to_rle(t) == mask_to_rle(t.numpy()) and np.array_equal(rle_to_mask({"size": [5, 7], "counts": mask_to_counts(t.numpy())}), t.numpy())
    assert counts_to_string([0]) == "0" and string_to_counts("0") == [0]


def test_malformed_counts_are_refused():
    with pytest.raises(ValueError):
        counts_to_mask([0, 1], (2, 2))                              # does not cover the mask
    with pytest.raises(ValueError):
        counts_to_mask([5], (2, 2))                                 # does not fit
    with pytest.raises(ValueError):
        mask_to_counts(np.zeros((2, 2, 2), dtype=bool))


def test_matches_pycocotools_when_installed():
    mask_util = pytest.importorskip("pycocotools.mask")
    for m in _masks():
        ref = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
        ours = mask_to_rle(m)
        assert ours["counts"] == ref["counts"].decode("ascii") and ours["size"] == list(ref["size"])
        assert np.array_equal(mask_util.decode(ref).astype(bool), rle_to_mask(ours))


def _packet(seq=7, ts_ns=123_000):
    return FramePacket(rgb=np.zeros((8, 8, 3), dtype=np.uint8), depth_m=np.ones((8, 8), dtype=np.float32),
                       pose=PoseStamped(stamp_ns=ts_ns, frame_id="arkit", t_wc=np.array([1.0, 2.0, 3.0], dtype=np.float32),
                                        q_wc_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)),
                       intr=None, is_keyframe=True, frame_epoch=4,
                       time=TimeBundle(t_mono_s=0.0, t_wall_utc_s=0.0, t_sensor_ns=ts_ns, seq=seq),
                       ingest=IngestMeta(lane="keyframe", keyframe_origin="minted", rx_seq=9))


def _cand():
    m = torch.zeros(6, 8, dtype=torch.bool); m[1:4, 2:6] = True
    st = MaskStats(idx=3, area_px=12, bbox=(2, 1, 6, 4), coverage=0.25, border_fraction=0.0, depth_valid=1.0, depth_p50=2.0, depth_spread=0.0, centroid_px=(4.0, 2.5))
    return SimpleNamespace(stats=st, mask=m, priority=0.7, label_topk=[("cup", 0.9)]), m


@pytest.mark.parametrize("on", [True, False])
def test_obs_line_carries_the_mask_only_when_saving(tmp_path: Path, on: bool):
    w = EventLogWriter(enabled=True, configured_path=str(tmp_path / "e.jsonl"), ledgers=True)
    pipe = Pipeline(cfg={"diagnostics": {"save_masks": on}}, segmenter=None, clip=None, working_mem=None, proximity_index=None, associator=None,
                    ingest_gate=None, ingest_q=IngestQueue(), sweep_cache=None, event_log=w)
    cand, m = _cand()
    pipe._write_obs_lines(_packet(), [dict(cand=cand, outcome="no_p_cam")], mask_hw=(6, 8))
    w.close()
    rows = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert rows[0]["ledgers"]["schema"] == LEDGER_SCHEMA == 5
    obs = [r for r in rows if r["kind"] == "obs"][0]
    if on:
        rle = obs["mask"]["rle"]
        assert rle["size"] == [6, 8] and np.array_equal(rle_to_mask(rle), m.numpy())
        assert int(rle_to_mask(rle).sum()) == obs["mask"]["area_px"] == 12       # the mask's pixel count is the ledger's area_px
    else:
        assert "rle" not in obs["mask"]
    assert obs["mask"]["bbox"] == [2, 1, 6, 4] and obs["mask"]["mask_hw"] == [6, 8]


def test_eval_save_masks_flag_and_config(tmp_path):
    pytest.importorskip("rosbags", reason="the [eval] extra is not installed")
    from rtsm.cfg import load_config
    from rtsm.evaluation import runner as R
    from test_bag_reader import simple_bag_messages, write_bag
    bag = write_bag(tmp_path / "bag", simple_bag_messages(4, seq_topic=True))
    cfg = load_config()
    r = R.resolve_eval(cfg, R.EvalOptions(input=str(bag), save_masks=True))
    assert r.save_masks is True and R.configure_run(cfg, r, tmp_path / "run_1")["diagnostics"]["save_masks"] is True
    r0 = R.resolve_eval(cfg, R.EvalOptions(input=str(bag)))
    assert r0.save_masks is False and R.configure_run(cfg, r0, tmp_path / "run_1")["diagnostics"]["save_masks"] is False
    cfg2 = json.loads(json.dumps(cfg)); cfg2["diagnostics"]["save_masks"] = True
    assert R.resolve_eval(cfg2, R.EvalOptions(input=str(bag))).save_masks is True
    assert "--save-masks" in R.build_parser().format_help()

