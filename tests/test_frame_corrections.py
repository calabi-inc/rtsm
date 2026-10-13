"""Stage 5 of the session registry: the Lens pose_corrections text message as the
session registry sees it. An empty batch is announced-only and still produces
the FrameCorrection event and the batch callback; kind and map_ref travel with
the batch; a one-argument callback keeps working.
"""
from __future__ import annotations

import json

import numpy as np

from rtsm.io.contracts import CORRECTION_KINDS, FrameCorrection
from rtsm.io.websocket import corrections_pose_to_4x4, handle_lens_text_message

IDENT16 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


class Sinks:
    def __init__(self) -> None:
        self.batches = []
        self.events = []
        self.singles = []

    def batch(self, batch, **kw):
        self.batches.append((dict(batch), dict(kw)))

    def run(self, msg: dict, **extra):
        handle_lens_text_message(json.dumps(msg), apply_camera_flip=False, session_id="S1",
                                 on_pose_corrections=lambda k, p: self.singles.append(k),
                                 on_pose_corrections_batch=self.batch, on_frame_correction=self.events.append, **extra)


def test_kinds_constant():
    assert CORRECTION_KINDS == ("loop_closure", "relocalized_in_prior_map", "reset", "unknown")


def test_empty_corrections_forwards_event_and_batch():
    s = Sinks()
    s.run({"type": "pose_corrections", "corrections": {}, "kind": "relocalized_in_prior_map", "map_ref": "A"})
    assert len(s.events) == 1
    ev = s.events[0]
    assert isinstance(ev, FrameCorrection) and ev.kind == "relocalized_in_prior_map" and ev.map_ref == "A"
    assert ev.corrections == {} and ev.session_id == "S1"
    assert s.batches == [({}, {"kind": "relocalized_in_prior_map", "session_id": "S1", "map_ref": "A"})]
    assert s.singles == []


def test_missing_corrections_field_is_announced_only_too():
    s = Sinks()
    s.run({"type": "pose_corrections", "kind": "loop_closure"})
    assert s.batches == [({}, {"kind": "loop_closure", "session_id": "S1", "map_ref": None})]


def test_kind_validation_and_default():
    s = Sinks()
    s.run({"type": "pose_corrections", "corrections": {}, "kind": "bogus"})
    s.run({"type": "pose_corrections", "corrections": {}})
    assert [b[1]["kind"] for b in s.batches] == ["unknown", "loop_closure"]


def test_nonempty_batch_unchanged():
    s = Sinks()
    shifted = list(IDENT16)
    shifted[14] = 0.25                                   # column-major: translation z
    s.run({"type": "pose_corrections", "corrections": {"ws_3": IDENT16, "ws_4": shifted}})
    (batch, kw), = s.batches
    assert sorted(batch) == ["ws_3", "ws_4"] and kw["kind"] == "loop_closure" and kw["map_ref"] is None
    assert np.allclose(batch["ws_4"][:3, 3], [0.0, 0.0, 0.25])
    assert s.events[0].corrections.keys() == batch.keys()
    assert s.singles == []                               # the batch callback wins over the per-correction one


def test_all_malformed_poses_produce_nothing():
    s = Sinks()
    s.run({"type": "pose_corrections", "corrections": {"ws_3": [1.0, 2.0]}})
    assert s.batches == [] and s.events == []


def test_one_argument_batch_callback_still_works():
    seen = []
    handle_lens_text_message(json.dumps({"type": "pose_corrections", "corrections": {"ws_1": IDENT16}}),
                             apply_camera_flip=False, on_pose_corrections_batch=lambda b: seen.append(sorted(b)))
    assert seen == [["ws_1"]]


def test_per_correction_fallback_when_no_batch_callback():
    singles = []
    handle_lens_text_message(json.dumps({"type": "pose_corrections", "corrections": {"ws_1": IDENT16, "ws_2": IDENT16}}),
                             apply_camera_flip=False, on_pose_corrections=lambda k, p: singles.append(k))
    assert sorted(singles) == ["ws_1", "ws_2"]


def test_other_text_messages_are_ignored():
    s = Sinks()
    s.run({"type": "hello", "session_id": "x"})
    assert s.batches == [] and s.events == []


def test_pose_7_form():
    m = corrections_pose_to_4x4([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0])
    assert np.allclose(m[:3, 3], [1.0, 2.0, 3.0]) and np.allclose(m[:3, :3], np.eye(3))
    assert corrections_pose_to_4x4([1.0, 2.0]) is None
