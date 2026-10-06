"""Shutdown closes the model adapters without noise: ``CLIPAdapter.close()``
exists and is idempotent; ``Pipeline.shutdown`` tolerates an adapter that has
no ``close`` (third-party, stub) and logs nothing for it. Both without loading
a model: the adapter is built around a stub, the pipeline around stubs for the
four attributes ``shutdown`` touches."""
from __future__ import annotations

import logging
from types import SimpleNamespace as NS

from rtsm.core.pipeline import Pipeline
from rtsm.models.clip.adapter import CLIPAdapter, ClipArtifacts


def _adapter_with_stub_model() -> CLIPAdapter:
    a = object.__new__(CLIPAdapter)                      # no model load
    a.artifacts = ClipArtifacts(model=object(), preprocess=object(), tokenizer=object())
    a.device = "cpu"
    a._prompt_wrap = False
    return a


def test_clip_adapter_close_drops_the_model_and_is_idempotent():
    a = _adapter_with_stub_model()
    assert a.artifacts.model is not None
    a.close()
    assert a.artifacts.model is None and a.artifacts.preprocess is None and a.artifacts.tokenizer is None
    a.close()                                             # second call: no error


class _Closable:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def _pipeline_with(clip, segmenter) -> Pipeline:
    p = object.__new__(Pipeline)                          # no models, no threads
    p.ingest_q = NS()                                     # no close(): skipped
    p.segmenter = segmenter
    p.clip = clip
    p._event_log = _Closable()
    return p


def test_shutdown_closes_adapters_that_can_close_and_is_quiet_about_the_rest(caplog):
    seg = _Closable()
    p = _pipeline_with(clip=NS(), segmenter=seg)         # a CLIP stand-in without close()
    with caplog.at_level(logging.WARNING, logger="rtsm.core.pipeline"):
        p.shutdown()
    assert seg.closed == 1 and p._event_log.closed == 1
    assert "Failed to close" not in caplog.text


def test_shutdown_reports_a_close_that_raises_and_continues(caplog):
    class Boom:
        def close(self):
            raise RuntimeError("no")
    p = _pipeline_with(clip=Boom(), segmenter=_Closable())
    with caplog.at_level(logging.WARNING, logger="rtsm.core.pipeline"):
        p.shutdown()
    assert "Failed to close CLIP adapter" in caplog.text and p._event_log.closed == 1
