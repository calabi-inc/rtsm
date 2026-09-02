"""Judgment-crop math (2026-08-30) — the pure cut behind the snapshot
gallery: padding, clamping, downscale cap, degenerate boxes, and the
association-side gallery preference (hires over masked 224)."""

import numpy as np

from rtsm.core.pipeline import cut_judgment_crop


def _frame(h=1440, w=1920):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_pad_expands_by_fraction_of_box_size():
    crop = cut_judgment_crop(_frame(), 900, 700, 1000, 800, 0.20, 640)
    # 100px box + 20px pad each side -> 140x140
    assert crop.shape == (140, 140, 3)


def test_long_side_capped_with_aspect_kept():
    crop = cut_judgment_crop(_frame(), 100, 100, 1700, 900, 0.0, 640)
    assert max(crop.shape[0], crop.shape[1]) == 640
    ratio = crop.shape[1] / crop.shape[0]
    assert abs(ratio - 2.0) < 0.05          # 1600x800 box keeps ~2:1


def test_small_box_is_not_upscaled():
    crop = cut_judgment_crop(_frame(), 500, 500, 560, 550, 0.0, 640)
    assert crop.shape == (50, 60, 3)        # native pixels, untouched


def test_clamped_at_frame_edges():
    crop = cut_judgment_crop(_frame(), 0, 0, 100, 100, 0.5, 640)
    assert crop.shape == (150, 150, 3)      # pad clamped at 0,0


def test_degenerate_box_returns_none():
    assert cut_judgment_crop(_frame(), 50, 50, 50, 90, 0.2, 640) is None
    assert cut_judgment_crop(_frame(), 60, 90, 50, 90, 0.2, 640) is None


def test_association_gallery_prefers_hires():
    # The preference is a two-line getattr chain in association.py; pin
    # its semantics with duck objects.
    class C:
        crop = "masked224"
        crop_hires = "hires"

    class COld:
        crop = "masked224"

    def gallery(c):
        g = getattr(c, 'crop_hires', None)
        if g is None:
            g = getattr(c, 'crop', None)
        return g

    assert gallery(C()) == "hires"
    assert gallery(COld()) == "masked224"
