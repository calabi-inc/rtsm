"""
Smoke test: verify all model files exist and can be loaded from model_store/.

Marked ``models`` + ``gpu``: it needs the local weights, ultralytics (AGPL,
not in the default ``[gpu]`` extra) and builds the configured segmenter on the
configured device. The canonical command deselects it; opt in with

    python -m pytest tests/test_model_loading.py -v -m "models or gpu"

Paths in the packaged rtsm.yaml are repo-root-relative (model_store/...), so
every test runs with cwd = repo root -- via a fixture, not an import-time
os.chdir that would leak into every later test in the session.
"""
import os

import pytest

from rtsm.cfg import load_config as _load_config

pytestmark = [pytest.mark.models, pytest.mark.gpu]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _repo_root_cwd(monkeypatch):
    """Resolve the yaml's relative model paths against the repo root."""
    monkeypatch.chdir(ROOT)


def test_yoloe_model_exists():
    """YOLOE model file exists at the configured path."""
    cfg = _load_config()
    path = cfg["segmentation"]["yoloe"]["model_path"]
    assert os.path.isfile(path), f"YOLOE model not found: {path}"
    assert os.path.getsize(path) > 1_000_000, f"YOLOE model too small (corrupt?): {path}"
    print(f"  YOLOE model OK: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def test_fastsam_model_exists():
    """FastSAM model file exists at the configured path."""
    cfg = _load_config()
    path = cfg["segmentation"]["fastsam"]["model_path"]
    assert os.path.isfile(path), f"FastSAM model not found: {path}"
    assert os.path.getsize(path) > 1_000_000, f"FastSAM model too small (corrupt?): {path}"
    print(f"  FastSAM model OK: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def test_clip_model_exists():
    """CLIP model file exists at the configured path."""
    cfg = _load_config()
    local_dir = cfg["clip"]["local_dir"]
    model_name = cfg["clip"]["model"]
    pretrained = cfg["clip"]["pretrained"]
    ckpt = os.path.join(local_dir, f"{model_name}-{pretrained}", "model.pt")
    assert os.path.isfile(ckpt), f"CLIP model not found: {ckpt}"
    assert os.path.getsize(ckpt) > 1_000_000, f"CLIP model too small (corrupt?): {ckpt}"
    print(f"  CLIP model OK: {ckpt} ({os.path.getsize(ckpt) / 1e6:.1f} MB)")


def test_yoloe_loads():
    """YOLOE model can be loaded by ultralytics."""
    cfg = _load_config()
    path = cfg["segmentation"]["yoloe"]["model_path"]
    from ultralytics import YOLOE
    model = YOLOE(path)
    assert model is not None
    print(f"  YOLOE loaded successfully from {path}")


def test_fastsam_loads():
    """FastSAM model can be loaded by ultralytics."""
    cfg = _load_config()
    path = cfg["segmentation"]["fastsam"]["model_path"]
    from ultralytics import FastSAM
    model = FastSAM(path)
    assert model is not None
    print(f"  FastSAM loaded successfully from {path}")


def test_clip_loads():
    """CLIP model can be loaded via rtsm loader."""
    cfg = _load_config()
    from rtsm.models.clip.loader import load_clip
    model, preprocess, tokenizer = load_clip(
        cfg["clip"]["model"],
        cfg["clip"]["pretrained"],
        cfg["clip"]["local_dir"],
        device="cpu",
    )
    assert model is not None
    assert preprocess is not None
    assert tokenizer is not None
    print(f"  CLIP loaded successfully from {cfg['clip']['local_dir']}")


def test_segmenter_factory():
    """get_segmenter() resolves all paths for the CONFIGURED backend."""
    cfg = _load_config()
    from rtsm.models.segmentation import get_segmenter
    segmenter = get_segmenter(cfg)
    assert segmenter is not None
    # The packaged default is grounded_sam2 (was hard-coded to "dual" here,
    # which made this test red on every default checkout).
    assert segmenter.name == cfg["segmentation"]["backend"]
    print(f"  Segmenter factory OK: backend={segmenter.name}")


if __name__ == "__main__":
    import sys

    os.chdir(ROOT)
    tests = [
        test_yoloe_model_exists,
        test_fastsam_model_exists,
        test_clip_model_exists,
        test_yoloe_loads,
        test_fastsam_loads,
        test_clip_loads,
        test_segmenter_factory,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__doc__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__doc__}\n  {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
