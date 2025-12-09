import os
import pytest

from rtsm.models.fastsam.adapter import FastSAMAapter

def test_fastsam_everything_mode():
    image_path = "test_dataset/rgb/1754989062.627478.png"
    if not os.path.isfile(image_path):
        pytest.skip("sample image missing")

    if FastSAMAapter is None:
        pytest.skip("fastsam package not installed; skipping everything-mode test")

    model_path = "rtsm/models/fastsam/FastSAM-x.pt"
    output_path = "test/fastsam/output.png"
    fastsam = FastSAMAapter(model_path=model_path, device="cuda")
    try:
        h.output_path = output_path
    except Exception:
        pass
    ann = h.run_fastsam_on_image(image_path)
    assert os.path.isfile(output_path)

    print(ann)


if __name__ == "__main__":
    test_fastsam_everything_mode()


