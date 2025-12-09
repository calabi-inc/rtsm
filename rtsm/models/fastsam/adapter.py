from __future__ import annotations
# import torch
from PIL import Image
from rtsm.models.fastsam.model import FastSAM
from rtsm.models.fastsam.prompt import FastSAMPrompt
import logging


class FastSAMAdapter:
    def __init__(self, model_path: str, device: str = 'cuda'):
        # Use local FastSAM implementation in rtsm/models/fastsam
        self.model = FastSAM(model_path)
        self.device = device
        self.output_path = "test/fastsam/output.png"

    def segment_everything(self, image: Image.Image):
        everything_results = self.model(
            image,
            device=self.device,
            retina_masks=False,
            imgsz=640,
            conf=0.4,
            iou=0.9,
            verbose=False
        )
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        ann = prompt_process.everything_prompt()

        return ann

    def run_fastsam_on_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        everything_results = self.model(
            image,
            device=self.device,
            retina_masks=False,
            imgsz=640,
            conf=0.4,
            iou=0.9
        )
        prompt_process = FastSAMPrompt(image, everything_results, device=self.device)
        ann = prompt_process.everything_prompt()

        return ann


# # quick test
# if __name__ == "__main__":
#     helper = FastSAMAdapter(model_path="model_store/fastsam/FastSAM-x.pt", device="cuda")
#     ann = helper.run_fastsam_on_image("test_dataset/rgb/1754989062.627478.png")






