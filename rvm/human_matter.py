import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from .model import MattingNetwork


class HumanMatter:
    def __init__(self, checkpoint_path: str, device=None, threshold: float = 0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MattingNetwork("resnet50").to(self.device).eval()
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.threshold: float = threshold

    def __call__(self, image: np.ndarray):
        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        # Начальные скрытые состояния (для одиночного изображения)
        rec = [None] * 4

        with torch.no_grad():
            fgr, pha, *rec = self.model(image_tensor, *rec)

        mask = pha[0, 0].cpu().numpy()
        mask = (mask > self.threshold).astype(np.uint8) * 255
        return mask