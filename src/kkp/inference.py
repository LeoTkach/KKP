"""Single-image inference helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose


def predict_image(
    model: nn.Module,
    image: Image.Image,
    transform: Compose,
    device: torch.device,
    class_names: tuple[str, ...],
) -> tuple[str, dict[str, float]]:
    model.eval()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]

    confidences = {name: float(probabilities[index]) for index, name in enumerate(class_names)}
    label = max(confidences, key=confidences.get)
    return label, confidences
