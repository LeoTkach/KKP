"""Model factory."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

SUPPORTED_MODELS = ("resnet18",)


def create_model(name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED_MODELS:
        msg = f"Unsupported model: {name}. Supported: {', '.join(SUPPORTED_MODELS)}"
        raise ValueError(msg)

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    msg = f"Unsupported model: {name}"
    raise ValueError(msg)
