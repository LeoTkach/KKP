"""Model factory."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18

SUPPORTED_MODELS = ("resnet18", "efficientnet_b0")


def _replace_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            in_features = classifier[-1].in_features
            classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        return model
    msg = f"Cannot replace classifier for {type(model).__name__}"
    raise TypeError(msg)


def create_model(name: str, num_classes: int, *, pretrained: bool = True) -> nn.Module:
    if name not in SUPPORTED_MODELS:
        msg = f"Unsupported model: {name}. Supported: {', '.join(SUPPORTED_MODELS)}"
        raise ValueError(msg)

    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return _replace_classifier(resnet18(weights=weights), num_classes)

    if name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        return _replace_classifier(efficientnet_b0(weights=weights), num_classes)

    msg = f"Unsupported model: {name}"
    raise ValueError(msg)
