from __future__ import annotations

import torch
from PIL import Image

from kkp.data.transforms import get_transforms
from kkp.inference import predict_image
from kkp.models.factory import create_model


def test_predict_image_returns_label_and_confidences() -> None:
    model = create_model("resnet18", num_classes=2, pretrained=False)
    transform = get_transforms(32, train=False)
    image = Image.new("RGB", (32, 32), (255, 0, 0))
    class_names = ("real", "ai_generated")

    label, confidences = predict_image(
        model,
        image,
        transform,
        torch.device("cpu"),
        class_names,
    )

    assert label in class_names
    assert set(confidences) == set(class_names)
    assert abs(sum(confidences.values()) - 1.0) < 1e-5


def test_predict_image_batch_norm_eval_mode() -> None:
    model = create_model("resnet18", num_classes=2, pretrained=False)
    model.train()
    transform = get_transforms(32, train=False)
    image = Image.new("RGB", (32, 32), (0, 255, 0))

    predict_image(model, image, transform, torch.device("cpu"), ("real", "ai_generated"))

    assert model.training is False
