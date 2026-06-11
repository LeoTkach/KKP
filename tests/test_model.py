from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from kkp.data.dataset import ImageBinaryDataset, Sample
from kkp.data.loaders import build_cifake_dataloaders
from kkp.data.transforms import get_transforms
from kkp.models.factory import create_model
from kkp.training.metrics import EpochMetrics, run_epoch
from kkp.training.trainer import load_checkpoint, save_checkpoint, train_model


def test_create_resnet18() -> None:
    model = create_model("resnet18", num_classes=2, pretrained=False)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 2)


def test_create_efficientnet_b0() -> None:
    model = create_model("efficientnet_b0", num_classes=2, pretrained=False)
    output = model(torch.randn(2, 3, 224, 224))
    assert output.shape == (2, 2)


def test_run_epoch_on_tiny_loader() -> None:
    samples = [
        Sample(path=Path("dummy0"), label=0),
        Sample(path=Path("dummy1"), label=1),
    ]
    dataset = _DummyDataset(samples)
    loader = DataLoader(dataset, batch_size=2)

    model = create_model("resnet18", num_classes=2, pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    metrics = run_epoch(
        model,
        loader,
        criterion,
        torch.device("cpu"),
        optimizer=optimizer,
        desc="test",
    )
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.loss >= 0.0


def test_save_and_load_checkpoint(tmp_path: Path) -> None:
    model = create_model("resnet18", num_classes=2, pretrained=False)
    config = {
        "model": {"name": "resnet18", "num_classes": 2},
    }
    path = tmp_path / "model.pth"
    save_checkpoint(
        path,
        model,
        config,
        epoch=1,
        metrics=EpochMetrics(loss=0.5, accuracy=0.8),
    )

    loaded, meta = load_checkpoint(path, torch.device("cpu"))
    assert meta["epoch"] == 1
    assert meta["val_accuracy"] == 0.8
    assert loaded.training is False


class _DummyDataset(ImageBinaryDataset):
    def __init__(self, samples: list[Sample]) -> None:
        super().__init__(samples, transform=get_transforms(32, train=False))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        color = (255, 0, 0) if sample.label == 0 else (0, 0, 255)
        image = Image.new("RGB", (32, 32), color)
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label


@pytest.fixture
def tiny_cifake_root(tmp_path: Path) -> Path:
    for split in ("train", "test"):
        for label, color in (("REAL", (255, 0, 0)), ("FAKE", (0, 0, 255))):
            for index in range(2):
                path = tmp_path / split / label / f"{split}_{label}_{index}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), color).save(path)
    return tmp_path


def test_train_model_smoke(tiny_cifake_root: Path, tmp_path: Path) -> None:
    config = {
        "project": {"seed": 42, "name": "test"},
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
            "device": "cpu",
            "image_size": 32,
        },
        "model": {"name": "resnet18", "num_classes": 2, "pretrained": False},
    }
    loaders = build_cifake_dataloaders(
        tiny_cifake_root,
        batch_size=2,
        image_size=32,
        seed=42,
        val_fraction=0.25,
    )
    checkpoint = train_model(config, loaders, tmp_path)
    assert checkpoint.is_file()
