from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from kkp.data.dataset import CLASS_NAMES, ImageBinaryDataset, discover_cifake_samples
from kkp.data.loaders import build_cifake_dataloaders
from kkp.data.transforms import get_transforms


def _create_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


@pytest.fixture
def cifake_root(tmp_path: Path) -> Path:
    for split in ("train", "test"):
        _create_image(tmp_path / split / "REAL" / f"{split}_real.jpg", (255, 0, 0))
        _create_image(tmp_path / split / "FAKE" / f"{split}_fake.jpg", (0, 0, 255))
    return tmp_path


def test_discover_cifake_samples(cifake_root: Path) -> None:
    samples = discover_cifake_samples(cifake_root / "train")
    assert len(samples) == 2
    labels = {sample.label for sample in samples}
    assert labels == {0, 1}


def test_image_binary_dataset(cifake_root: Path) -> None:
    samples = discover_cifake_samples(cifake_root / "train")
    dataset = ImageBinaryDataset(
        samples=samples,
        transform=get_transforms(32, train=False),
    )
    image, label = dataset[0]
    assert label in (0, 1)
    assert tuple(image.shape) == (3, 32, 32)


def test_build_cifake_dataloaders(cifake_root: Path) -> None:
    loaders = build_cifake_dataloaders(
        cifake_root,
        batch_size=1,
        image_size=32,
        seed=42,
        val_fraction=0.5,
    )

    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) == 1
    assert len(loaders["val"].dataset) == 1
    assert len(loaders["test"].dataset) == 2

    images, labels = next(iter(loaders["train"]))
    assert images.shape[0] == 1
    assert labels.shape[0] == 1


def test_class_names() -> None:
    assert CLASS_NAMES == ("real", "ai_generated")
