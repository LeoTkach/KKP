from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from kkp.data.dataset import CLASS_NAMES, ImageBinaryDataset, discover_cifake_samples
from kkp.data.loaders import build_cifake_dataloaders, build_split_dataloaders
from kkp.data.transforms import get_transforms


def _create_image(
    path: Path,
    color: tuple[int, int, int],
    size: tuple[int, int] = (32, 32),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def cifake_root(tmp_path: Path) -> Path:
    for split in ("train", "test"):
        _create_image(tmp_path / split / "REAL" / f"{split}_real.jpg", (255, 0, 0))
        _create_image(tmp_path / split / "FAKE" / f"{split}_fake.jpg", (0, 0, 255))
    return tmp_path


@pytest.fixture
def hires_root(tmp_path: Path) -> Path:
    for split in ("train", "val", "test"):
        _create_image(
            tmp_path / split / "REAL" / f"{split}_real.jpg",
            (255, 0, 0),
            size=(512, 512),
        )
        _create_image(
            tmp_path / split / "FAKE" / f"{split}_fake.jpg",
            (0, 0, 255),
            size=(512, 512),
        )
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


def test_discover_real_world_samples(tmp_path: Path) -> None:
    _create_image(tmp_path / "real" / "photo.jpg", (255, 0, 0))
    _create_image(tmp_path / "ai_generated" / "fake.jpg", (0, 0, 255))

    from kkp.data.dataset import discover_real_world_samples

    samples = discover_real_world_samples(tmp_path)
    assert len(samples) == 2
    assert {sample.label for sample in samples} == {0, 1}


def test_strong_augment_transform_shape() -> None:
    from torch import Tensor

    transform = get_transforms(224, train=True, strong_augment=True)
    image = Image.new("RGB", (640, 480), (128, 64, 32))
    tensor = transform(image)
    assert isinstance(tensor, Tensor)
    assert tuple(tensor.shape) == (3, 224, 224)


def test_build_split_dataloaders(hires_root: Path) -> None:
    loaders = build_split_dataloaders(
        hires_root,
        batch_size=1,
        image_size=224,
        min_side=256,
    )

    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) == 2
    assert len(loaders["test"].dataset) == 2


def test_min_side_filter_excludes_small_images(cifake_root: Path) -> None:
    _create_image(cifake_root / "train" / "REAL" / "large.jpg", (1, 2, 3), size=(512, 512))

    samples = discover_cifake_samples(cifake_root / "train", min_side=256)
    assert len(samples) == 1
    assert samples[0].path.name == "large.jpg"


def test_build_loaders_from_config_hires(hires_root: Path) -> None:
    from kkp.data.factory import build_loaders_from_config

    config = {
        "project": {"seed": 42},
        "paths": {"data_dir": str(hires_root)},
        "training": {"batch_size": 1, "image_size": 224},
        "data": {"val_split": 0.1, "strong_augment": False, "min_side": 256},
        "datasets": {"primary": "split_folder"},
    }
    loaders = build_loaders_from_config(config)
    assert len(loaders["val"].dataset) == 2


def test_supplement_merged_into_train(cifake_root: Path, tmp_path: Path) -> None:
    supplement = tmp_path / "supplement" / "train"
    _create_image(supplement / "REAL" / "extra_real.jpg", (0, 255, 0))
    _create_image(supplement / "FAKE" / "extra_fake.jpg", (0, 0, 128))

    loaders = build_cifake_dataloaders(
        cifake_root,
        batch_size=1,
        image_size=32,
        seed=42,
        val_fraction=0.5,
        supplement_dir=tmp_path / "supplement",
        supplement_max_per_class=10,
    )

    assert len(loaders["train"].dataset) == 3
