"""Build PyTorch DataLoaders for project datasets."""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import DataLoader

from kkp.data.dataset import ImageBinaryDataset, Sample, discover_cifake_samples
from kkp.data.transforms import get_transforms


def _split_train_val(
    samples: list[Sample],
    val_fraction: float,
    seed: int,
) -> tuple[list[Sample], list[Sample]]:
    if not 0 < val_fraction < 1:
        msg = f"val_fraction must be between 0 and 1, got {val_fraction}"
        raise ValueError(msg)

    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)

    val_size = int(len(shuffled) * val_fraction)
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    return train_samples, val_samples


def build_cifake_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_samples, val_samples = _split_train_val(
        discover_cifake_samples(train_dir),
        val_fraction=val_fraction,
        seed=seed,
    )
    test_samples = discover_cifake_samples(test_dir)

    loaders = {
        "train": _make_loader(
            train_samples,
            batch_size=batch_size,
            image_size=image_size,
            train=True,
            num_workers=num_workers,
            shuffle=True,
        ),
        "val": _make_loader(
            val_samples,
            batch_size=batch_size,
            image_size=image_size,
            train=False,
            num_workers=num_workers,
            shuffle=False,
        ),
        "test": _make_loader(
            test_samples,
            batch_size=batch_size,
            image_size=image_size,
            train=False,
            num_workers=num_workers,
            shuffle=False,
        ),
    }
    return loaders


def _make_loader(
    samples: list[Sample],
    *,
    batch_size: int,
    image_size: int,
    train: bool,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ImageBinaryDataset(
        samples=samples,
        transform=get_transforms(image_size, train=train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
    )
