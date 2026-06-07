"""Build PyTorch DataLoaders for project datasets."""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import DataLoader

from kkp.data.dataset import (
    ImageBinaryDataset,
    Sample,
    discover_cifake_samples,
    discover_real_world_samples,
)
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


def _cap_samples_by_class(
    samples: list[Sample],
    max_per_class: int | None,
    seed: int,
) -> list[Sample]:
    if max_per_class is None or max_per_class <= 0:
        return samples

    by_label: dict[int, list[Sample]] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)

    rng = random.Random(seed)
    capped: list[Sample] = []
    for label in (0, 1):
        pool = by_label[label].copy()
        rng.shuffle(pool)
        capped.extend(pool[:max_per_class])
    return capped


def _merge_supplement_train(
    train_samples: list[Sample],
    supplement_dir: Path | None,
    *,
    seed: int,
    max_per_class: int | None,
) -> list[Sample]:
    if supplement_dir is None:
        return train_samples

    supplement_train = Path(supplement_dir) / "train"
    if not supplement_train.is_dir():
        return train_samples

    extra = discover_cifake_samples(supplement_train)
    extra = _cap_samples_by_class(extra, max_per_class, seed + 1)
    if not extra:
        return train_samples
    return train_samples + extra


def build_split_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    image_size: int,
    num_workers: int = 0,
    min_side: int = 0,
    strong_augment: bool = False,
) -> dict[str, DataLoader]:
    """Load train/val/test splits with REAL/FAKE layout (hi-res datasets)."""
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    for split, train_mode in (("train", True), ("val", False), ("test", False)):
        split_dir = data_dir / split
        samples = discover_cifake_samples(split_dir, min_side=min_side)
        loaders[split] = _make_loader(
            samples,
            batch_size=batch_size,
            image_size=image_size,
            train=train_mode,
            strong_augment=strong_augment and train_mode,
            num_workers=num_workers,
            shuffle=train_mode,
        )

    return loaders


def build_cifake_dataloaders(
    data_dir: Path,
    *,
    batch_size: int,
    image_size: int,
    seed: int,
    val_fraction: float = 0.1,
    num_workers: int = 0,
    extra_train_dir: Path | None = None,
    supplement_dir: Path | None = None,
    supplement_max_per_class: int | None = None,
    strong_augment: bool = False,
    min_side: int = 0,
) -> dict[str, DataLoader]:
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    train_samples, val_samples = _split_train_val(
        discover_cifake_samples(train_dir, min_side=min_side),
        val_fraction=val_fraction,
        seed=seed,
    )
    if extra_train_dir is not None:
        extra = discover_real_world_samples(extra_train_dir, required=False)
        if extra:
            train_samples = train_samples + extra

    train_samples = _merge_supplement_train(
        train_samples,
        supplement_dir,
        seed=seed,
        max_per_class=supplement_max_per_class,
    )

    test_samples = discover_cifake_samples(test_dir, min_side=min_side)

    loaders = {
        "train": _make_loader(
            train_samples,
            batch_size=batch_size,
            image_size=image_size,
            train=True,
            strong_augment=strong_augment,
            num_workers=num_workers,
            shuffle=True,
        ),
        "val": _make_loader(
            val_samples,
            batch_size=batch_size,
            image_size=image_size,
            train=False,
            strong_augment=False,
            num_workers=num_workers,
            shuffle=False,
        ),
        "test": _make_loader(
            test_samples,
            batch_size=batch_size,
            image_size=image_size,
            train=False,
            strong_augment=False,
            num_workers=num_workers,
            shuffle=False,
        ),
    }
    return loaders


def build_folder_test_loader(
    data_dir: Path,
    *,
    batch_size: int,
    image_size: int,
    num_workers: int = 0,
) -> DataLoader:
    samples = discover_real_world_samples(data_dir)
    return _make_loader(
        samples,
        batch_size=batch_size,
        image_size=image_size,
        train=False,
        strong_augment=False,
        num_workers=num_workers,
        shuffle=False,
    )


def _make_loader(
    samples: list[Sample],
    *,
    batch_size: int,
    image_size: int,
    train: bool,
    strong_augment: bool,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ImageBinaryDataset(
        samples=samples,
        transform=get_transforms(image_size, train=train, strong_augment=strong_augment),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=train,
    )
