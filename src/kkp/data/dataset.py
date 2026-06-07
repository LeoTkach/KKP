"""Image dataset for binary classification (CIFAKE layout)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# CIFAKE folder names -> class index
CIFAKE_LABEL_DIRS = {
    "REAL": 0,
    "FAKE": 1,
}

# Flat layout: data/real_world/{real,ai_generated}/
REAL_WORLD_LABEL_DIRS = {
    "real": 0,
    "ai_generated": 1,
}

CLASS_NAMES = ("real", "ai_generated")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def _image_min_side(path: Path) -> int:
    with Image.open(path) as image:
        width, height = image.size
    return min(width, height)


def discover_cifake_samples(
    split_dir: Path,
    *,
    min_side: int = 0,
) -> list[Sample]:
    if not split_dir.is_dir():
        msg = f"Split directory not found: {split_dir}"
        raise FileNotFoundError(msg)

    samples: list[Sample] = []
    for label_dir, label in CIFAKE_LABEL_DIRS.items():
        class_dir = split_dir / label_dir
        if not class_dir.is_dir():
            msg = f"Class directory not found: {class_dir}"
            raise FileNotFoundError(msg)

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            if min_side > 0 and _image_min_side(path) < min_side:
                continue
            samples.append(Sample(path=path, label=label))

    if not samples:
        msg = f"No images found in {split_dir}"
        if min_side > 0:
            msg = f"{msg} (min_side={min_side})"
        raise FileNotFoundError(msg)

    return samples


def discover_labeled_folder_samples(
    root: Path,
    label_dirs: dict[str, int],
    *,
    required: bool = True,
) -> list[Sample]:
    if not root.is_dir():
        if required:
            msg = f"Directory not found: {root}"
            raise FileNotFoundError(msg)
        return []

    samples: list[Sample] = []
    for label_dir, label in label_dirs.items():
        class_dir = root / label_dir
        if not class_dir.is_dir():
            if required:
                msg = f"Class directory not found: {class_dir}"
                raise FileNotFoundError(msg)
            continue

        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                samples.append(Sample(path=path, label=label))

    if required and not samples:
        msg = f"No images found in {root}"
        raise FileNotFoundError(msg)

    return samples


def discover_real_world_samples(root: Path, *, required: bool = True) -> list[Sample]:
    return discover_labeled_folder_samples(
        root,
        REAL_WORLD_LABEL_DIRS,
        required=required,
    )


class ImageBinaryDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        transform: Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label
