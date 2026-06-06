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

CLASS_NAMES = ("real", "ai_generated")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int


def discover_cifake_samples(split_dir: Path) -> list[Sample]:
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
            if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
                samples.append(Sample(path=path, label=label))

    if not samples:
        msg = f"No images found in {split_dir}"
        raise FileNotFoundError(msg)

    return samples


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
