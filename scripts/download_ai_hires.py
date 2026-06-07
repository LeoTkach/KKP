#!/usr/bin/env python3
"""Download a balanced hi-res AI vs real dataset (same native resolution per profile).

Profiles
--------
faces (default)
    REAL: pravatar.cc portraits (512×512)
    FAKE: thispersondoesnotexist.com StyleGAN faces (resized to 512×512)

Both classes are face portraits at the same resolution — suitable for in-domain training
and demo without mixing 32 px CIFAKE tiles with full-size photos.

Optional Kaggle (large scale, recommended for the report):
    pip install kaggle && place token at ~/.kaggle/kaggle.json
    python scripts/download_ai_hires.py --profile kaggle --dataset xhlulu/140k-real-and-fake-faces
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm

IMAGE_SIZE = 512
REAL_URL = "https://i.pravatar.cc/{size}?u=kkp-real-{index}"
FAKE_URL = "https://thispersondoesnotexist.com/"
USER_AGENT = "KKP-course-project/1.0 (educational dataset builder)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download balanced hi-res AI vs real dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ai_hires"),
        help="Output root (train/val/test/REAL|FAKE layout)",
    )
    parser.add_argument(
        "--profile",
        choices=("faces", "kaggle"),
        default="faces",
        help="Dataset source profile",
    )
    parser.add_argument(
        "--dataset",
        default="xhlulu/140k-real-and-fake-faces",
        help="Kaggle dataset slug (profile=kaggle)",
    )
    parser.add_argument("--train-per-class", type=int, default=800)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Target square side")
    parser.add_argument("--min-side", type=int, default=256, help="Minimum image side")
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between requests")
    return parser.parse_args()


def _fetch_image(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    return Image.open(BytesIO(payload)).convert("RGB")


def _save_square(image: Image.Image, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(path, format="JPEG", quality=92)


def _download_faces_profile(args: argparse.Namespace) -> None:
    splits = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    index = 0

    for split, count in splits.items():
        for label_name, label_dir in (("REAL", "REAL"), ("FAKE", "FAKE")):
            dest_dir = args.output / split / label_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            desc = f"{split}/{label_name}"

            for offset in tqdm(range(count), desc=desc):
                index += 1
                filename = f"{split.lower()}_{label_name.lower()}_{offset:04d}.jpg"
                path = dest_dir / filename
                if path.is_file():
                    continue

                for attempt in range(5):
                    try:
                        if label_name == "REAL":
                            url = REAL_URL.format(size=args.size, index=index)
                            image = _fetch_image(url)
                        else:
                            image = _fetch_image(FAKE_URL)

                        if min(image.size) < args.min_side:
                            msg = f"Image too small: {image.size}"
                            raise ValueError(msg)

                        _save_square(image, path, args.size)
                        break
                    except (HTTPError, URLError, OSError, ValueError) as exc:
                        if attempt == 4:
                            raise RuntimeError(f"Failed to download {path.name}") from exc
                        time.sleep(args.sleep * (attempt + 1))
                time.sleep(args.sleep)

    _print_summary(args.output, args.min_side)


def _run_kaggle_download(dataset: str, target: Path) -> Path:
    if shutil.which("kaggle") is None:
        msg = (
            "Kaggle CLI not found. Install: pip install kaggle\n"
            "Then place API token at ~/.kaggle/kaggle.json"
        )
        raise SystemExit(msg)

    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target), "--unzip"],
        check=True,
    )
    return target


def _find_class_dirs(root: Path) -> tuple[Path, Path]:
    """Return (real_dir, fake_dir) under a downloaded Kaggle tree."""
    candidates_real = list(root.rglob("real"))
    candidates_fake = list(root.rglob("fake"))
    real_dirs = [path for path in candidates_real if path.is_dir()]
    fake_dirs = [path for path in candidates_fake if path.is_dir()]

    if not real_dirs or not fake_dirs:
        msg = f"Could not find real/ and fake/ folders under {root}"
        raise FileNotFoundError(msg)

    return real_dirs[0], fake_dirs[0]


def _copy_split(
    sources: list[Path],
    dest: Path,
    *,
    size: int,
    min_side: int,
) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for source in sources:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            if min(rgb.size) < min_side:
                continue
            out = dest / f"{source.stem}.jpg"
            _save_square(rgb, out, size)
            saved += 1
    return saved


def _download_kaggle_profile(args: argparse.Namespace) -> None:
    staging = args.output / "_kaggle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    _run_kaggle_download(args.dataset, staging)

    real_dir, fake_dir = _find_class_dirs(staging)
    real_files = sorted(
        path
        for path in real_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    fake_files = sorted(
        path
        for path in fake_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )

    import random

    rng = random.Random(42)
    rng.shuffle(real_files)
    rng.shuffle(fake_files)

    splits = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    offset = 0
    for split, count in splits.items():
        real_chunk = real_files[offset : offset + count]
        fake_chunk = fake_files[offset : offset + count]
        offset += count
        _copy_split(
            real_chunk,
            args.output / split / "REAL",
            size=args.size,
            min_side=args.min_side,
        )
        _copy_split(
            fake_chunk,
            args.output / split / "FAKE",
            size=args.size,
            min_side=args.min_side,
        )

    shutil.rmtree(staging)
    _print_summary(args.output, args.min_side)


def _print_summary(output: Path, min_side: int) -> None:
    print(f"\nDataset ready at {output.resolve()}")
    for split in ("train", "val", "test"):
        for label in ("REAL", "FAKE"):
            folder = output / split / label
            count = len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0
            print(f"  {split}/{label}: {count}")
    print(f"  min_side filter: {min_side}px")
    print("\nTrain: python -m kkp.train --config configs/ai_generated.yaml")


def main() -> None:
    args = parse_args()
    if args.profile == "faces":
        _download_faces_profile(args)
    else:
        _download_kaggle_profile(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
