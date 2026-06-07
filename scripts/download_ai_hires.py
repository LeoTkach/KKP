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

Optional Hugging Face (Parveshiiii/AI-vs-Real, streams without loading full archive):
    pip install -e ".[data]" && export HF_TOKEN=...
    python scripts/download_ai_hires.py --profile hf
"""

from __future__ import annotations

import argparse
import os
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
HF_DEFAULT_DATASET = "Parveshiiii/AI-vs-Real"
HF_LABEL_TO_DIR = {0: "FAKE", 1: "REAL"}


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
        choices=("faces", "kaggle", "hf"),
        default="faces",
        help="Dataset source profile",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset slug (kaggle: xhlulu/140k-real-and-fake-faces; hf: Parveshiiii/AI-vs-Real)",
    )
    parser.add_argument("--train-per-class", type=int, default=800)
    parser.add_argument("--val-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=100)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE, help="Target square side")
    parser.add_argument("--min-side", type=int, default=128, help="Minimum image side")
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
    dataset_slug = _resolve_dataset_slug(args)
    staging = args.output / "_kaggle_staging"
    if staging.exists():
        shutil.rmtree(staging)
    _run_kaggle_download(dataset_slug, staging)

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


def _resolve_dataset_slug(args: argparse.Namespace) -> str:
    if args.dataset is not None:
        return args.dataset
    if args.profile == "hf":
        return HF_DEFAULT_DATASET
    return "xhlulu/140k-real-and-fake-faces"


def _load_hf_token() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        msg = (
            "HF_TOKEN not set. Export HF_TOKEN or HUGGING_FACE_HUB_TOKEN "
            "(optionally via .env with python-dotenv)."
        )
        raise SystemExit(msg)
    return token


def _split_for_class_index(index: int, args: argparse.Namespace) -> str:
    if index < args.train_per_class:
        return "train"
    if index < args.train_per_class + args.val_per_class:
        return "val"
    return "test"


def _scan_hf_progress(output: Path) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    class_totals = {label: 0 for label in HF_LABEL_TO_DIR.values()}
    split_offsets: dict[tuple[str, str], int] = {}
    for split in ("train", "val", "test"):
        for label_dir in HF_LABEL_TO_DIR.values():
            folder = output / split / label_dir
            count = len(list(folder.glob("*.jpg"))) if folder.is_dir() else 0
            split_offsets[(split, label_dir)] = count
            class_totals[label_dir] += count
    return class_totals, split_offsets


def _download_hf_profile(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        msg = 'datasets package required. Install: pip install -e ".[data]"'
        raise SystemExit(msg) from exc

    dataset_slug = _resolve_dataset_slug(args)
    token = _load_hf_token()
    per_class = args.train_per_class + args.val_per_class + args.test_per_class
    class_totals, split_offsets = _scan_hf_progress(args.output)

    print(f"Target: {per_class} images per class (REAL + FAKE)")
    print(f"Already on disk: REAL={class_totals['REAL']}, FAKE={class_totals['FAKE']}")
    if class_totals["REAL"] >= per_class and class_totals["FAKE"] < per_class:
        print("REAL is full — streaming will skip real photos and collect FAKE only (slower).")
    print("Connecting to Hugging Face… first progress may take 2–10 minutes.")

    stream = load_dataset(dataset_slug, split="train", streaming=True, token=token)
    progress = tqdm(stream, desc=f"hf/{dataset_slug}")

    for example in progress:
        label_dir = HF_LABEL_TO_DIR.get(example["binary_label"])
        if label_dir is None:
            continue
        if class_totals[label_dir] >= per_class:
            if all(count >= per_class for count in class_totals.values()):
                break
            continue

        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image["bytes"] if isinstance(image, dict) else image))
        rgb = image.convert("RGB")
        if min(rgb.size) < args.min_side:
            continue

        class_index = class_totals[label_dir]
        split = _split_for_class_index(class_index, args)
        offset = split_offsets.get((split, label_dir), 0)
        filename = f"{split.lower()}_{label_dir.lower()}_{offset:04d}.jpg"
        path = args.output / split / label_dir / filename
        if not path.is_file():
            _save_square(rgb, path, args.size)

        split_offsets[(split, label_dir)] = offset + 1
        class_totals[label_dir] += 1

        if all(count >= per_class for count in class_totals.values()):
            break

    for label_dir, count in class_totals.items():
        if count < per_class:
            print(
                f"Warning: only collected {count}/{per_class} {label_dir} images "
                f"from {dataset_slug}",
                file=sys.stderr,
            )

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
    elif args.profile == "kaggle":
        _download_kaggle_profile(args)
    else:
        _download_hf_profile(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
