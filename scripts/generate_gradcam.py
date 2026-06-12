#!/usr/bin/env python3
"""Generate Grad-CAM figure for KKP report (misclassification analysis)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kkp.config import load_config  # noqa: E402
from kkp.data.transforms import get_transforms  # noqa: E402
from kkp.gradcam import explain_image  # noqa: E402
from kkp.training.trainer import load_checkpoint  # noqa: E402

logger = logging.getLogger(__name__)

# file, test subdir, model config, checkpoint rel path, target class index, caption
GRADCAM_CASES: list[tuple[str, str, str, str, int, str]] = [
    (
        "test_real_0049.jpg",
        "REAL",
        "configs/ai_generated.yaml",
        "outputs/ai_generated/checkpoints/best.pth",
        1,
        "ResNet18 → AI (98,4 %), FP",
    ),
    (
        "test_real_0004.jpg",
        "REAL",
        "configs/ai_generated_efficientnet.yaml",
        "outputs/ai_generated_efficientnet/checkpoints/best.pth",
        1,
        "EffNet → AI (99,3 %), FP",
    ),
    (
        "test_fake_0043.jpg",
        "FAKE",
        "configs/ai_generated.yaml",
        "outputs/ai_generated/checkpoints/best.pth",
        0,
        "ResNet18 → REAL (58,5 %), FN",
    ),
]


def generate_gradcam_figure(
    output: Path,
    *,
    data_root: Path,
    device: torch.device | None = None,
) -> Path | None:
    device = device or torch.device("cpu")
    rows = len(GRADCAM_CASES)
    fig, axes = plt.subplots(rows, 2, figsize=(7.2, 3.4 * rows))
    if rows == 1:
        axes = [axes]

    any_rendered = False
    for row, case in enumerate(GRADCAM_CASES):
        fname, subdir, config_rel, ckpt_rel, target_class, caption = case
        image_path = data_root / "ai_hires" / "test" / subdir / fname
        config_path = ROOT / config_rel
        ckpt_path = ROOT / ckpt_rel
        ax_orig, ax_cam = axes[row]

        for ax in (ax_orig, ax_cam):
            ax.axis("off")

        if not image_path.is_file():
            ax_orig.text(0.5, 0.5, f"{fname}\nвідсутній", ha="center", va="center", fontsize=9)
            continue
        if not ckpt_path.is_file():
            ax_orig.text(0.5, 0.5, f"checkpoint\n{ckpt_rel}", ha="center", va="center", fontsize=8)
            continue

        config = load_config(config_path)
        model, _ = load_checkpoint(ckpt_path, device)
        transform = get_transforms(config["training"]["image_size"], train=False)
        from PIL import Image

        image = Image.open(image_path)
        original, overlay = explain_image(
            model,
            config["model"]["name"],
            image,
            transform,
            device,
            target_class=target_class,
            image_size=config["training"]["image_size"],
        )
        ax_orig.imshow(original)
        ax_orig.set_title(f"{fname}\nоригінал · {caption}", fontsize=8.5, pad=4)
        ax_cam.imshow(overlay)
        ax_cam.set_title("Grad-CAM (клас передбачення)", fontsize=8.5, pad=4)
        any_rendered = True

    if not any_rendered:
        plt.close(fig)
        logger.warning("Grad-CAM skipped: missing images or checkpoints")
        return None

    fig.suptitle(
        "Grad-CAM: області зображення, що вплинули на передбачення",
        fontsize=10,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM report figure")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "report_assets" / "gradcam_examples.png",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    device = torch.device(args.device)
    generate_gradcam_figure(args.output, data_root=args.data_root, device=device)


if __name__ == "__main__":
    main()
