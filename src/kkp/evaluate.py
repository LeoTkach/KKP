"""Точка входа для оценки модели."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from dotenv import load_dotenv

from kkp.config import load_config
from kkp.data import build_cifake_dataloaders
from kkp.training import load_checkpoint, run_epoch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KKP model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ai_generated.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pth)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    config = load_config(args.config)
    training = config["training"]
    project = config["project"]
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_path = args.checkpoint or output_dir / "checkpoints" / "best.pth"

    if not checkpoint_path.is_file():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    device = torch.device(training["device"])
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, checkpoint["epoch"])

    loaders = build_cifake_dataloaders(
        Path(config["paths"]["data_dir"]),
        batch_size=training["batch_size"],
        image_size=training["image_size"],
        seed=project["seed"],
        val_fraction=config["data"]["val_split"],
    )

    criterion = nn.CrossEntropyLoss()
    test_metrics = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        desc="test",
    )

    logger.info(
        "test loss %.4f | accuracy %.4f",
        test_metrics.loss,
        test_metrics.accuracy,
    )


if __name__ == "__main__":
    main()
