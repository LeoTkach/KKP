"""Evaluate model and export classification metrics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from dotenv import load_dotenv

from kkp.config import load_config
from kkp.data import build_folder_test_loader, build_loaders_from_config
from kkp.data.dataset import CLASS_NAMES
from kkp.training import load_checkpoint, run_epoch
from kkp.training.metrics import (
    ClassificationMetrics,
    collect_predictions,
    compute_classification_metrics,
)

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
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Evaluate on folder with real/ and ai_generated/ subdirs (real-world set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for metrics JSON (default: <output_dir>/metrics.json)",
    )
    return parser.parse_args()


def evaluate_model(
    config: dict,
    checkpoint_path: Path,
    *,
    data_dir: Path | None = None,
) -> tuple[ClassificationMetrics, object]:
    training = config["training"]
    device = torch.device(training["device"])
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    criterion = nn.CrossEntropyLoss()
    model_name = config["model"]["name"]
    class_names = tuple(config["data"]["classes"])

    if data_dir is not None:
        loader = build_folder_test_loader(
            data_dir,
            batch_size=training["batch_size"],
            image_size=training["image_size"],
        )
        split = "real-world"
        logger.info("Evaluating real-world set: %s (%d samples)", data_dir, len(loader.dataset))
    else:
        loader = build_loaders_from_config(config)["test"]
        split = "test"
        logger.info("Evaluating test set (%d samples)", len(loader.dataset))

    loss_metrics = run_epoch(model, loader, criterion, device, desc=split)
    y_true, y_pred, y_prob = collect_predictions(model, loader, device, desc=f"{split}-pred")
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
        class_names=class_names or CLASS_NAMES,
        model=model_name,
        split=split,
        checkpoint=str(checkpoint_path),
        epoch=int(checkpoint["epoch"]),
    )
    logger.info(
        "%s | loss %.4f | accuracy %.4f | precision %.4f | recall %.4f | f1 %.4f | roc_auc %.4f",
        split,
        loss_metrics.loss,
        metrics.accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.roc_auc,
    )
    return metrics, loss_metrics


def save_metrics(metrics: ClassificationMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding="utf-8")
    logger.info("Metrics saved to %s", path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    config = load_config(args.config)
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_path = args.checkpoint or output_dir / "checkpoints" / "best.pth"

    if not checkpoint_path.is_file():
        msg = f"Checkpoint not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    logger.info("Loaded checkpoint from %s", checkpoint_path)
    metrics, _ = evaluate_model(config, checkpoint_path, data_dir=args.data_dir)

    metrics_path = args.output or output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)


if __name__ == "__main__":
    main()
