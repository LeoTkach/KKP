"""Training loop."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from kkp.models.factory import create_model
from kkp.training.metrics import EpochMetrics, run_epoch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    config: dict[str, Any],
    loaders: dict[str, DataLoader],
    output_dir: Path,
) -> Path:
    training = config["training"]
    model_cfg = config["model"]
    project = config["project"]

    set_seed(project["seed"])

    device = torch.device(training["device"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(
        model_cfg["name"],
        model_cfg["num_classes"],
        pretrained=model_cfg.get("pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])

    best_val_acc = 0.0
    best_path = checkpoint_dir / "best.pth"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, training["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            desc=f"train {epoch}/{training['epochs']}",
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            desc=f"val {epoch}/{training['epochs']}",
        )

        logger.info(
            "epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch,
            training["epochs"],
            train_metrics.loss,
            train_metrics.accuracy,
            val_metrics.loss,
            val_metrics.accuracy,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
            },
        )

        if val_metrics.accuracy > best_val_acc:
            best_val_acc = val_metrics.accuracy
            save_checkpoint(best_path, model, config, epoch, val_metrics)

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    logger.info("Best val accuracy: %.4f", best_val_acc)
    logger.info("Checkpoint saved to %s", best_path)
    return best_path


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: dict[str, Any],
    epoch: int,
    metrics: EpochMetrics,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "model_name": config["model"]["name"],
            "num_classes": config["model"]["num_classes"],
            "val_loss": metrics.loss,
            "val_accuracy": metrics.accuracy,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = create_model(
        checkpoint["model_name"],
        checkpoint["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
