"""Training metrics and evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "epoch",
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return EpochMetrics(
        loss=total_loss / total,
        accuracy=correct / total,
    )
