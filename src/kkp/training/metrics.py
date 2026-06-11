"""Training metrics and evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class ClassificationMetrics:
    model: str
    split: str
    checkpoint: str
    epoch: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    class_names: tuple[str, ...]
    fpr: list[float]
    tpr: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    desc: str = "predict",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[int] = []
    preds_all: list[int] = []
    probs_all: list[list[float]] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            preds_all.extend(outputs.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.tolist())
            probs_all.extend(probabilities.cpu().tolist())

    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(preds_all, dtype=np.int64),
        np.asarray(probs_all, dtype=np.float64),
    )


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    class_names: tuple[str, ...],
    model: str,
    split: str,
    checkpoint: str,
    epoch: int,
) -> ClassificationMetrics:
    positive_label = 1
    binary_kwargs = {"average": "binary", "pos_label": positive_label, "zero_division": 0}
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, positive_label], pos_label=positive_label)
    return ClassificationMetrics(
        model=model,
        split=split,
        checkpoint=checkpoint,
        epoch=epoch,
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, **binary_kwargs)),
        recall=float(recall_score(y_true, y_pred, **binary_kwargs)),
        f1=float(f1_score(y_true, y_pred, **binary_kwargs)),
        roc_auc=float(roc_auc_score(y_true, y_prob[:, positive_label])),
        confusion_matrix=confusion_matrix(y_true, y_pred).astype(int).tolist(),
        class_names=class_names,
        fpr=fpr.astype(float).tolist(),
        tpr=tpr.astype(float).tolist(),
    )
