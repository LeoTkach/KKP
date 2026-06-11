from __future__ import annotations

import numpy as np

from kkp.training.metrics import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 1, 1], dtype=np.int64)
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=np.float64,
    )

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
        class_names=("real", "ai_generated"),
        model="resnet18",
        split="test",
        checkpoint="best.pth",
        epoch=1,
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]
