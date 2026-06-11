"""Compare trained models: metrics tables and report plots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

from kkp.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = (
    Path("configs/ai_generated.yaml"),
    Path("configs/ai_generated_efficientnet.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare KKP models and plot report charts")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        help="Model config (repeatable; default: resnet18 + efficientnet_b0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/comparison"),
        help="Directory for comparison plots",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_label(config: dict[str, Any]) -> str:
    return config["model"]["name"].upper()


def _load_run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    label = _model_label(config)
    history_path = output_dir / "history.json"
    metrics_path = output_dir / "metrics.json"
    if not history_path.is_file():
        msg = f"Missing history.json for {label}: {history_path}"
        raise FileNotFoundError(msg)
    if not metrics_path.is_file():
        msg = (
            f"Missing metrics.json for {label}. Run: python -m kkp.evaluate --config {config_path}"
        )
        raise FileNotFoundError(msg)
    return {
        "label": label,
        "config_path": config_path,
        "output_dir": output_dir,
        "history": _load_json(history_path),
        "metrics": _load_json(metrics_path),
    }


def plot_learning_curves(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for run in runs:
        history = run["history"]
        epochs = [row["epoch"] for row in history]
        train_loss = [row["train_loss"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        axes[0].plot(epochs, train_loss, label=f"{run['label']} train")
        axes[0].plot(epochs, val_loss, "--", label=f"{run['label']} val")
        axes[1].plot(
            epochs,
            [row["train_accuracy"] for row in history],
            label=f"{run['label']} train",
        )
        axes[1].plot(
            epochs,
            [row["val_accuracy"] for row in history],
            "--",
            label=f"{run['label']} val",
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    path = output_dir / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_metric_bars(runs: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [run["label"] for run in runs]
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    values = {name: [run["metrics"][name] for run in runs] for name in metric_names}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_positions = range(len(labels))
    bar_width = 0.15
    for index, name in enumerate(metric_names):
        offset = (index - len(metric_names) / 2) * bar_width + bar_width / 2
        ax.bar(
            [x + offset for x in x_positions],
            values[name],
            width=bar_width,
            label=name.upper(),
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Test metrics comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_confusion_matrices(runs: list[dict[str, Any]], output_dir: Path) -> None:
    count = len(runs)
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 4))
    if count == 1:
        axes = [axes]

    for ax, run in zip(axes, runs, strict=True):
        matrix = run["metrics"]["confusion_matrix"]
        class_names = run["metrics"]["class_names"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=False,
        )
        ax.set_title(run["label"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    path = output_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_roc_curves(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for run in runs:
        metrics = run["metrics"]
        ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{run['label']} (AUC={metrics['roc_auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "roc_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def write_summary_table(runs: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        metrics = run["metrics"]
        lines.append(
            f"| {run['label']} "
            f"| {metrics['accuracy']:.4f} "
            f"| {metrics['precision']:.4f} "
            f"| {metrics['recall']:.4f} "
            f"| {metrics['f1']:.4f} "
            f"| {metrics['roc_auc']:.4f} |",
        )
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved %s", path)


def compare_models(config_paths: list[Path], output_dir: Path) -> None:
    runs = [_load_run(path) for path in config_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(runs, output_dir)
    plot_metric_bars(runs, output_dir)
    plot_confusion_matrices(runs, output_dir)
    plot_roc_curves(runs, output_dir)
    write_summary_table(runs, output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()
    config_paths = args.configs or list(DEFAULT_CONFIGS)
    compare_models(config_paths, args.output_dir)


if __name__ == "__main__":
    main()
