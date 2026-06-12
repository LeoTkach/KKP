"""Compare trained models: metrics tables and report plots."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv

from kkp.config import load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIGS = (
    Path("configs/ai_generated.yaml"),
    Path("configs/ai_generated_efficientnet.yaml"),
)

MODEL_COLORS = {
    "RESNET18": "#2E86AB",
    "EFFICIENTNET_B0": "#A23B72",
}
METRIC_COLORS = ["#1B4965", "#2E86AB", "#5FA8D3", "#A23B72", "#F18F01"]


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


def _apply_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFBFC",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#1E293B",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#E2E8F0",
            "font.family": "DejaVu Sans",
        }
    )


def _model_color(label: str) -> str:
    return MODEL_COLORS.get(label.upper(), "#475569")


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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle("Криві навчання моделей", fontsize=14, fontweight="bold", y=1.02)

    for run in runs:
        color = _model_color(run["label"])
        history = run["history"]
        epochs = [row["epoch"] for row in history]
        train_loss = [row["train_loss"] for row in history]
        val_loss = [row["val_loss"] for row in history]
        train_acc = [row["train_accuracy"] for row in history]
        val_acc = [row["val_accuracy"] for row in history]

        axes[0].plot(epochs, train_loss, color=color, linewidth=2.2, label=f"{run['label']} train")
        axes[0].plot(
            epochs,
            val_loss,
            color=color,
            linewidth=2.2,
            linestyle="--",
            alpha=0.85,
            label=f"{run['label']} val",
        )
        axes[1].plot(epochs, train_acc, color=color, linewidth=2.2, label=f"{run['label']} train")
        axes[1].plot(
            epochs,
            val_acc,
            color=color,
            linewidth=2.2,
            linestyle="--",
            alpha=0.85,
            label=f"{run['label']} val",
        )

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Епоха")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.02)

    for ax in axes:
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "learning_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_metric_bars(runs: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [run["label"] for run in runs]
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    metric_labels = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
    values = {name: [run["metrics"][name] for run in runs] for name in metric_names}

    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(labels))
    bar_width = 0.14
    for index, (name, label) in enumerate(zip(metric_names, metric_labels, strict=True)):
        offset = (index - len(metric_names) / 2) * bar_width + bar_width / 2
        bars = ax.bar(
            x_positions + offset,
            values[name],
            width=bar_width,
            label=label,
            color=METRIC_COLORS[index],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0.85, 1.06)
    ax.set_ylabel("Score")
    ax.set_title("Порівняння метрик на test set", fontweight="bold", pad=12)
    ax.axhline(0.95, color="#94A3B8", linestyle=":", linewidth=1.2, label="95 % baseline")
    ax.legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_confusion_matrices(runs: list[dict[str, Any]], output_dir: Path) -> None:
    count = len(runs)
    fig, axes = plt.subplots(1, count, figsize=(5.2 * count, 4.5))
    if count == 1:
        axes = [axes]

    fig.suptitle("Матриці помилок (test set)", fontsize=14, fontweight="bold", y=1.02)

    for ax, run in zip(axes, runs, strict=True):
        matrix = np.array(run["metrics"]["confusion_matrix"])
        class_names = run["metrics"]["class_names"]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar=False,
            linewidths=1.5,
            linecolor="white",
            annot_kws={"fontsize": 12, "fontweight": "bold"},
        )
        acc = run["metrics"]["accuracy"]
        ax.set_title(f"{run['label']}\naccuracy = {acc:.1%}", fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.tight_layout()
    path = output_dir / "confusion_matrices.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def plot_roc_curves(runs: list[dict[str, Any]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for run in runs:
        metrics = run["metrics"]
        color = _model_color(run["label"])
        ax.plot(
            metrics["fpr"],
            metrics["tpr"],
            color=color,
            linewidth=2.5,
            label=f"{run['label']} (AUC={metrics['roc_auc']:.4f})",
        )
        ax.fill_between(metrics["fpr"], metrics["tpr"], alpha=0.08, color=color)

    ax.plot([0, 1], [0, 1], "--", color="#94A3B8", linewidth=1.2, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-криві моделей", fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = output_dir / "roc_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
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
    _apply_plot_style()
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
