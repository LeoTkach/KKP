#!/usr/bin/env python3
"""Train and evaluate ablation runs; export docs/artifacts/ablation_runs.json."""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kkp.config import load_config
from kkp.data import build_loaders_from_config
from kkp.evaluate import evaluate_model, save_metrics
from kkp.training import train_model

ARTIFACTS = ROOT / "docs" / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    config: Path
    strong_augment: bool
    epochs: int
    skip_train: bool = False


SPECS = (
    AblationSpec(
        key="baseline",
        label="aug=on, epochs=10 (baseline)",
        config=ROOT / "configs" / "ai_generated_efficientnet.yaml",
        strong_augment=True,
        epochs=10,
        skip_train=True,
    ),
    AblationSpec(
        key="no_augment",
        label="aug=off, epochs=10",
        config=ROOT / "configs" / "ablation_effnet_no_augment.yaml",
        strong_augment=False,
        epochs=10,
    ),
    AblationSpec(
        key="epochs5",
        label="aug=on, epochs=5",
        config=ROOT / "configs" / "ablation_effnet_epochs5.yaml",
        strong_augment=True,
        epochs=5,
    ),
)


def _checkpoint_path(config: dict) -> Path:
    return Path(config["paths"]["output_dir"]) / "checkpoints" / "best.pth"


def _history_val(config: dict) -> list[dict]:
    history_path = Path(config["paths"]["output_dir"]) / "history.json"
    if not history_path.is_file():
        return []
    return json.loads(history_path.read_text(encoding="utf-8"))


def run_spec(spec: AblationSpec, *, force_train: bool) -> dict:
    config = load_config(spec.config)
    output_dir = Path(config["paths"]["output_dir"])
    ckpt = _checkpoint_path(config)

    if not spec.skip_train and (force_train or not ckpt.is_file()):
        logger.info("Training ablation: %s", spec.key)
        loaders = build_loaders_from_config(config)
        train_model(config, loaders, output_dir)
    elif not ckpt.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {spec.key}: {ckpt}")

    metrics, _ = evaluate_model(config, ckpt)
    metrics_path = ARTIFACTS / f"ablation_{spec.key}_test_metrics.json"
    save_metrics(metrics, metrics_path)

    history = _history_val(config)
    best_epoch = metrics.epoch
    val_at_best = next(
        (row["val_accuracy"] for row in history if row["epoch"] == best_epoch),
        None,
    )

    cm = metrics.confusion_matrix
    errors = int(cm[0][1] + cm[1][0])

    return {
        "key": spec.key,
        "label": spec.label,
        "config": str(spec.config.relative_to(ROOT)),
        "strong_augment": spec.strong_augment,
        "epochs": spec.epochs,
        "best_epoch": best_epoch,
        "val_accuracy": val_at_best,
        "test_accuracy": metrics.accuracy,
        "test_f1": metrics.f1,
        "test_roc_auc": metrics.roc_auc,
        "errors": errors,
        "confusion_matrix": cm,
        "metrics_artifact": str(metrics_path.relative_to(ROOT)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Run EfficientNet ablation grid")
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if checkpoint exists",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Subset of ablation keys (default: all)",
    )
    args = parser.parse_args()

    selected = {s.key for s in SPECS} if not args.keys else set(args.keys)
    if args.keys:
        # Keep baseline in JSON when training partial grid
        selected.add("baseline")
    results = [run_spec(s, force_train=args.force_train) for s in SPECS if s.key in selected]

    out = ARTIFACTS / "ablation_runs.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote %s (%d runs)", out, len(results))
    for row in results:
        logger.info(
            "  %s: test acc %.1f%%, F1 %.3f, best epoch %s",
            row["key"],
            row["test_accuracy"] * 100,
            row["test_f1"],
            row["best_epoch"],
        )


if __name__ == "__main__":
    main()
