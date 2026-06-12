#!/usr/bin/env python3
"""Export and verify frozen experiment results without retraining."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

ARTIFACTS = ROOT / "docs" / "artifacts"

RUNS = (
    {
        "key": "resnet18",
        "config": ROOT / "configs" / "ai_generated.yaml",
        "checkpoint": ROOT / "outputs" / "ai_generated" / "checkpoints" / "best.pth",
        "metrics_artifact": ARTIFACTS / "resnet18_test_metrics.json",
    },
    {
        "key": "efficientnet_b0",
        "config": ROOT / "configs" / "ai_generated_efficientnet.yaml",
        "checkpoint": ROOT / "outputs" / "ai_generated_efficientnet" / "checkpoints" / "best.pth",
        "metrics_artifact": ARTIFACTS / "efficientnet_b0_test_metrics.json",
    },
)

REAL_WORLD = {
    "config": ROOT / "configs" / "ai_generated_efficientnet.yaml",
    "checkpoint": ROOT / "outputs" / "ai_generated_efficientnet" / "checkpoints" / "best.pth",
    "data_dir": ROOT / "data" / "real_world",
    "metrics_artifact": ARTIFACTS / "efficientnet_real_world_metrics.json",
}

METRIC_TOLERANCE = 1e-4


@dataclass(frozen=True)
class Misclassification:
    file: str
    true_label: str
    predicted_label: str
    confidence: float
    error_type: str


def _round_metric(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _error_type(true_label: str, predicted_label: str) -> str:
    if true_label == "real" and predicted_label == "ai_generated":
        return "FP"
    if true_label == "ai_generated" and predicted_label == "real":
        return "FN"
    return "other"


def _bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        scores.append(float(np.mean(y_true[idx] == y_pred[idx])))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(low), float(high)


def evaluate_run(run: dict) -> tuple[dict, list[Misclassification], dict]:
    from kkp.config import load_config
    from kkp.data import build_loaders_from_config
    from kkp.training import load_checkpoint
    from kkp.training.metrics import collect_predictions, compute_classification_metrics

    config = load_config(run["config"])
    device = torch.device(config["training"]["device"])
    model, checkpoint = load_checkpoint(run["checkpoint"], device)
    loader = build_loaders_from_config(config)["test"]
    class_names = tuple(config["data"]["classes"])

    y_true, y_pred, y_prob = collect_predictions(model, loader, device, desc=f"{run['key']}-audit")
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        y_prob,
        class_names=class_names,
        model=config["model"]["name"],
        split="test",
        checkpoint=str(run["checkpoint"]),
        epoch=int(checkpoint["epoch"]),
    )
    metrics_dict = metrics.to_dict()

    misclassifications: list[Misclassification] = []
    dataset = loader.dataset
    for index in range(len(dataset)):
        if int(y_pred[index]) == int(y_true[index]):
            continue
        sample = dataset.samples[index]
        true_label = class_names[int(y_true[index])]
        pred_label = class_names[int(y_pred[index])]
        misclassifications.append(
            Misclassification(
                file=sample.path.name,
                true_label=true_label,
                predicted_label=pred_label,
                confidence=float(y_prob[index, int(y_pred[index])]),
                error_type=_error_type(true_label, pred_label),
            )
        )

    mean_acc, ci_low, ci_high = _bootstrap_accuracy_ci(y_true, y_pred)
    bootstrap = {
        "n_samples": int(len(y_true)),
        "n_bootstrap": 2000,
        "accuracy_mean": mean_acc,
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
    }
    return metrics_dict, misclassifications, bootstrap


def evaluate_real_world() -> dict | None:
    from kkp.config import load_config
    from kkp.evaluate import evaluate_model

    if not REAL_WORLD["data_dir"].is_dir():
        return None
    if not REAL_WORLD["checkpoint"].is_file():
        return None
    config = load_config(REAL_WORLD["config"])
    metrics, _ = evaluate_model(config, REAL_WORLD["checkpoint"], data_dir=REAL_WORLD["data_dir"])
    return metrics.to_dict()


def _metrics_close(actual: dict, expected: dict) -> tuple[bool, list[str]]:
    diffs: list[str] = []
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        a = float(actual[key])
        e = float(expected[key])
        if not math.isclose(a, e, rel_tol=0, abs_tol=METRIC_TOLERANCE):
            diffs.append(f"{key}: actual={a:.6f} expected={e:.6f}")
    if actual.get("confusion_matrix") != expected.get("confusion_matrix"):
        diffs.append(
            f"confusion_matrix: actual={actual.get('confusion_matrix')} "
            f"expected={expected.get('confusion_matrix')}"
        )
    return not diffs, diffs


def export_all(*, include_real_world: bool = True) -> None:
    from export_report_artifacts import export_report_artifacts

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    audit_payload: dict[str, object] = {"runs": {}, "real_world": None}

    for run in RUNS:
        metrics, misclassifications, bootstrap = evaluate_run(run)
        metrics_path = (
            ROOT
            / "outputs"
            / ("ai_generated" if run["key"] == "resnet18" else "ai_generated_efficientnet")
            / "metrics.json"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        mis_path = ARTIFACTS / f"{run['key']}_misclassifications.json"
        mis_path.write_text(
            json.dumps([asdict(item) for item in misclassifications], indent=2),
            encoding="utf-8",
        )
        bootstrap_path = ARTIFACTS / f"{run['key']}_bootstrap.json"
        bootstrap_path.write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
        audit_payload["runs"][run["key"]] = {
            "metrics": metrics,
            "misclassifications": len(misclassifications),
            "bootstrap": bootstrap,
        }

    rw_metrics = evaluate_real_world() if include_real_world else None
    if rw_metrics is not None:
        REAL_WORLD["metrics_artifact"].write_text(
            json.dumps(rw_metrics, indent=2), encoding="utf-8"
        )
        audit_payload["real_world"] = rw_metrics

    export_report_artifacts(
        real_world_metrics=REAL_WORLD["metrics_artifact"] if rw_metrics is not None else None,
    )
    generate_summary(audit_payload)
    (ARTIFACTS / "audit_manifest.json").write_text(
        json.dumps(audit_payload, indent=2), encoding="utf-8"
    )


def generate_summary(audit_payload: dict[str, object]) -> None:
    lines = [
        "# Experiment summary (frozen results)",
        "",
        "Цей файл генерується `make experiment-audit`. "
        "Метрики можна **перевірити без повторного навчання**:",
        "",
        "```bash",
        "make download-hires-hf    # один раз",
        "make train && make train-efficientnet   # якщо checkpoint-ів ще немає",
        "make verify-results       # evaluate + порівняння з цим файлом",
        "```",
        "",
        "Checkpoint-и (`.pth`) не в git — зберігаються локально в `outputs/`.",
        "У git комітяться лише `metrics.json`, графіки та цей summary.",
        "",
        "## Test set (Parveshiiii/AI-vs-Real, N=204)",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Errors |",
        "|-------|----------|-----------|--------|-----|---------|--------|",
    ]

    for run in RUNS:
        key = run["key"]
        metrics = audit_payload["runs"][key]["metrics"]  # type: ignore[index]
        bootstrap = audit_payload["runs"][key]["bootstrap"]  # type: ignore[index]
        errors = audit_payload["runs"][key]["misclassifications"]  # type: ignore[index]
        lines.append(
            f"| {metrics['model']} | {_pct(metrics['accuracy'])} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f1']:.3f} | {metrics['roc_auc']:.3f} | {errors} |"
        )

    for run in RUNS:
        key = run["key"]
        metrics = audit_payload["runs"][key]["metrics"]  # type: ignore[index]
        bootstrap = audit_payload["runs"][key]["bootstrap"]  # type: ignore[index]
        cm = metrics["confusion_matrix"]
        lines.extend(
            [
                "",
                f"### {metrics['model']}",
                "",
                f"- Bootstrap accuracy 95% CI: "
                f"[{_pct(bootstrap['accuracy_ci95_low'])}, "
                f"{_pct(bootstrap['accuracy_ci95_high'])}] "
                f"(mean {_pct(bootstrap['accuracy_mean'])})",
                f"- Confusion matrix [[TN, FP], [FN, TP]]: `{cm}`",
            ]
        )

    rw = audit_payload.get("real_world")
    if rw:
        lines.extend(
            [
                "",
                "## Out-of-domain: data/real_world/",
                "",
                f"- N = {sum(sum(row) for row in rw['confusion_matrix'])}",
                f"- Accuracy: {_pct(rw['accuracy'])}",
                f"- F1: {rw['f1']:.3f}",
                f"- ROC-AUC: {rw['roc_auc']:.3f}",
                "",
                "Падіння якості vs in-domain підтверджує domain shift (див. звіт §5.3).",
            ]
        )

    lines.extend(
        [
            "",
            "## Files in docs/artifacts/",
            "",
            "- `*_test_metrics.json` — повні метрики test set",
            "- `*_misclassifications.json` — список помилкових передбачень",
            "- `*_bootstrap.json` — bootstrap 95% CI для accuracy",
            "- `*.png` — графіки порівняння (`make compare`)",
            "",
            "## Config fingerprint",
            "",
            "- Dataset: `data/ai_hires/` (HF Parveshiiii/AI-vs-Real, 512px)",
            "- Train: 10 epochs, batch 32, lr 0.001, image 224, strong_augment",
            "- Seed: 42",
            "- Checkpoint selection: best val accuracy",
        ]
    )
    (ARTIFACTS / "EXPERIMENT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify() -> int:
    failures: list[str] = []
    for run in RUNS:
        if not run["checkpoint"].is_file():
            failures.append(f"{run['key']}: checkpoint missing at {run['checkpoint']}")
            continue
        if not run["metrics_artifact"].is_file():
            failures.append(f"{run['key']}: artifact missing at {run['metrics_artifact']}")
            continue
        actual, _, _ = evaluate_run(run)
        expected = json.loads(run["metrics_artifact"].read_text(encoding="utf-8"))
        ok, diffs = _metrics_close(actual, expected)
        if not ok:
            failures.append(f"{run['key']}: " + "; ".join(diffs))

    if (
        REAL_WORLD["metrics_artifact"].is_file()
        and REAL_WORLD["checkpoint"].is_file()
        and REAL_WORLD["data_dir"].is_dir()
    ):
        actual_rw = evaluate_real_world()
        if actual_rw is None:
            failures.append("real_world: evaluation failed")
        else:
            expected_rw = json.loads(REAL_WORLD["metrics_artifact"].read_text(encoding="utf-8"))
            ok, diffs = _metrics_close(actual_rw, expected_rw)
            if not ok:
                failures.append("real_world: " + "; ".join(diffs))

    if failures:
        print("Verification FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("Verification OK: live evaluate matches docs/artifacts/*.json")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or verify KKP experiment artifacts")
    parser.add_argument(
        "command",
        choices=("export", "verify", "summary"),
        help="export=regenerate artifacts; verify=compare checkpoints to committed metrics",
    )
    args = parser.parse_args()

    if args.command == "export":
        export_all()
        print(f"Artifacts exported to {ARTIFACTS}")
    elif args.command == "verify":
        raise SystemExit(verify())
    else:
        if not (ARTIFACTS / "resnet18_test_metrics.json").is_file():
            export_all()
        else:
            payload = json.loads((ARTIFACTS / "audit_manifest.json").read_text(encoding="utf-8"))
            generate_summary(payload)
        print(f"Summary: {ARTIFACTS / 'EXPERIMENT_SUMMARY.md'}")


if __name__ == "__main__":
    main()
