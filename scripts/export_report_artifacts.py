# Copy experiment artifacts into docs/artifacts for report reproducibility.

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs" / "artifacts"
COMPARISON = ROOT / "outputs" / "comparison"

RUNS = (
    ("resnet18", ROOT / "outputs" / "ai_generated" / "metrics.json"),
    ("efficientnet_b0", ROOT / "outputs" / "ai_generated_efficientnet" / "metrics.json"),
)

COMPARISON_PLOTS = (
    "metrics_comparison.png",
    "learning_curves.png",
    "confusion_matrices.png",
    "roc_curves.png",
)


def export_report_artifacts(*, real_world_metrics: Path | None = None) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    for name, path in RUNS:
        if not path.is_file():
            msg = f"Missing metrics file: {path}. Run make evaluate-all first."
            raise FileNotFoundError(msg)
        shutil.copy2(path, ARTIFACTS / f"{name}_test_metrics.json")

    if COMPARISON.is_dir():
        for plot in COMPARISON_PLOTS:
            src = COMPARISON / plot
            if src.is_file():
                shutil.copy2(src, ARTIFACTS / plot)

    if real_world_metrics is not None and real_world_metrics.is_file():
        dst = ARTIFACTS / "efficientnet_real_world_metrics.json"
        if real_world_metrics.resolve() != dst.resolve():
            shutil.copy2(real_world_metrics, dst)

    summary = {
        "runs": [name for name, _ in RUNS],
        "comparison_plots": list(COMPARISON_PLOTS),
    }
    (ARTIFACTS / "README.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ARTIFACTS


def main() -> None:
    path = export_report_artifacts()
    print(f"Artifacts: {path}")


if __name__ == "__main__":
    main()
