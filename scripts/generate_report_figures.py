"""Generate static figures for KKP explanatory note (architecture diagrams)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle


def _solid_line(ax, start, end, *, lw=1.4, color="#1a1a1a"):
    ax.plot(
        [start[0], end[0]], [start[1], end[1]], color=color, linewidth=lw, solid_capstyle="round"
    )


def _assoc_arrow(ax, start, end, *, color="#1a1a1a"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def _extend_arrow(ax, start, end, *, label="<<extend>>", label_offset=(0.0, 0.0)):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color="#1a1a1a",
            linestyle=(0, (5, 3)),
            shrinkA=4,
            shrinkB=4,
        )
    )
    mid = ((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1])
    ax.text(mid[0], mid[1], label, fontsize=7, ha="center", va="center", color="#1a1a1a")


def _stick_figure(ax, x, y, label: str, *, scale: float = 1.0) -> None:
    s = scale
    head_y = y + 0.38 * s
    ax.add_patch(plt.Circle((x, head_y), 0.11 * s, fill=False, ec="#1a1a1a", lw=1.4))
    _solid_line(ax, (x, head_y - 0.11 * s), (x, y + 0.02 * s))
    _solid_line(ax, (x - 0.18 * s, y + 0.18 * s), (x + 0.18 * s, y + 0.18 * s))
    _solid_line(ax, (x, y + 0.02 * s), (x - 0.14 * s, y - 0.32 * s))
    _solid_line(ax, (x, y + 0.02 * s), (x + 0.14 * s, y - 0.32 * s))
    ax.text(x, y - 0.48 * s, label, ha="center", va="top", fontsize=9.5)


def _use_case(ax, cx, cy, text: str, *, w=2.1, h=0.62) -> None:
    ell = Ellipse((cx, cy), w, h, fill=False, ec="#1a1a1a", lw=1.4)
    ax.add_patch(ell)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.5)


def _ellipse_edge(cx, cy, tx, ty, *, w=2.1, h=0.62) -> tuple[float, float]:
    dx, dy = tx - cx, ty - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return cx, cy
    rx, ry = w / 2, h / 2
    scale = 1.0 / ((dx / rx) ** 2 + (dy / ry) ** 2) ** 0.5
    return cx + dx * scale, cy + dy * scale


def draw_use_cases(output: Path) -> None:
    """UML use case diagram: actors, ovals, extend to base use cases."""
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    bx, by, bw, bh = 2.05, 0.5, 7.5, 6.0
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor="#1a1a1a", linewidth=1.6))
    ax.text(
        bx + 0.18,
        by + bh - 0.28,
        "Програмна система детекції AI-зображень (KKP)",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )

    w, h = 2.15, 0.62
    pos = {
        "download": (4.5, 4.0),
        "train": (4.5, 2.6),
        "eval": (7.2, 2.6),
        "compare": (7.2, 1.1),
        "classify": (8.0, 4.0),
    }
    labels = {
        "download": "Завантаження\nдатасету",
        "train": "Навчання\nмоделі",
        "eval": "Оцінка на\ntest set",
        "compare": "Порівняння\nмоделей",
        "classify": "Класифікація\nзображення",
    }
    for key in pos:
        cx, cy = pos[key]
        _use_case(ax, cx, cy, labels[key], w=w, h=h)

    tr, ev, co = pos["train"], pos["eval"], pos["compare"]
    # extend: розширення → базовий сценарій (як у методичці)
    _extend_arrow(
        ax,
        _ellipse_edge(*co, ev[0], ev[1] + 0.5, w=w, h=h),
        (ev[0], ev[1] - h / 2 - 0.02),
        label_offset=(0.45, 0),
    )
    _extend_arrow(
        ax,
        _ellipse_edge(*ev, *tr, w=w, h=h),
        _ellipse_edge(*tr, *ev, w=w, h=h),
        label_offset=(0, 0.18),
    )

    _stick_figure(ax, 0.65, 2.85, "Дослідник", scale=1.05)
    _stick_figure(ax, 10.05, 3.75, "Користувач", scale=1.05)

    res_hand = (0.92, 3.05)
    _assoc_arrow(ax, res_hand, _ellipse_edge(*pos["download"], *res_hand, w=w, h=h))
    _assoc_arrow(ax, res_hand, _ellipse_edge(*pos["train"], *res_hand, w=w, h=h))

    user_hand = (9.78, 3.85)
    _assoc_arrow(ax, user_hand, _ellipse_edge(*pos["classify"], *user_hand, w=w, h=h))

    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_pipeline(output: Path) -> None:
    """Training + usage flow: command → action → result per block."""
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def step_box(xy, cmd, action, result, *, width=2.55, height=1.05, fc="#E8F4FC", ec="#1B4965"):
        x, y = xy
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height * 0.78,
            cmd,
            ha="center",
            va="center",
            fontsize=8,
            fontfamily="monospace",
            fontweight="bold",
            color="#0F172A",
        )
        ax.text(
            x + width / 2,
            y + height * 0.48,
            action,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#1E293B",
        )
        ax.text(
            x + width / 2,
            y + height * 0.18,
            result,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#64748B",
            style="italic",
        )

    def h_arrow(y, x1, x2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y),
                (x2, y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#475569",
            )
        )

    train_y = 3.55
    train_steps = [
        ("make download-hires-hf", "Завантажити датасет з HF", "→ data/ai_hires/"),
        ("make train", "Навчити CNN-модель", "→ outputs/.../best.pth"),
        ("make evaluate-all", "Оцінити на test set", "→ metrics.json"),
        ("make compare", "Порівняти моделі", "→ comparison/*.png"),
    ]
    x0 = 0.35
    gap = 0.35
    w = 2.55
    for i, (cmd, action, result) in enumerate(train_steps):
        x = x0 + i * (w + gap)
        step_box((x, train_y), cmd, action, result, width=w)
        if i < len(train_steps) - 1:
            h_arrow(train_y + 0.52, x + w + 0.02, x + w + gap - 0.02)

    bridge_y = 2.35
    ax.add_patch(
        FancyBboxPatch(
            (4.2, bridge_y),
            3.6,
            0.65,
            boxstyle="round,pad=0.03",
            facecolor="#EDE9FE",
            edgecolor="#7C3AED",
            linewidth=1.2,
        )
    )
    ax.text(
        6,
        bridge_y + 0.32,
        "best.pth → demo",
        ha="center",
        fontsize=9,
        color="#5B21B6",
        fontweight="bold",
    )
    ax.add_patch(
        FancyArrowPatch(
            (6, train_y),
            (6, bridge_y + 0.65),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#7C3AED",
        )
    )

    usage_y = 0.85
    usage_steps = [
        ("make demo", "Запустити Gradio-сервер", "→ localhost:7860"),
        ("Завантажити JPEG", "Користувач обирає фото", "→ predict_image()"),
        ("Відповідь UI", "Клас + confidence", "→ real / ai_generated"),
    ]
    uw = 3.15
    ugap = 0.45
    for i, (cmd, action, result) in enumerate(usage_steps):
        x = x0 + i * (uw + ugap)
        step_box((x, usage_y), cmd, action, result, width=uw, fc="#FEF3C7", ec="#B45309")
        if i < len(usage_steps) - 1:
            h_arrow(usage_y + 0.52, x + uw + 0.02, x + uw + ugap - 0.02)

    ax.add_patch(
        FancyArrowPatch(
            (6, bridge_y),
            (6, usage_y + 1.05),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#B45309",
        )
    )

    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_modules(output: Path) -> None:
    """File tree with hints on key nodes."""
    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    nodes: list[tuple[str, str, str | None]] = [
        ("", "KKP/", None),
        ("├── ", "Makefile", "точка входу CLI"),
        ("├── ", "configs/", "YAML-параметри"),
        ("│   ├── ", "ai_generated.yaml", "ResNet18"),
        ("│   └── ", "ai_generated_efficientnet.yaml", "EfficientNet-B0"),
        ("├── ", "data/", None),
        ("│   └── ", "ai_hires/", "ImageFolder REAL/FAKE"),
        ("│       ├── ", "train/", None),
        ("│       ├── ", "val/", None),
        ("│       └── ", "test/", None),
        ("├── ", "src/kkp/", "основний код"),
        ("│   ├── ", "train.py", None),
        ("│   ├── ", "evaluate.py", None),
        ("│   ├── ", "compare.py", None),
        ("│   ├── ", "demo.py", "Gradio UI"),
        ("│   ├── ", "data/", "loaders, transforms"),
        ("│   ├── ", "models/", "create_model()"),
        ("│   ├── ", "training/", "train loop, metrics"),
        ("│   └── ", "inference.py", "predict_image()"),
        ("├── ", "outputs/", "артефакти експериментів"),
        ("│   ├── ", "ai_generated/", "ResNet18"),
        ("│   ├── ", "ai_generated_efficientnet/", "EfficientNet-B0"),
        ("│   └── ", "comparison/", "графіки порівняння"),
        ("├── ", "docs/artifacts/", "metrics.json для звіту"),
        ("└── ", "tests/", "pytest"),
    ]

    y = 7.95
    line_h = 0.28
    tree_x = 0.25
    name_x = 0.72
    hint_x = 5.05

    for prefix, name, hint in nodes:
        is_root = name == "KKP/"
        is_dir = name.endswith("/")
        weight = "bold" if is_root or is_dir else "normal"
        color = "#1D4ED8" if is_dir or is_root else "#334155"
        ax.text(tree_x, y, prefix, fontsize=9, fontfamily="monospace", color="#94A3B8", va="top")
        ax.text(
            name_x,
            y,
            name,
            fontsize=10 if is_root else 9,
            fontfamily="monospace",
            fontweight=weight,
            color=color,
            va="top",
        )
        if hint:
            ax.text(hint_x, y, f"— {hint}", fontsize=8.5, color="#B45309", va="top")
        y -= line_h

    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_testing_pyramid(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    levels = [
        (1.2, 2.95, 7.6, 0.9, "CI/CD", "lint · pytest · docker build", "#FEE2E2"),
        (1.7, 1.75, 6.6, 0.9, "Інтеграційні", "loaders · smoke train", "#FEF3C7"),
        (2.2, 0.55, 5.6, 1.0, "Модульні", "dataset · model · metrics · inference", "#DCFCE7"),
    ]
    for x, y, width, height, title, subtitle, color in levels:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            linewidth=1.2,
            edgecolor="#1B4965",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y + height * 0.28,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#334155",
        )

    fig.tight_layout(pad=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_misclassification_examples(output: Path, *, data_root: Path) -> None:
    """Collage of representative test-set misclassifications (ResNet + EfficientNet)."""
    from PIL import Image

    examples: list[tuple[str, str, str, str]] = [
        ("test_real_0049.jpg", "REAL", "ResNet18 → AI (98,4 %)", "REAL"),
        ("test_fake_0043.jpg", "FAKE", "ResNet18 → REAL (58,5 %)", "FAKE"),
        ("test_real_0004.jpg", "REAL", "EffNet → AI (99,3 %)", "REAL"),
        ("test_real_0043.jpg", "REAL", "EffNet → AI (85,4 %)", "REAL"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.2))
    axes_flat = axes.flatten()

    for ax, (fname, true_label, pred_text, subdir) in zip(axes_flat, examples, strict=True):
        path = data_root / "ai_hires" / "test" / subdir / fname
        ax.axis("off")
        if path.is_file():
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"{fname}\n(файл відсутній)", ha="center", va="center", fontsize=9)
        ax.set_title(f"{fname}\nСправжній: {true_label} · {pred_text}", fontsize=8.5, pad=6)

    fig.suptitle("Приклади помилкових класифікацій на test set (N=204)", fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_report_figures(output_dir: Path, *, data_root: Path | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "use_cases": output_dir / "use_cases.png",
        "pipeline": output_dir / "pipeline.png",
        "modules": output_dir / "modules.png",
        "testing_pyramid": output_dir / "testing_pyramid.png",
    }
    draw_use_cases(paths["use_cases"])
    draw_pipeline(paths["pipeline"])
    draw_modules(paths["modules"])
    draw_testing_pyramid(paths["testing_pyramid"])
    if data_root is not None:
        paths["misclassification_examples"] = output_dir / "misclassification_examples.png"
        draw_misclassification_examples(paths["misclassification_examples"], data_root=data_root)
    return paths


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    out = root / "docs" / "report_assets"
    generate_report_figures(out, data_root=root / "data")
    print(f"Report figures: {out}")
