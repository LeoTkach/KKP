"""Gradio demo for AI-generated image detection."""

from __future__ import annotations

import argparse
import logging
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from dotenv import load_dotenv
from PIL import Image

from kkp.config import load_config
from kkp.data.transforms import get_transforms
from kkp.inference import predict_image
from kkp.training import load_checkpoint

logger = logging.getLogger(__name__)

LABELS_UA = {
    "real": "Реальне фото",
    "ai_generated": "AI-генерація",
}

VERDICT_REAL = "Реальне фото"
VERDICT_AI = "AI-генерація"

EMPTY_RESULTS = """
<div class="results-card results-card--empty">
    <p>Оберіть зображення та натисніть «Перевірити».</p>
</div>
"""
EMPTY_SOURCE = ""

RADIUS = "8px"
RADIUS_SM = "4px"
RADIUS_PILL = "999px"

CUSTOM_CSS = f"""
:root {{
    --kkp-radius: {RADIUS};
    --kkp-radius-sm: {RADIUS_SM};
    --kkp-radius-pill: {RADIUS_PILL};
    --kkp-font: "IBM Plex Sans", "Segoe UI", sans-serif;
    --kkp-font-mono: "IBM Plex Mono", ui-monospace, monospace;
    --kkp-text-xs: 0.6875rem;
    --kkp-text-sm: 0.8125rem;
    --kkp-text-base: 0.9375rem;
    --kkp-text-lg: 1.0625rem;
    --kkp-text-xl: 1.25rem;
    --kkp-text-2xl: 1.375rem;
    --kkp-bg: #0b0b0c;
    --kkp-surface: #121214;
    --kkp-border: rgba(255, 255, 255, 0.08);
    --kkp-border-strong: rgba(255, 255, 255, 0.14);
    --kkp-muted: #8b8b93;
    --kkp-gap: 0.625rem;
    --kkp-gap-lg: 1rem;
    --kkp-page-x: clamp(1rem, 2.5vw, 1.5rem);
    --kkp-page-max: 1200px;
    --kkp-panel-pad: 1.25rem 1.35rem;
}}
.gradio-container {{
    background: var(--kkp-bg) !important;
    font-family: var(--kkp-font) !important;
}}
.gradio-container .main {{
    max-width: var(--kkp-page-max) !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 1rem var(--kkp-page-x) 1.25rem !important;
}}
.gradio-container .wrap {{
    max-width: none !important;
}}
.page-header {{
    margin-bottom: 0.35rem;
}}
.hero-title {{
    margin: 0;
    font-size: var(--kkp-text-2xl);
    font-weight: 500;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}
.hero-subtitle {{
    margin: 0.35rem 0 0;
    color: var(--kkp-muted);
    font-size: var(--kkp-text-sm);
    line-height: 1.45;
}}
#kkp-main-row {{
    gap: 1.25rem !important;
    align-items: flex-start !important;
}}
#kkp-input-col {{
    gap: var(--kkp-gap) !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}}
#kkp-results-col {{
    gap: var(--kkp-gap) !important;
    flex: 1 1 0 !important;
    min-width: 0 !important;
}}
#kkp-image-input,
#kkp-image-input .wrap {{
    width: 100% !important;
}}
#kkp-image-input .image-container {{
    border: 1px solid var(--kkp-border) !important;
    border-radius: var(--kkp-radius) !important;
    background: var(--kkp-surface) !important;
    min-height: 280px !important;
    height: 320px !important;
    max-height: none !important;
    overflow: hidden !important;
}}
#kkp-image-input img {{
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
    max-height: none !important;
}}
#kkp-actions {{
    gap: 0.45rem !important;
    width: 100% !important;
}}
#kkp-actions > .form {{
    flex: 1 1 0 !important;
}}
#kkp-actions button {{
    width: 100% !important;
    padding: 0.35rem 0.75rem !important;
    font-size: var(--kkp-text-sm) !important;
    font-weight: 500 !important;
    border: 1px solid var(--kkp-border) !important;
    border-radius: var(--kkp-radius-sm) !important;
    box-shadow: none !important;
}}
#kkp-actions button.primary {{
    background: #ececee !important;
    color: #111 !important;
    border-color: #ececee !important;
}}
#kkp-actions button.primary:hover {{
    background: #fff !important;
}}
#kkp-actions button:not(.primary) {{
    background: transparent !important;
    color: #d4d4d8 !important;
}}
#kkp-actions button:not(.primary):hover {{
    background: rgba(255, 255, 255, 0.04) !important;
    border-color: var(--kkp-border-strong) !important;
}}
#kkp-results-output .block,
#kkp-results-output .html-container,
#kkp-results-output .wrap {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}}
.results-card {{
    border: 1px solid var(--kkp-border);
    border-radius: var(--kkp-radius);
    padding: var(--kkp-panel-pad);
    background: transparent;
}}
.results-card--empty {{
    color: var(--kkp-muted);
    font-size: var(--kkp-text-sm);
    line-height: 1.45;
}}
.results-card--empty p {{
    margin: 0;
}}
.results-title {{
    margin: 0;
    font-size: var(--kkp-text-xl);
    font-weight: 500;
    letter-spacing: -0.015em;
    line-height: 1.2;
    color: #f4f4f5;
}}
.results-conf {{
    margin: 0.35rem 0 0;
    font-size: var(--kkp-text-sm);
    color: var(--kkp-muted);
}}
.results-probs {{
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--kkp-border);
}}
.prob-row {{ margin-bottom: 0.65rem; }}
.prob-row:last-child {{ margin-bottom: 0; }}
.prob-header {{
    display: flex;
    justify-content: space-between;
    font-size: var(--kkp-text-sm);
    margin-bottom: 0.3rem;
    color: #c4c4cc;
}}
.prob-track {{
    height: 3px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: var(--kkp-radius-pill);
    overflow: hidden;
}}
.prob-fill {{
    height: 100%;
    background: #d4d4d8;
    border-radius: var(--kkp-radius-pill);
}}
.prob-fill.real {{ background: #f4f4f5; }}
#kkp-source-wrap:empty {{
    display: none;
}}
.source-box {{
    margin-top: var(--kkp-gap);
    padding: 0.55rem 0.75rem;
    border-radius: var(--kkp-radius-sm);
    border: 1px dashed var(--kkp-border-strong);
    background: rgba(255, 255, 255, 0.02);
    font-size: var(--kkp-text-sm);
    line-height: 1.45;
    color: #b4b4bc;
}}
.source-box .match-ok {{ color: #a3e635; }}
.source-box .match-bad {{ color: #fb7185; }}
.source-tag {{
    display: block;
    margin-top: 0.25rem;
    font-size: var(--kkp-text-xs);
    color: var(--kkp-muted);
    font-family: var(--kkp-font-mono);
}}
.gradio-container .contain,
.gradio-container .gap {{
    gap: var(--kkp-gap) !important;
}}
.gradio-container .block {{
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}
.gradio-container .form {{
    gap: var(--kkp-gap) !important;
}}
#kkp-image-input .label-wrap,
#kkp-image-input .icon-button,
#kkp-image-input .icon-buttons {{
    display: none !important;
}}
#kkp-actions button {{
    min-height: 2.125rem !important;
}}
.model-footer {{
    margin-top: 0.5rem;
    padding: 0.85rem 0 0;
    border-top: 1px solid var(--kkp-border);
}}
.model-footer-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.35rem 0.55rem;
    font-size: var(--kkp-text-sm);
    color: #a1a1aa;
    line-height: 1.5;
}}
.model-footer-row b {{
    color: #e4e4e7;
    font-weight: 500;
}}
.model-footer-sep {{
    color: #3f3f46;
    user-select: none;
}}
.model-footer-path {{
    margin-top: 0.35rem;
    font-size: var(--kkp-text-xs);
    color: #52525b;
    font-family: var(--kkp-font-mono);
    word-break: break-all;
}}
.gradio-container .loader,
.gradio-container .loading,
.gradio-container .progress-bar,
.gradio-container .eta-bar,
.gradio-container .meta-text-center,
.gradio-container [class*="progress"] {{
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    pointer-events: none !important;
}}
#kkp-image-input .progress-bar,
#kkp-image-input .loader {{
    display: none !important;
}}
"""


def _dark_theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue="neutral",
        secondary_hue="neutral",
        neutral_hue="neutral",
        font=[
            gr.themes.GoogleFont("IBM Plex Sans"),
            "Segoe UI",
            "sans-serif",
        ],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    ).set(
        body_background_fill="#0b0b0c",
        body_text_color="#ececee",
        body_text_color_subdued="#8b8b93",
        background_fill_primary="#121214",
        background_fill_secondary="#121214",
        border_color_primary="rgba(255,255,255,0.08)",
        block_background_fill="transparent",
        block_border_color="transparent",
        block_label_text_color="#a1a1aa",
        block_title_text_color="#ececee",
        input_background_fill="#121214",
        button_primary_background_fill="#ececee",
        button_primary_background_fill_hover="#ffffff",
        button_primary_text_color="#111111",
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_hover="rgba(255,255,255,0.04)",
        panel_background_fill="transparent",
        table_even_background_fill="#121214",
        table_odd_background_fill="#0b0b0c",
        color_accent="#a1a1aa",
        color_accent_soft="rgba(255,255,255,0.06)",
    )


@dataclass(frozen=True)
class ModelProfile:
    """Metadata shown in the demo footer; swap when adding model selection."""

    profile_id: str
    display_name: str
    architecture: str
    epoch: int
    input_size: int
    dataset: str
    test_accuracy: str
    train_samples: str
    checkpoint: str

    def footer_html(self) -> str:
        parts = (
            f"<b>{self.architecture}</b>",
            f"ep. {self.epoch}",
            f"{self.input_size}px",
            f"acc {self.test_accuracy}",
            self.dataset,
            f"{self.train_samples} train",
        )
        row = '<span class="model-footer-sep">·</span>'.join(
            f"<span>{part}</span>" for part in parts
        )
        return f"""
        <div class="model-footer" id="model-footer-{self.profile_id}">
            <div class="model-footer-row">{row}</div>
            <div class="model-footer-path">{self.checkpoint}</div>
        </div>
        """


def _count_train_samples(data_dir: Path) -> str:
    train_real = data_dir / "train" / "REAL"
    train_fake = data_dir / "train" / "FAKE"
    if not train_real.is_dir() or not train_fake.is_dir():
        return "—"
    count = len(list(train_real.glob("*.jpg"))) + len(list(train_fake.glob("*.jpg")))
    return str(count) if count else "—"


def build_model_profile(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    *,
    test_accuracy: str = "93.6%",
    dataset_name: str = "Parveshiiii/AI-vs-Real",
) -> ModelProfile:
    data_dir = Path(config["paths"]["data_dir"])
    project = config.get("project", {}).get("name", "kkp")
    model_name = config["model"]["name"]
    return ModelProfile(
        profile_id=f"{project}-{model_name}",
        display_name=f"{project} / {model_name}",
        architecture=model_name.upper(),
        epoch=int(checkpoint["epoch"]),
        input_size=int(config["training"]["image_size"]),
        dataset=dataset_name,
        test_accuracy=test_accuracy,
        train_samples=_count_train_samples(data_dir),
        checkpoint=str(checkpoint_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KKP Gradio demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ai_generated.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser.parse_args()


def _probability_rows_html(confidences: dict[str, float]) -> str:
    rows: list[str] = []
    for name, score in confidences.items():
        label = LABELS_UA.get(name, name)
        pct = score * 100
        css_class = "real" if name == "real" else ""
        rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-header"><span>{label}</span><span>{pct:.1f}%</span></div>
                <div class="prob-track">
                    <div class="prob-fill {css_class}" style="width:{pct:.1f}%"></div>
                </div>
            </div>
            """,
        )
    return "".join(rows)


def _results_html(title: str, confidence_pct: float, confidences: dict[str, float]) -> str:
    return f"""
    <div class="results-card">
        <div class="results-verdict">
            <p class="results-title">{title}</p>
            <p class="results-conf">Впевненість: {confidence_pct:.1f}%</p>
        </div>
        <div class="results-probs">{_probability_rows_html(confidences)}</div>
    </div>
    """


def _empty_results_html(message: str) -> str:
    return f"""
    <div class="results-card results-card--empty">
        <p>{message}</p>
    </div>
    """


def _discover_dataset_samples(
    data_dir: Path,
    *,
    splits: tuple[str, ...] = ("test", "val", "train"),
) -> list[tuple[Path, str, str]]:
    samples: list[tuple[Path, str, str]] = []
    for split in splits:
        for label_dir in ("REAL", "FAKE"):
            folder = data_dir / split / label_dir
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob("*.jpg")):
                samples.append((path, split, label_dir))
    return samples


def _label_dir_to_class(label_dir: str) -> str:
    return "real" if label_dir == "REAL" else "ai_generated"


def _source_html(
    source: dict[str, str],
    *,
    predicted: str,
) -> str:
    expected = source["expected_class"]
    expected_ua = LABELS_UA[expected]
    match = predicted == expected
    match_text = "Збіг" if match else "Помилка"
    match_class = "match-ok" if match else "match-bad"
    return f"""
    <div class="source-box">
        <strong>З датасету:</strong> {source["split"]} / {source["label_dir"]}
        · очікується {expected_ua}
        · <span class="{match_class}">{match_text}</span>
        <br>
        <span class="source-tag">{source["relative_path"]}</span>
    </div>
    """


def build_predict_fn(
    config_path: Path,
    checkpoint_path: Path | None,
) -> tuple:
    config = load_config(config_path)
    training = config["training"]
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    resolved_checkpoint = (
        checkpoint_path if checkpoint_path else output_dir / "checkpoints" / "best.pth"
    )

    if not resolved_checkpoint.is_file():
        msg = f"Checkpoint not found: {resolved_checkpoint}"
        raise FileNotFoundError(msg)

    device = torch.device(training["device"])
    model, checkpoint = load_checkpoint(resolved_checkpoint, device)
    transform = get_transforms(training["image_size"], train=False)
    class_names = tuple(config["data"]["classes"])

    dataset_samples = _discover_dataset_samples(data_dir)

    def _run_prediction(
        image: Image.Image,
        source: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        label, confidences = predict_image(model, image, transform, device, class_names)
        winner_score = confidences[label] * 100
        title = VERDICT_REAL if label == "real" else VERDICT_AI
        source_html = _source_html(source, predicted=label) if source is not None else EMPTY_SOURCE
        return _results_html(title, winner_score, confidences), source_html

    def predict(image: Image.Image) -> tuple[str, str]:
        if image is None:
            return EMPTY_RESULTS, EMPTY_SOURCE
        return _run_prediction(image)

    def random_from_dataset() -> tuple[Image.Image | None, str, str]:
        if not dataset_samples:
            msg = _empty_results_html("Датасет не знайдено. Запустіть `make download-hires-hf`.")
            return None, msg, EMPTY_SOURCE

        path, split, label_dir = random.choice(dataset_samples)
        image = Image.open(path).convert("RGB")
        source = {
            "split": split,
            "label_dir": label_dir,
            "expected_class": _label_dir_to_class(label_dir),
            "relative_path": f"{split}/{label_dir}/{path.name}",
        }
        results_html, source_html = _run_prediction(image, source=source)
        return image, results_html, source_html

    logger.info(
        "Loaded %s (epoch %d) from %s",
        config["model"]["name"],
        checkpoint["epoch"],
        resolved_checkpoint,
    )

    model_profile = build_model_profile(config, checkpoint, resolved_checkpoint)

    return predict, random_from_dataset, model_profile


def create_demo(
    predict_fn,
    random_from_dataset,
    model_profile: ModelProfile,
) -> gr.Blocks:
    with gr.Blocks(
        title="KKP — AI vs Real",
        theme=_dark_theme(),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(
            """
            <header class="page-header">
                <h1 class="hero-title">Детекція AI-зображень</h1>
                <p class="hero-subtitle">
                    Завантажте фото або візьміть випадкове з датасету — модель оцінить,
                    чи це реальне зображення чи AI-генерація.
                </p>
            </header>
            """,
        )

        with gr.Row(elem_id="kkp-main-row"):
            with gr.Column(scale=1, elem_id="kkp-input-col"):
                image_input = gr.Image(
                    type="pil",
                    label="Зображення",
                    height=320,
                    sources=["upload"],
                    buttons=[],
                    elem_id="kkp-image-input",
                    container=False,
                )
                with gr.Row(elem_id="kkp-actions"):
                    analyze_btn = gr.Button("Перевірити", variant="primary", scale=1)
                    random_btn = gr.Button("Випадкове", scale=1)
                    clear_btn = gr.Button("Очистити", scale=1)

            with gr.Column(scale=1, elem_id="kkp-results-col"):
                results_output = gr.HTML(value=EMPTY_RESULTS, elem_id="kkp-results-output")
                source_output = gr.HTML(value=EMPTY_SOURCE, elem_id="kkp-source-wrap")

        outputs = [results_output, source_output]
        all_outputs = [image_input, *outputs]
        analyze_btn.click(
            predict_fn,
            inputs=image_input,
            outputs=outputs,
            show_progress="hidden",
        )
        random_btn.click(
            random_from_dataset,
            outputs=all_outputs,
            show_progress="hidden",
        )
        clear_btn.click(
            lambda: (None, EMPTY_RESULTS, EMPTY_SOURCE),
            outputs=[image_input, results_output, source_output],
            show_progress="hidden",
        )

        gr.HTML(model_profile.footer_html(), elem_id="kkp-model-footer")

    return demo


def _resolve_server_port(host: str, preferred: int, *, attempts: int = 10) -> int:
    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    msg = f"No free port in range {preferred}-{preferred + attempts - 1}"
    raise OSError(msg)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    load_dotenv()
    args = parse_args()

    predict_fn, random_from_dataset, model_profile = build_predict_fn(
        args.config,
        args.checkpoint,
    )
    demo = create_demo(predict_fn, random_from_dataset, model_profile)
    port = _resolve_server_port(args.host, args.port)
    if port != args.port:
        logger.info("Port %d busy, using %d instead", args.port, port)
    demo.launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        footer_links=[],
        pwa=False,
    )


if __name__ == "__main__":
    main()
