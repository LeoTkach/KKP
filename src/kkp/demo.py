"""Gradio demo for AI-generated image detection."""

from __future__ import annotations

import argparse
import json
import logging
import random
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from torchvision.transforms import Compose

from kkp.config import load_config
from kkp.data.transforms import get_transforms
from kkp.inference import predict_image
from kkp.training import load_checkpoint

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIGS: dict[str, Path] = {
    "resnet18": Path("configs/ai_generated.yaml"),
    "efficientnet_b0": Path("configs/ai_generated_efficientnet.yaml"),
}

MODEL_LABELS_UA: dict[str, str] = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "resnet18": "Класична residual CNN — швидша, baseline для порівняння.",
    "efficientnet_b0": "Compound scaling — менше параметрів, часто точніша на тому ж датасеті.",
}

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500&display=swap');
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
    width: 100% !important;
    align-self: stretch !important;
}}
#kkp-page,
#kkp-page.column {{
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    flex: 1 1 100% !important;
    align-self: stretch !important;
}}
#kkp-page > .gap {{
    align-items: stretch !important;
    width: 100% !important;
}}
#kkp-page > .gap > *,
#kkp-compare-section,
#kkp-compare-section.column {{
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    align-self: stretch !important;
    flex: 1 1 100% !important;
}}
.gradio-container .prose,
#kkp-model-compare .prose,
#kkp-model-compare .html-content,
#kkp-model-info .prose,
#kkp-results-output .prose {{
    max-width: none !important;
    width: 100% !important;
}}
.gradio-container .main .column:not(.compact) {{
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}}
#kkp-results-output,
#kkp-model-compare,
#kkp-compare-section,
#kkp-compare-section .block,
#kkp-compare-section .html-container,
#kkp-compare-section .wrap,
#kkp-model-section,
#kkp-model-info {{
    width: 100% !important;
    max-width: 100% !important;
    align-self: stretch !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
}}
#kkp-results-output .block,
#kkp-results-output .html-container,
#kkp-results-output .wrap,
#kkp-model-compare .block,
#kkp-model-compare .html-container,
#kkp-model-compare .wrap,
#kkp-model-compare.container {{
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}}
#kkp-model-info .block,
#kkp-model-info .html-container,
#kkp-model-info .wrap {{
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}}
#kkp-model-compare .html-container {{
    display: block !important;
}}
#kkp-model-section {{
    width: 100% !important;
    max-width: 100% !important;
}}
.results-stack {{
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: var(--kkp-gap);
}}
.results-stack .results-card,
.results-stack .source-box {{
    width: 100%;
    box-sizing: border-box;
}}
.results-stack .source-box {{
    margin-top: 0;
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
    display: block;
    width: 100%;
    box-sizing: border-box;
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
.model-info {{
    margin-top: 0.55rem;
    padding: 0.75rem 1.15rem;
    border: 1px solid var(--kkp-border);
    border-radius: var(--kkp-radius-sm);
    background: rgba(0, 0, 0, 0.18);
}}
.model-info-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.35rem 0.55rem;
    font-size: var(--kkp-text-sm);
    color: #a1a1aa;
    line-height: 1.5;
}}
.model-info-arch {{
    color: #e4e4e7;
    font-weight: 500;
    font-family: var(--kkp-font-mono);
    font-size: var(--kkp-text-xs);
    letter-spacing: 0.02em;
}}
.model-info-sep {{
    color: #3f3f46;
    user-select: none;
}}
.model-info-path {{
    margin-top: 0.45rem;
    font-size: var(--kkp-text-xs);
    color: #52525b;
    font-family: var(--kkp-font-mono);
    word-break: break-all;
    line-height: 1.4;
}}
#kkp-model-section {{
    gap: 0 !important;
    margin-bottom: 0.85rem !important;
    width: 100% !important;
}}
#kkp-compare-section {{
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) !important;
    gap: 0 !important;
    margin-top: 0.85rem !important;
    padding: 0 !important;
    width: 100% !important;
}}
#kkp-compare-section > *,
#kkp-compare-section .block,
#kkp-compare-section .html-container,
#kkp-compare-section .wrap {{
    grid-column: 1 / -1 !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    align-self: stretch !important;
}}
#kkp-model-compare,
#kkp-model-compare.block {{
    display: block !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    align-self: stretch !important;
}}
.bench-shell {{
    display: block;
    width: 100%;
    min-width: 100%;
    box-sizing: border-box;
}}
#kkp-model-picker {{
    align-items: stretch !important;
    gap: 0 !important;
    margin: 0 !important;
    border: 1px solid var(--kkp-border);
    border-radius: var(--kkp-radius);
    overflow: hidden;
    background: var(--kkp-surface);
}}
#kkp-model-info .block,
#kkp-model-info .html-container,
#kkp-model-info .wrap {{
    padding: 0 !important;
    margin: 0 !important;
}}
#kkp-model-info .model-info {{
    margin-top: 0.55rem;
}}
#kkp-model-picker-label,
#kkp-model-picker-label .block,
#kkp-model-picker-label .html-container,
#kkp-model-picker-label .wrap {{
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}}
#kkp-model-picker-label {{
    flex: 0 0 clamp(9.5rem, 22vw, 11.5rem) !important;
    border-right: 1px solid var(--kkp-border);
    background: rgba(255, 255, 255, 0.018);
}}
.model-picker-side {{
    height: 100%;
    padding: 0.95rem 1.15rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.25rem;
}}
.model-picker-eyebrow {{
    margin: 0;
    font-size: var(--kkp-text-xs);
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #71717a;
}}
.model-picker-lead {{
    margin: 0;
    font-size: var(--kkp-text-sm);
    line-height: 1.4;
    color: #d4d4d8;
}}
#kkp-model-picker-controls {{
    flex: 1 1 auto !important;
    min-width: 0 !important;
    padding: 0.85rem 1.15rem !important;
    gap: 0.55rem !important;
    justify-content: center !important;
}}
#kkp-model-select .wrap,
#kkp-model-select fieldset {{
    display: flex !important;
    flex-direction: row !important;
    gap: 0.4rem !important;
    width: 100% !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-width: 0 !important;
}}
#kkp-model-select label {{
    flex: 1 1 0 !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0.55rem 0.65rem !important;
    border: 1px solid var(--kkp-border) !important;
    border-radius: var(--kkp-radius-sm) !important;
    background: transparent !important;
    color: #8b8b93 !important;
    font-size: var(--kkp-text-sm) !important;
    font-weight: 500 !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: border-color 0.15s ease, color 0.15s ease, background 0.15s ease !important;
}}
#kkp-model-select label:hover {{
    border-color: var(--kkp-border-strong) !important;
    color: #c4c4cc !important;
}}
#kkp-model-select label:has(input:checked) {{
    border-color: rgba(255, 255, 255, 0.22) !important;
    background: rgba(255, 255, 255, 0.06) !important;
    color: #f4f4f5 !important;
}}
#kkp-model-select input {{
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}}
#kkp-model-note .block,
#kkp-model-note .html-container,
#kkp-model-note .wrap {{
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}}
.model-picker-note {{
    margin: 0;
    padding-left: 0.15rem;
    font-size: var(--kkp-text-xs);
    line-height: 1.45;
    color: #71717a;
}}
.bench-panel {{
    display: block;
    width: 100%;
    box-sizing: border-box;
    margin-top: 0.85rem;
    padding: 1.15rem 1.25rem 1.2rem;
    border: 1px solid var(--kkp-border);
    border-radius: var(--kkp-radius);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.02) 0%, transparent 100%);
}}
.bench-head {{
    margin-bottom: 1rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--kkp-border);
}}
.bench-eyebrow {{
    display: block;
    margin-bottom: 0.3rem;
    font-size: var(--kkp-text-xs);
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #71717a;
}}
.bench-title {{
    margin: 0;
    font-size: var(--kkp-text-lg);
    font-weight: 500;
    letter-spacing: -0.02em;
    color: #f4f4f5;
}}
.bench-compare {{
    display: flex;
    flex-direction: column;
    gap: 0;
    width: 100%;
}}
.bench-compare-head,
.bench-compare-row {{
    display: grid;
    grid-template-columns: minmax(5.5rem, 7rem) 1fr 1fr;
    gap: 0.65rem;
    align-items: baseline;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--kkp-border);
    width: 100%;
    box-sizing: border-box;
}}
.bench-compare-head {{
    padding-top: 0;
    padding-bottom: 0.65rem;
}}
.bench-compare-row:last-child {{
    border-bottom: none;
    padding-bottom: 0;
}}
.bench-compare-head span {{
    font-size: var(--kkp-text-xs);
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #71717a;
}}
.bench-compare-head span:first-child {{
    visibility: hidden;
}}
.bench-metric {{
    font-size: var(--kkp-text-sm);
    color: #a1a1aa;
}}
.bench-val {{
    font-family: var(--kkp-font-mono);
    font-size: var(--kkp-text-base);
    font-weight: 500;
    color: #9ca3af;
    text-align: left;
}}
.bench-val--lead {{
    color: #f4f4f5;
}}
.bench-verdict {{
    margin: 0.95rem 0 0;
    padding-top: 0.85rem;
    border-top: 1px solid var(--kkp-border);
    font-size: var(--kkp-text-sm);
    line-height: 1.5;
    color: #a1a1aa;
}}
.bench-verdict strong {{
    color: #e4e4e7;
    font-weight: 500;
}}
.bench-verdict em {{
    font-style: normal;
    font-family: var(--kkp-font-mono);
    color: #d4d4d8;
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
    """Metadata shown under the model picker."""

    key: str
    profile_id: str
    display_name: str
    architecture: str
    epoch: int
    input_size: int
    dataset: str
    test_accuracy: float | None
    f1: float | None
    roc_auc: float | None
    train_samples: str
    checkpoint: str
    description: str

    def profile_info_html(self) -> str:
        acc = f"{self.test_accuracy * 100:.1f}%" if self.test_accuracy is not None else "—"
        parts = (
            f'<span class="model-info-arch">{self.architecture}</span>',
            f"<span>ep. {self.epoch}</span>",
            f"<span>{self.input_size}px</span>",
            f"<span>test acc {acc}</span>",
            f"<span>{self.dataset}</span>",
            f"<span>{self.train_samples} train</span>",
        )
        row = '<span class="model-info-sep">·</span>'.join(parts)
        return f"""
        <div class="model-info" id="model-info-{self.profile_id}">
            <div class="model-info-row">{row}</div>
            <div class="model-info-path">{self.checkpoint}</div>
        </div>
        """


@dataclass
class LoadedModel:
    key: str
    model: nn.Module
    transform: Compose
    device: torch.device
    class_names: tuple[str, ...]
    profile: ModelProfile


class DemoEngine:
    def __init__(
        self,
        models: dict[str, LoadedModel],
        dataset_samples: list[tuple[Path, str, str]],
    ):
        self.models = models
        self.dataset_samples = dataset_samples
        self.default_key = max(
            models,
            key=lambda key: (
                models[key].profile.test_accuracy or 0.0,
                models[key].profile.f1 or 0.0,
            ),
        )

    @property
    def choices(self) -> list[tuple[str, str]]:
        return [(MODEL_LABELS_UA[key], key) for key in self.models]

    def _run_prediction(
        self,
        model_key: str,
        image: Image.Image,
        source: dict[str, str] | None = None,
    ) -> tuple[str, str]:
        loaded = self.models[model_key]
        label, confidences = predict_image(
            loaded.model,
            image,
            loaded.transform,
            loaded.device,
            loaded.class_names,
        )
        winner_score = confidences[label] * 100
        title = VERDICT_REAL if label == "real" else VERDICT_AI
        results = _results_html(title, winner_score, confidences)
        if source is None:
            return results, EMPTY_SOURCE
        source_html = _source_html(source, predicted=label)
        return _combine_results(results, source_html), EMPTY_SOURCE

    def predict(self, image: Image.Image | None, model_key: str) -> tuple[str, str]:
        model_info = self.models[model_key].profile.profile_info_html()
        if image is None:
            return model_info, EMPTY_RESULTS
        results, _source = self._run_prediction(model_key, image)
        return model_info, results

    def random_from_dataset(self, model_key: str) -> tuple[Image.Image | None, str, str]:
        model_info = self.models[model_key].profile.profile_info_html()
        if not self.dataset_samples:
            msg = _empty_results_html("Датасет не знайдено. Запустіть `make download-hires-hf`.")
            return None, model_info, msg

        path, split, label_dir = random.choice(self.dataset_samples)
        image = Image.open(path).convert("RGB")
        source = {
            "split": split,
            "label_dir": label_dir,
            "expected_class": _label_dir_to_class(label_dir),
            "relative_path": f"{split}/{label_dir}/{path.name}",
        }
        results, _source = self._run_prediction(model_key, image, source=source)
        return image, model_info, results

    def comparison_html(self) -> str:
        return _comparison_panel_html([loaded.profile for loaded in self.models.values()])


def _load_metrics_json(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _count_train_samples(data_dir: Path) -> str:
    train_real = data_dir / "train" / "REAL"
    train_fake = data_dir / "train" / "FAKE"
    if not train_real.is_dir() or not train_fake.is_dir():
        return "—"
    count = len(list(train_real.glob("*.jpg"))) + len(list(train_fake.glob("*.jpg")))
    return str(count) if count else "—"


def build_model_profile(
    key: str,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    *,
    metrics: dict[str, Any] | None = None,
    dataset_name: str = "Parveshiiii/AI-vs-Real",
) -> ModelProfile:
    data_dir = Path(config["paths"]["data_dir"])
    project = config.get("project", {}).get("name", "kkp")
    model_name = config["model"]["name"]
    return ModelProfile(
        key=key,
        profile_id=f"{project}-{model_name}",
        display_name=f"{project} / {model_name}",
        architecture=model_name.upper(),
        epoch=int(checkpoint["epoch"]),
        input_size=int(config["training"]["image_size"]),
        dataset=dataset_name,
        test_accuracy=float(metrics["accuracy"]) if metrics else None,
        f1=float(metrics["f1"]) if metrics else None,
        roc_auc=float(metrics["roc_auc"]) if metrics else None,
        train_samples=_count_train_samples(data_dir),
        checkpoint=str(checkpoint_path),
        description=MODEL_DESCRIPTIONS.get(key, ""),
    )


def _pct(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _model_note_html(model_key: str) -> str:
    description = MODEL_DESCRIPTIONS.get(model_key, "")
    return f'<p class="model-picker-note">{description}</p>'


def _compare_value_html(value: float | None, *, lead: bool) -> str:
    lead_class = " bench-val--lead" if lead else ""
    return f'<span class="bench-val{lead_class}">{_pct(value)}</span>'


def _comparison_panel_html(profiles: list[ModelProfile]) -> str:
    if len(profiles) < 2:
        profile = profiles[0]
        profile_name = MODEL_LABELS_UA.get(profile.key, profile.architecture)
        return f"""
        <div class="bench-panel">
            <header class="bench-head">
                <span class="bench-eyebrow">Архітектура</span>
                <h2 class="bench-title">{profile_name}</h2>
            </header>
            <p class="bench-verdict">{profile.description}</p>
        </div>
        """

    ranked = sorted(
        profiles,
        key=lambda item: (item.test_accuracy or 0.0, item.f1 or 0.0),
        reverse=True,
    )
    best = ranked[0]
    best_label = MODEL_LABELS_UA.get(best.key, best.architecture)
    metric_rows = (
        ("Accuracy", "test_accuracy"),
        ("F1", "f1"),
        ("ROC-AUC", "roc_auc"),
    )

    headers = "".join(
        f"<span>{MODEL_LABELS_UA.get(profile.key, profile.architecture)}</span>"
        for profile in profiles
    )
    rows: list[str] = []
    for label, attr in metric_rows:
        values = [getattr(profile, attr) for profile in profiles]
        best_value = max((value or 0.0 for value in values), default=0.0)
        cells = "".join(
            _compare_value_html(
                value,
                lead=value is not None and value == best_value and best_value > 0,
            )
            for value in values
        )
        rows.append(
            f"""
            <div class="bench-compare-row">
                <span class="bench-metric">{label}</span>
                {cells}
            </div>
            """
        )

    acc_delta = None
    if best.test_accuracy is not None and ranked[-1].test_accuracy is not None:
        acc_delta = (best.test_accuracy - ranked[-1].test_accuracy) * 100

    delta_text = (
        f" на <em>+{acc_delta:.1f} pp</em> accuracy"
        if acc_delta is not None and acc_delta > 0
        else ""
    )
    verdict = (
        f"<strong>{best_label}</strong> стабільно краща на test set{delta_text}. "
        f"Обидві навчені на одному hi-res split (512px)."
    )

    return f"""
    <div class="bench-shell">
        <div class="bench-panel">
            <header class="bench-head">
                <span class="bench-eyebrow">Test set · метрики</span>
                <h2 class="bench-title">Порівняння моделей</h2>
            </header>
            <div class="bench-compare">
                <div class="bench-compare-head">
                    <span></span>
                    {headers}
                </div>
                {"".join(rows)}
            </div>
            <p class="bench-verdict">{verdict}</p>
        </div>
    </div>
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KKP Gradio demo")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        help="Model config (repeatable; default: ResNet18 + EfficientNet-B0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override checkpoint for the first --config only",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    return parser.parse_args()


def _resolve_model_configs(args: argparse.Namespace) -> dict[str, Path]:
    if args.configs:
        paths = args.configs
        keys = [load_config(path)["model"]["name"] for path in paths]
        return dict(zip(keys, paths, strict=True))
    return DEFAULT_MODEL_CONFIGS.copy()


def build_demo_engine(
    model_configs: dict[str, Path],
    *,
    checkpoint_override: Path | None = None,
) -> DemoEngine:
    loaded: dict[str, LoadedModel] = {}
    data_dir: Path | None = None

    for index, (key, config_path) in enumerate(model_configs.items()):
        config = load_config(config_path)
        training = config["training"]
        output_dir = Path(config["paths"]["output_dir"])
        if data_dir is None:
            data_dir = Path(config["paths"]["data_dir"])
        resolved_checkpoint = output_dir / "checkpoints" / "best.pth"
        if index == 0 and checkpoint_override is not None:
            resolved_checkpoint = checkpoint_override

        if not resolved_checkpoint.is_file():
            msg = f"Checkpoint not found for {key}: {resolved_checkpoint}"
            raise FileNotFoundError(msg)

        device = torch.device(training["device"])
        model, checkpoint = load_checkpoint(resolved_checkpoint, device)
        metrics = _load_metrics_json(output_dir)
        profile = build_model_profile(
            key,
            config,
            checkpoint,
            resolved_checkpoint,
            metrics=metrics,
        )
        loaded[key] = LoadedModel(
            key=key,
            model=model,
            transform=get_transforms(training["image_size"], train=False),
            device=device,
            class_names=tuple(config["data"]["classes"]),
            profile=profile,
        )
        logger.info(
            "Loaded %s (epoch %d) from %s",
            config["model"]["name"],
            checkpoint["epoch"],
            resolved_checkpoint,
        )

    samples = _discover_dataset_samples(data_dir) if data_dir else []
    return DemoEngine(loaded, samples)


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


def _combine_results(results: str, source: str) -> str:
    if not source:
        return results
    return f'<div class="results-stack">{results}{source}</div>'


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


def create_demo(engine: DemoEngine) -> gr.Blocks:
    default_key = engine.default_key
    default_profile = engine.models[default_key].profile

    with gr.Blocks(
        title="KKP — AI vs Real",
        theme=_dark_theme(),
        css=CUSTOM_CSS,
        fill_width=True,
    ) as demo:
        with gr.Column(elem_id="kkp-page"):
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

            with gr.Column(elem_id="kkp-model-section"):
                with gr.Row(elem_id="kkp-model-picker"):
                    gr.HTML(
                        """
                        <div class="model-picker-side">
                            <p class="model-picker-eyebrow">Модель</p>
                            <p class="model-picker-lead">Що аналізує зображення</p>
                        </div>
                        """,
                        elem_id="kkp-model-picker-label",
                    )
                    with gr.Column(elem_id="kkp-model-picker-controls"):
                        model_select = gr.Radio(
                            choices=engine.choices,
                            value=default_key,
                            show_label=False,
                            container=False,
                            elem_id="kkp-model-select",
                        )
                        model_note = gr.HTML(
                            value=_model_note_html(default_key),
                            elem_id="kkp-model-note",
                        )
                model_info = gr.HTML(
                    default_profile.profile_info_html(),
                    elem_id="kkp-model-info",
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

            with gr.Column(elem_id="kkp-compare-section"):
                gr.HTML(
                    engine.comparison_html(),
                    elem_id="kkp-model-compare",
                    apply_default_css=False,
                    container=True,
                    padding=False,
                )

        outputs = [model_info, results_output]
        all_outputs = [image_input, *outputs]

        def on_model_change(
            image: Image.Image | None,
            model_key: str,
        ) -> tuple[str, str, str]:
            info, results = engine.predict(image, model_key)
            return info, results, _model_note_html(model_key)

        analyze_btn.click(
            engine.predict,
            inputs=[image_input, model_select],
            outputs=outputs,
            show_progress="hidden",
        )
        random_btn.click(
            engine.random_from_dataset,
            inputs=[model_select],
            outputs=all_outputs,
            show_progress="hidden",
        )
        model_select.change(
            on_model_change,
            inputs=[image_input, model_select],
            outputs=[*outputs, model_note],
            show_progress="hidden",
        )
        clear_btn.click(
            lambda _image, model_key: (None, *engine.predict(None, model_key)),
            inputs=[image_input, model_select],
            outputs=[image_input, model_info, results_output],
            show_progress="hidden",
        )

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

    model_configs = _resolve_model_configs(args)
    engine = build_demo_engine(model_configs, checkpoint_override=args.checkpoint)
    demo = create_demo(engine)
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
