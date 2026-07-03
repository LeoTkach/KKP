"""Demo service: model loading, inference, dataset samples (no UI)."""

from __future__ import annotations

import base64
import json
import logging
import random
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose

from kkp.config import load_config
from kkp.data.transforms import get_transforms
from kkp.gradcam import explain_image
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

LABELS_UA: dict[str, str] = {
    "real": "Реальне фото",
    "ai_generated": "AI-генерація",
}

VERDICT_REAL = "Реальне фото"
VERDICT_AI = "AI-генерація"

DATASET_LABEL_UA = "Parveshiiii/AI-vs-Real"

SPLIT_LABELS_UA: dict[str, str] = {
    "train": "навчальна вибірка",
    "val": "валідаційна вибірка",
    "test": "тестова вибірка",
}

LABEL_DIR_UA: dict[str, str] = {
    "REAL": "реальне фото (REAL)",
    "FAKE": "AI-генерація (FAKE)",
}

SPLIT_NOTES_UA: dict[str, str] = {
    "train": (
        "Зображення з train модель уже бачила під час навчання "
        "(можливо з іншими аугментаціями) — збіг тут не доводить узагальнення."
    ),
    "val": (
        "Валідаційна вибірка використовувалась для контролю епох, "
        "не для фінальної оцінки якості на test."
    ),
    "test": (
        "Тестова вибірка — модель не бачила ці зображення під час навчання; "
        "найчесніша перевірка якості."
    ),
}


@dataclass(frozen=True)
class ModelProfile:
    key: str
    input_size: int
    test_accuracy: float | None = None
    f1: float | None = None


@dataclass
class LoadedModel:
    key: str
    model: nn.Module
    transform: Compose
    device: torch.device
    class_names: tuple[str, ...]
    profile: ModelProfile


def image_to_data_url(image: Image.Image, *, quality: int = 88) -> str:
    buffer = BytesIO()
    rgb = image.convert("RGB")
    rgb.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_heatmap_images(
    base_image: Image.Image,
    overlay_image: Image.Image,
    *,
    max_side: int = 960,
) -> tuple[Image.Image, Image.Image]:
    display_base = base_image.convert("RGB")
    display_overlay = overlay_image.convert("RGB")
    display_base.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    display_overlay = display_overlay.resize(display_base.size, Image.Resampling.LANCZOS)
    return display_base, display_overlay


def enrich_source(source: dict[str, str]) -> dict[str, str]:
    split = source["split"]
    label_dir = source["label_dir"]
    expected = source["expected_class"]
    return {
        **source,
        "dataset_label": DATASET_LABEL_UA,
        "split_label": SPLIT_LABELS_UA.get(split, split),
        "label_dir_label": LABEL_DIR_UA.get(label_dir, label_dir),
        "expected_label": LABELS_UA[expected],
        "split_note": SPLIT_NOTES_UA.get(split, ""),
    }


def source_with_prediction(source: dict[str, str], predicted: str) -> dict[str, Any]:
    enriched = enrich_source(source)
    match = predicted == source["expected_class"]
    return {
        **enriched,
        "match": match,
        "match_label": "Збіг" if match else "Помилка",
    }


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

    def list_models(self) -> dict[str, Any]:
        return {
            "default": self.default_key,
            "models": [{"key": key, "label": MODEL_LABELS_UA[key]} for key in self.models],
        }

    def pick_random_sample(self) -> dict[str, Any]:
        if not self.dataset_samples:
            return {
                "error": "Датасет не знайдено. Запустіть `make download-hires-hf`.",
            }

        path, split, label_dir = random.choice(self.dataset_samples)
        image = Image.open(path).convert("RGB")
        source = _make_source(path, split, label_dir)
        return {
            "image": image_to_data_url(image),
            "source": enrich_source(source),
        }

    def analyze(
        self,
        image: Image.Image,
        model_key: str,
        source: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if model_key not in self.models:
            msg = f"Unknown model: {model_key}"
            raise ValueError(msg)

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

        result: dict[str, Any] = {
            "label": label,
            "title": title,
            "confidence_pct": round(winner_score, 1),
            "confidences": [
                {
                    "key": name,
                    "label": LABELS_UA.get(name, name),
                    "pct": round(score * 100, 1),
                }
                for name, score in confidences.items()
            ],
            "heatmap": None,
            "source": None,
        }

        if source is not None:
            result["source"] = source_with_prediction(source, label)

        heatmap_pair = self._build_heatmap(loaded, image, label)
        if heatmap_pair is not None:
            original, overlay = heatmap_pair
            result["heatmap"] = {
                "original": image_to_data_url(original),
                "overlay": image_to_data_url(overlay),
            }

        return result

    def _build_heatmap(
        self,
        loaded: LoadedModel,
        image: Image.Image,
        label: str,
    ) -> tuple[Image.Image, Image.Image] | None:
        try:
            class_index = loaded.class_names.index(label)
            _original, overlay = explain_image(
                loaded.model,
                loaded.key,
                image,
                loaded.transform,
                loaded.device,
                target_class=class_index,
                image_size=loaded.profile.input_size,
            )
            base = image.convert("RGB")
            overlay_img = Image.fromarray(overlay)
            return prepare_heatmap_images(base, overlay_img)
        except Exception:
            logger.exception("Grad-CAM failed for %s", loaded.key)
            return None


def _load_metrics_json(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_profile(
    key: str,
    config: dict[str, Any],
    *,
    metrics: dict[str, Any] | None = None,
) -> ModelProfile:
    return ModelProfile(
        key=key,
        input_size=int(config["training"]["image_size"]),
        test_accuracy=float(metrics["accuracy"]) if metrics else None,
        f1=float(metrics["f1"]) if metrics else None,
    )


def build_demo_engine(
    model_configs: dict[str, Path] | None = None,
    *,
    checkpoint_override: Path | None = None,
) -> DemoEngine:
    configs = model_configs or DEFAULT_MODEL_CONFIGS.copy()
    loaded: dict[str, LoadedModel] = {}
    data_dir: Path | None = None

    for index, (key, config_path) in enumerate(configs.items()):
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
        profile = build_model_profile(key, config, metrics=metrics)
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


def _make_source(path: Path, split: str, label_dir: str) -> dict[str, str]:
    return {
        "split": split,
        "label_dir": label_dir,
        "expected_class": _label_dir_to_class(label_dir),
        "relative_path": f"{split}/{label_dir}/{path.name}",
    }


def parse_source_json(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        msg = "source must be a JSON object"
        raise ValueError(msg)
    return {str(key): str(value) for key, value in data.items()}
