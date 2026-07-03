"""API smoke tests with a lightweight stub engine."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from kkp.api import create_app
from kkp.demo_engine import DemoEngine, image_to_data_url


@pytest.fixture
def stub_engine() -> DemoEngine:
    engine = MagicMock(spec=DemoEngine)
    engine.models = {"resnet18": object()}
    engine.list_models.return_value = {
        "default": "resnet18",
        "models": [{"key": "resnet18", "label": "ResNet18"}],
    }
    engine.pick_random_sample.return_value = {
        "image": image_to_data_url(Image.new("RGB", (8, 8), color=(10, 20, 30))),
        "source": {
            "split": "test",
            "label_dir": "REAL",
            "expected_class": "real",
            "relative_path": "test/REAL/sample.jpg",
            "split_label": "тестова вибірка",
            "label_dir_label": "реальне фото (REAL)",
            "expected_label": "Реальне фото",
            "split_note": "note",
        },
    }
    engine.analyze.return_value = {
        "label": "real",
        "title": "Реальне фото",
        "confidence_pct": 91.0,
        "confidences": [
            {"key": "real", "label": "Реальне фото", "pct": 91.0},
            {"key": "ai_generated", "label": "AI-генерація", "pct": 9.0},
        ],
        "heatmap": None,
        "source": None,
    }
    return engine


@pytest.fixture
def client(stub_engine: DemoEngine) -> TestClient:
    return TestClient(create_app(engine=stub_engine))


def test_list_models(client: TestClient) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload["default"] == "resnet18"
    assert payload["models"][0]["label"] == "ResNet18"


def test_random_sample(client: TestClient) -> None:
    response = client.get("/api/random")
    assert response.status_code == 200
    payload = response.json()
    assert payload["image"].startswith("data:image/jpeg;base64,")
    assert payload["source"]["relative_path"] == "test/REAL/sample.jpg"


def test_analyze_upload(client: TestClient, stub_engine: DemoEngine) -> None:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), color=(255, 0, 0)).save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/api/analyze",
        data={"model": "resnet18"},
        files={"image": ("sample.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Реальне фото"
    stub_engine.analyze.assert_called_once()


def test_index_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Детекція AI-зображень" in response.text
