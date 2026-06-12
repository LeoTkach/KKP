PYTHON ?= .venv/bin/python

.PHONY: install lint test train demo download-hires download-hires-hf train-cifake docker-build docker-dev docker-test docker-lint docker-jupyter docs compare evaluate evaluate-efficientnet evaluate-all train-efficientnet demo-screenshots experiment-audit verify-results export-artifacts ablation ablation-train gradcam

install:
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

lint:
	ruff check src tests
	ruff format --check src tests

test:
	$(PYTHON) -m pytest -v --cov=kkp --cov-report=term-missing

train:
	$(PYTHON) -m kkp.train

train-efficientnet:
	$(PYTHON) -m kkp.train --config configs/ai_generated_efficientnet.yaml

evaluate:
	$(PYTHON) -m kkp.evaluate --config configs/ai_generated.yaml

evaluate-efficientnet:
	$(PYTHON) -m kkp.evaluate --config configs/ai_generated_efficientnet.yaml

evaluate-all:
	$(PYTHON) -m kkp.evaluate --config configs/ai_generated.yaml
	$(PYTHON) -m kkp.evaluate --config configs/ai_generated_efficientnet.yaml

compare:
	$(PYTHON) -m kkp.compare

train-cifake:
	$(PYTHON) -m kkp.train --config configs/ai_generated_cifake.yaml

download-hires:
	$(PYTHON) scripts/download_ai_hires.py

download-hires-smoke:
	$(PYTHON) scripts/download_ai_hires.py --train-per-class 4 --val-per-class 2 --test-per-class 2

download-hires-hf:
	$(PYTHON) scripts/download_ai_hires.py --profile hf

download-hires-hf-smoke:
	$(PYTHON) scripts/download_ai_hires.py --profile hf --train-per-class 4 --val-per-class 2 --test-per-class 2

demo:
	$(PYTHON) -m kkp.demo

docs:
	$(PYTHON) scripts/generate_kkp_documents.py

experiment-audit:
	$(PYTHON) scripts/experiment_audit.py export

ablation:
	$(PYTHON) scripts/run_ablation.py

ablation-train:
	$(PYTHON) scripts/run_ablation.py --force-train

verify-results:
	$(PYTHON) scripts/experiment_audit.py verify

export-artifacts:
	$(PYTHON) scripts/export_report_artifacts.py

gradcam:
	$(PYTHON) scripts/generate_gradcam.py

demo-screenshots:
	$(PYTHON) scripts/capture_demo_screenshots.py --connect-only

docker-build:
	docker compose build

docker-dev:
	docker compose run --rm dev

docker-test:
	docker compose run --rm test

docker-lint:
	docker compose run --rm lint

docker-jupyter:
	docker compose up jupyter
