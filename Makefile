.PHONY: install lint test train demo download-hires train-cifake docker-build docker-dev docker-test docker-lint docker-jupyter

install:
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -v --cov=kkp --cov-report=term-missing

train:
	python -m kkp.train

train-cifake:
	python -m kkp.train --config configs/ai_generated_cifake.yaml

download-hires:
	python scripts/download_ai_hires.py

download-hires-smoke:
	python scripts/download_ai_hires.py --train-per-class 4 --val-per-class 2 --test-per-class 2

demo:
	python -m kkp.demo

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
