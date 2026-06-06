.PHONY: install lint test train docker-build docker-dev docker-test docker-lint docker-jupyter

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
