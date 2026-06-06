# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY configs/ ./configs/
COPY src/ ./src/

RUN pip install --upgrade pip && pip install .

# --- production / training runtime ---
FROM base AS runtime

ENV DATA_DIR=/app/data \
    OUTPUT_DIR=/app/outputs

CMD ["python", "-m", "kkp.train", "--config", "configs/default.yaml"]

# --- development: lint, tests, jupyter ---
FROM base AS dev

COPY tests/ ./tests/
COPY .pre-commit-config.yaml ./

RUN pip install -e ".[dev]"

CMD ["bash"]
