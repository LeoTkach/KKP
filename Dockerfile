# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=600

WORKDIR /app

# Default: CPU wheels (Mac Docker / CI). Linux + NVIDIA GPU:
#   PYTORCH_INDEX=https://download.pytorch.org/whl/cu124 docker compose build
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY docker/constraints.txt ./docker/constraints.txt
COPY configs/ ./configs/

RUN pip install --upgrade pip

# Torch layer must not depend on src/ — otherwise every demo edit re-downloads ~90MB.
RUN --mount=type=cache,target=/root/.cache/pip \
    for attempt in 1 2 3 4 5; do \
        pip install --retries 10 \
            -c docker/constraints.txt \
            torch==2.5.1 torchvision==0.20.1 \
            --index-url "${PYTORCH_INDEX}" \
        && exit 0; \
        echo "PyTorch install attempt ${attempt} failed, retrying..."; \
        sleep 20; \
    done; \
    exit 1

# Stub package so dependency install is cached separately from application source.
RUN mkdir -p src/kkp && touch src/kkp/__init__.py

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 -c docker/constraints.txt . hatchling

COPY src/ ./src/

# Reinstall only kkp (no pandas/matplotlib re-download) when src/ changes.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 --no-deps --no-build-isolation -c docker/constraints.txt .

# --- production / training runtime ---
FROM base AS runtime

ENV DATA_DIR=/app/data \
    OUTPUT_DIR=/app/outputs

CMD ["python", "-m", "kkp.train", "--config", "configs/ai_generated.yaml"]

# --- development: lint, tests, jupyter ---
FROM base AS dev

COPY tests/ ./tests/
COPY .pre-commit-config.yaml ./

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 -c docker/constraints.txt -e ".[dev]"

CMD ["bash"]

# --- Web UI: FastAPI + static frontend (make demo) ---
FROM base AS demo

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 10 "fastapi>=0.115" "uvicorn[standard]>=0.32" "python-multipart>=0.0.9"

COPY frontend/ ./frontend/

ENV DEVICE=cpu

EXPOSE 7860

CMD ["python", "-m", "kkp.api", "--host", "0.0.0.0", "--port", "7860"]
