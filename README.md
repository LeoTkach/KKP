# KKP — Комплексний курсовий проєкт

**KKP** (*Комплексний курсовий проєкт*) — курсовий проєкт з **computer vision**: автоматичне виявлення **AI-генерованих зображень** (real vs fake).

Задача формулюється як **бінарна класифікація**: `real` vs `ai_generated`.

## Мета

- Зібрати та підготувати hi-res датасет AI vs Real.
- Навчити та порівняти дві CNN-моделі (ResNet18, EfficientNet-B0) з transfer learning.
- Оцінити якість (accuracy, precision, recall, F1, ROC-AUC) і оформити результати у звіті.
- Надати інтерактивне Gradio-демо для перевірки зображень.

## Підхід

```
Зображення → препроцесинг (224px) → CNN (transfer learning) → real / ai_generated
```

## Стек

- Python 3.11+, PyTorch, torchvision
- scikit-learn — метрики класифікації
- Gradio — веб-демо
- Ruff, pytest, pre-commit
- Docker, GitHub Actions

## Структура проєкту

```
configs/
  ai_generated.yaml              — ResNet18, hi-res датасет
  ai_generated_efficientnet.yaml — EfficientNet-B0
  ai_generated_cifake.yaml       — legacy CIFAKE baseline (32×32)
data/
  ai_hires/                      — Parveshiiii/AI-vs-Real (512px, train/val/test)
docs/
  artifacts/                     — зафіксовані метрики, графіки, misclassifications (в git)
  *.docx / *.pptx                — звіт і презентація ККП
src/kkp/
  train.py, evaluate.py, compare.py, demo.py
outputs/                         — checkpoints (.pth) локально, не в git
  ai_generated/
  ai_generated_efficientnet/
  comparison/
scripts/
  experiment_audit.py            — export/verify frozen results
  generate_kkp_documents.py      — генерація звіту
```

Параметри експериментів — у `configs/*.yaml`. Змінні з `.env` (`DATA_DIR`, `DEVICE`) підставляються автоматично.

## Результати (test set, N=204)

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC | Помилок |
|--------|----------|-----------|--------|-----|---------|---------|
| ResNet18 | 93.6% | 0.908 | 0.971 | 0.938 | 0.992 | 13 |
| EfficientNet-B0 | **98.5%** | **0.971** | **1.000** | **0.986** | **1.000** | 3 |

**Out-of-domain** (`data/real_world/`, N=30): EfficientNet-B0 — accuracy **80.0%**, F1 0.800 (domain shift).

Повний звіт з bootstrap CI, confusion matrix і списком помилок:

→ [`docs/artifacts/EXPERIMENT_SUMMARY.md`](docs/artifacts/EXPERIMENT_SUMMARY.md)

Графіки: `docs/artifacts/*.png` або `make compare` → `outputs/comparison/`.

## Відтворюваність без повторного навчання

Checkpoint-и (`.pth`) **не комітяться** — занадто великі. У git зберігаються **метрики та графіки** в `docs/artifacts/`.

```bash
# Якщо checkpoint-и вже є локально (після make train):
make evaluate-all        # ~хвилина, без GPU-тренування
make verify-results      # порівняти з docs/artifacts/*.json

# Оновити всі артефакти + summary + misclassifications:
make experiment-audit
```

`make verify-results` проганяє inference на test set і перевіряє, що числа збігаються з закоміченими `docs/artifacts/*_test_metrics.json` (tol 1e-4).

## Запуск

### Локально

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev,demo,docs]"
pre-commit install
pre-commit install --hook-type commit-msg
cp .env.example .env

make test
make lint
```

### Дані

```bash
make download-hires-hf           # повний hi-res датасет з Hugging Face
make download-hires-hf-smoke     # міні-версія для перевірки
```

### Навчання та оцінка

```bash
make train                       # ResNet18
make train-efficientnet          # EfficientNet-B0
make evaluate-all                # метрики на test → metrics.json
make compare                     # графіки порівняння
make experiment-audit            # docs/artifacts/ + EXPERIMENT_SUMMARY.md
```

### Звіт ККП

```bash
make docs                        # DOCX + PPTX у docs/
```

Потрібні: Pages (macOS) або LibreOffice для номерів змісту; `pip install -e ".[docs]"` (pymupdf).

### Демо

```bash
make demo                        # http://127.0.0.1:7860
make demo-screenshots            # знімки UI для звіту (потрібен playwright)
```

**Веб-інтерфейс системи** (`make demo` → http://127.0.0.1:7860/):

1. Оберіть модель: **ResNet18** або **EfficientNet-B0**
2. Завантажте зображення (JPEG/PNG/WebP) або натисніть **«Випадкове»**
3. Натисніть **«Перевірити»** — verdict і confidence
4. У нижній панелі — порівняння метрик обох моделей

### Docker

```bash
cp .env.example .env

# Веб-інтерфейс (Gradio) — http://localhost:7860
docker compose up --build demo

# CI / розробка (окремі профілі)
docker compose --profile ci run --rm test
docker compose --profile ci run --rm lint
docker compose --profile dev run --rm dev
docker compose --profile jupyter up jupyter
docker compose --profile train up app   # потрібен ./data
```

Перед demo потрібні checkpoint-и в `outputs/` (локально: `make train`).

На Mac (Apple Silicon) за замовчуванням ставиться **CPU-only PyTorch** (~100 MB замість CUDA-збірки >400 MB).
Якщо збірка знову впаде через мережу — просто перезапустіть `docker compose build`.

Linux + NVIDIA GPU (опційно):

```bash
PYTORCH_INDEX=https://download.pytorch.org/whl/cu124 docker compose build demo
```

## CI

При push/PR: Ruff, pytest, збірка Docker-образу.

## Git / GitHub

- Основна гілка розробки: **`dev`**
- Default на GitHub: **`main`** (може відставати від `dev`)
- Checkpoint-и та датасет — локально; у репо — код, тести, `docs/artifacts/`

## Виконавець

Ткач Леонід, група ПЗПІ-23-5
