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
src/kkp/
  train.py, evaluate.py, compare.py, demo.py
outputs/
  ai_generated/                  — ResNet18 checkpoints + metrics.json
  ai_generated_efficientnet/     — EfficientNet checkpoints + metrics.json
  comparison/                    — графіки порівняння моделей
```

Параметри експериментів — у `configs/*.yaml`. Змінні з `.env` (`DATA_DIR`, `DEVICE`) підставляються автоматично.

## Результати (test set)

| Модель | Accuracy | F1 | ROC-AUC |
|--------|----------|-----|---------|
| ResNet18 | 93.6% | 0.938 | 0.992 |
| EfficientNet-B0 | **98.5%** | **0.986** | **1.000** |

Детальні метрики та графіки: `make compare` → `outputs/comparison/`.

## Запуск

### Локально

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

make install
cp .env.example .env

make test
make lint
```

### Дані

```bash
make download-hires-hf           # повний hi-res датасет з Hugging Face
make download-hires-hf-smoke       # міні-версія для перевірки
```

### Навчання та оцінка

```bash
make train                       # ResNet18
make train-efficientnet          # EfficientNet-B0
make evaluate-all                # метрики на test → metrics.json
make compare                     # графіки порівняння
```

### Демо

```bash
make demo                        # http://127.0.0.1:7860
```

Демо завантажує обидві моделі, дозволяє перемикати архітектуру, показує порівняння метрик і перевіряти завантажені або випадкові зображення з датасету.

### Docker

```bash
cp .env.example .env
docker compose build

docker compose run --rm dev
docker compose run --rm test
docker compose run --rm lint
```

## CI

При push/PR: Ruff, pytest, збірка Docker-образу.

## Виконавець

Ткач Леонід, група ПЗПІ-23-5
