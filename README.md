# KKP — Комплексний курсовий проєкт

**KKP** (*Комплексний курсовий проєкт*) — курсовий проєкт з **computer vision**: автоматичне виявлення неавтентичних зображень двох типів:

1. **AI-генеровані зображення** — відрізнити реальне фото від зображення, створеного нейромережею (Stable Diffusion, DALL·E, Midjourney тощо).
2. **Відредаговані зображення** — виявити маніпуляції в оригінальному фото (кроп, вставка фрагментів, ретуш, заміна об'єктів).

Обидві задачі формулюються як **бінарна класифікація**: `real` vs `fake` / `original` vs `edited`.

## Мета

- Зібрати та підготувати дані для обох задач.
- Навчити baseline-модель (transfer learning, наприклад ResNet/EfficientNet).
- Порівняти метрики якості (accuracy, precision, recall, F1, ROC-AUC).
- Проаналізувати, наскільки ознаки «AI-фейку» та «редагування» схожі або різняться між собою.
- Оформити результати у курсовій роботі з відтворюваним кодом.

## Підхід

```
Зображення → препроцесинг → CNN (transfer learning) → real / fake
```

На першому етапі — дві незалежні моделі (по одній на задачу).

## Стек

- Python 3.11+, PyTorch, torchvision
- scikit-learn, pandas — метрики та аналіз
- Ruff, pytest, pre-commit
- Docker, GitHub Actions

## Структура проєкту

```
configs/
  default.yaml        — спільні параметри
  ai_generated.yaml   — real vs AI
  edited.yaml         — original vs edited
data/
  ai_generated/       — датасет AI-детекції
  edited/             — датасет редагувань
src/kkp/              — вихідний код
tests/                — тести
notebooks/            — EDA
outputs/              — результати (не в git)
```

Параметри експериментів — у `configs/*.yaml`. Змінні з `.env` (`DATA_DIR`, `DEVICE`) підставляються автоматично.

## Запуск

### Локально

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

make install                       # залежності + pre-commit
cp .env.example .env

make test
make lint
```

Перед комітом автоматично запускаються перевірки коду (Ruff) і формату повідомлення.

### Коміти

[Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <description>`

| Тип | Коли використовувати |
|-----|----------------------|
| `feat` | нова функціональність |
| `fix` | виправлення бага |
| `docs` | документація |
| `test` | тести |
| `refactor` | рефакторинг без зміни поведінки |
| `chore` | інфраструктура, налаштування |
| `ci` | CI/CD |

```bash
git commit -m "chore: init project"
git commit -m "feat: add data loader"
git commit -m "fix(data): correct image path"
```

### Docker

```bash
cp .env.example .env
docker compose build

docker compose run --rm dev        # робоче середовище
docker compose run --rm test       # тести
docker compose run --rm lint       # лінтер
```

## CI

При push/PR: Ruff, pytest, збірка Docker-образу.

## Виконавець

Ткач Леонід, група ПЗПІ-23-5
