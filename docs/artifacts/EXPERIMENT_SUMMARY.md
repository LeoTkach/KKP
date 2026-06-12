# Experiment summary (frozen results)

Цей файл генерується `make experiment-audit`. Метрики можна **перевірити без повторного навчання**:

```bash
make download-hires-hf    # один раз
make train && make train-efficientnet   # якщо checkpoint-ів ще немає
make verify-results       # evaluate + порівняння з цим файлом
```

Checkpoint-и (`.pth`) не в git — зберігаються локально в `outputs/`.
У git комітяться лише `metrics.json`, графіки та цей summary.

## Test set (Parveshiiii/AI-vs-Real, N=204)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Errors |
|-------|----------|-----------|--------|-----|---------|--------|
| resnet18 | 93.6% | 0.908 | 0.971 | 0.938 | 0.992 | 13 |
| efficientnet_b0 | 98.5% | 0.971 | 1.000 | 0.986 | 1.000 | 3 |

### resnet18

- Bootstrap accuracy 95% CI: [90.2%, 96.6%] (mean 93.6%)
- Confusion matrix [[TN, FP], [FN, TP]]: `[[92, 10], [3, 99]]`

### efficientnet_b0

- Bootstrap accuracy 95% CI: [96.6%, 100.0%] (mean 98.5%)
- Confusion matrix [[TN, FP], [FN, TP]]: `[[99, 3], [0, 102]]`

## Out-of-domain: data/real_world/

- N = 30
- Accuracy: 80.0%
- F1: 0.800
- ROC-AUC: 0.858

Падіння якості vs in-domain підтверджує domain shift (див. звіт §5.3).

## Ablation (EfficientNet-B0)

```bash
make ablation    # train missing variants + evaluate → docs/artifacts/ablation_runs.json
```

Сітка: baseline (aug=on, ep=10), aug=off ep=10, aug=on ep=5.
Конфіги: `configs/ablation_effnet_*.yaml`.

| Variant | Epochs | Aug | Best ep. | Test acc | F1 | Errors |
|---------|--------|-----|----------|----------|-----|--------|
| baseline | 10 | on | 9 | 98.5% | 0.986 | 3 |
| no_augment | 10 | off | 9 | 97.5% | 0.976 | 5 |
| epochs5 | 5 | on | 2 | 97.5% | 0.976 | 5 |

Обидва ablation-варіанти гірші за baseline (−1.0 п.п.); strong_augment + 10 epochs залишаються основним конфігом.

- `*_test_metrics.json` — повні метрики test set
- `*_misclassifications.json` — список помилкових передбачень
- `*_bootstrap.json` — bootstrap 95% CI для accuracy
- `*.png` — графіки порівняння (`make compare`)

## Config fingerprint

- Dataset: `data/ai_hires/` (HF Parveshiiii/AI-vs-Real, 512px)
- Train: 10 epochs, batch 32, lr 0.001, image 224, strong_augment
- Seed: 42
- Checkpoint selection: best val accuracy
