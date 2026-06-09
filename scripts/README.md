# Скрипты — пошаговая инструкция

Все скрипты воспроизводимы и автономны. Каждый отдельно описывает что делает, что
ожидает на входе и что создаёт на выходе. После любого скрипта можно остановиться
и проверить промежуточный результат.

---

## Карта скриптов

| Скрипт | Что делает | Время | Сценарий |
|---|---|---|---|
| `smoke_test.py` | Проверка импортов и компиляции 19 модулей | <1 мин | Всегда |
| `gee_export_2003_2009.py` | GEE экспорт 7 лет × 4 поддиапазона = 28 задач | 1–3 ч (фон) | B |
| `gee_export_2024.py` | GEE экспорт 2024 года (4 задачи) | 15–30 мин (фон) | B, D |
| `merge_bands_extended.py` | Склейка 4 поддиапазонов → годовой geojson × 21 год | ~5 мин | B |
| `rasterize_extended.py` | geojson → X-тензор (21, 231, 1501, 20) | ~30 сек | B |
| `compute_target_extended.py` | TTOP target через `src.landcover` | ~10 сек | B |
| `train_extended_21years.py` | Обучение P2 ConvLSTM на 21 годе | 5–10 мин | B |
| `inference_p2_all_years.py` | Inference P2 на 2007–2023 (17 карт) | <1 мин | B, C |
| `process_2024.py` | Полный pipeline для одного нового года | ~3 мин | B, D |
| `plot_yearly_trend_p2.py` | Графики тренда + first vs last year | ~10 сек | B, C |
| `build_interactive_map.py` | Plotly slider 2007–2024 (HTML) | ~1 мин | B, C |
| `compare_p2_vs_p4_on_boreholes.py` | P2 vs P4 на 33 бурениях | ~2 мин | C |
| `reproject_obu.py` | Obu 2019 TIFF (EPSG:3995) → наша сетка | ~1 мин | A |
| `dedupe_p2_yearly_maps.py` | Утилита: убрать дубли годов в p2_yearly_maps | <5 сек | По нужде |
| `cleanup_repo.py` | Утилита: archive research history + .gitignore | ~10 сек | Перед commit |

**Сценарии (см. ниже): A — базовая P4 модель, B — P2 с нуля, C — использовать готовое, D — добавить новый год.**

---

## Сценарий A: воспроизвести базовую модель P4 с нуля

**Цель:** получить P4 winner модель (14 лет 2010–2023, val RMSE 0.76°C) и финальную карту 2023.

**Что нужно:**
- `data/tensor_01deg_v2.npz` (~200 МБ) — скачать из `data/external_links.md`
- Obu 2019 TIFF (опционально, для валидации) — скачать тот же файл

**Шаги (через Jupyter notebooks):**

```bash
cd ~/Ai4Arctic
source .venv/bin/activate
jupyter notebook notebooks/
```

Запускай notebooks по порядку:

1. `01_data_preparation.ipynb` — загружает X тензор, строит landcover, считает rk_map
2. `02_train_model.ipynb` — обучает ConvLSTM на 14 годах, lr=1e-4, 80 epochs (~30 мин на T4 GPU, ~10 мин на MPS Mac M1)
3. `03_inference.ipynb` — inference 2023, + lh коррекция
4. `04_validation.ipynb` — валидация: Obu, ESA, буры, MC Dropout
5. `05_bias_correction.ipynb` — P1: pure shift калибровка → финальная карта
6. `07_feature_ablation.ipynb` — P4: ablation 7 конфигураций (~15 мин на MPS)

**Артефакты после Сценария A:**
- `models/convlstm_ttop_rk_v2_<winner>.pt` — P4 winner (лучший val RMSE из ablation)
- `results/maps/MAGT_2023_bias_corrected.npz` — финальная карта
- `results/metrics/bias_correction_summary.csv` — P1 метрики
- `results/metrics/ablation_table.csv` — P4 метрики

---

## Сценарий B: воспроизвести P2 (расширенный 2003–2024)

**Цель:** получить P2 модель + 18 годовых карт 2007–2024 + интерактивный slider.

**Что нужно:**
- Google Cloud project с включённым Earth Engine (~5 мин регистрации)
- ~150 МБ свободного места на Google Drive
- ~5 ГБ свободного места локально

### Шаг B.1: GEE экспорт (~3 часа фоном)

```bash
# Авторизация (один раз)
earthengine authenticate

# Экспорт 2003–2009 (28 задач)
python3 scripts/gee_export_2003_2009.py

# Экспорт 2024 (4 задачи)
python3 scripts/gee_export_2024.py

# Мониторить статус: https://code.earthengine.google.com/tasks
# Когда все COMPLETED — скачать с Drive в data/gee/
```

Должно быть **84 файлов** в `data/gee/` (21 год × 4 поддиапазона) для 2003–2023 + 4 для 2024.

### Шаг B.2: Локальный pipeline (~15 мин)

```bash
# Склейка 4 поддиапазонов в годовые файлы (21 × 4 → 21 годовых)
python3 scripts/merge_bands_extended.py
# → data/gee/RussiaGrid_0.1deg_v2_<year>.geojson (21 файл)

# Растеризация → X-тензор
python3 scripts/rasterize_extended.py
# → data/tensor_01deg_extended.npz (285 МБ, shape (21, 231, 1501, 20))

# Пересчёт TTOP target через landcover-rk
python3 scripts/compute_target_extended.py
# → data/y_new_rk_landcover_extended.npz (17.5 МБ)
# Sanity check: max abs diff vs старого y_new для 2010-2023 = 0.0000°C ✓

# Обучение P2 ConvLSTM на 21 годе (5–10 мин на MPS)
python3 scripts/train_extended_21years.py
# → models/convlstm_ttop_extended_21years.pt (val RMSE ~0.84°C)

# Inference P2 на 2007–2023 (17 карт)
python3 scripts/inference_p2_all_years.py
# → results/maps/p2_yearly_maps.npz (28 МБ, 17 годов)

# Добавить 2024 (полный pipeline в одном скрипте)
python3 scripts/process_2024.py
# → results/maps/p2_yearly_maps.npz (31 МБ, 18 годов)
```

### Шаг B.3: Визуализации (~2 мин)

```bash
# Графики тренда + 2007 vs 2024
python3 scripts/plot_yearly_trend_p2.py
# → results/figures/p2_yearly_trend.png
# → results/figures/p2_first_vs_last_year.png

# Интерактивный Plotly slider (HTML, открывается в браузере)
python3 scripts/build_interactive_map.py
# → results/figures/p2_interactive_full.html (40 МБ, для отчёта)
# → results/figures/p2_interactive_lite.html (10 МБ, для устной защиты)
```

**Артефакты после Сценария B:**
- `models/convlstm_ttop_extended_21years.pt` — P2 модель
- `data/tensor_01deg_extended.npz` — X для 21 года
- `data/y_new_rk_landcover_extended.npz` — TTOP target для 21 года
- `results/maps/p2_yearly_maps.npz` — 18 карт 2007–2024
- `results/figures/p2_*.png/html` — графики и slider

---

## Сценарий C: использовать готовые модели

**Цель:** не переобучать, а взять existing checkpoint и сделать inference.

**Что нужно:**
- Загруженные `models/*.pt` (5 МБ всего)
- `data/tensor_01deg_extended.npz` (если хочешь работать с 2003–2024)

### Inference на бурениях

```bash
# Сравнить P2 и P4 на 33 бурениях
python3 scripts/compare_p2_vs_p4_on_boreholes.py
# → results/metrics/p2_vs_p4_boreholes_comparison.json
```

### Только обновить графики из готовых карт

```bash
python3 scripts/plot_yearly_trend_p2.py        # читает p2_yearly_maps.npz
python3 scripts/build_interactive_map.py       # читает p2_yearly_maps.npz
```

### Загрузить модель в своём Python коде

```python
import torch
import sys
sys.path.insert(0, '/path/to/Ai4Arctic')

from src.model import ConvLSTMNet

ckpt = torch.load('models/convlstm_ttop_extended_21years.pt',
                  map_location='cpu', weights_only=False)
model = ConvLSTMNet(in_ch=6)
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Метаданные:
print(f"Feature names: {ckpt['feature_names']}")
print(f"Val RMSE: {ckpt['val_rmse']}")
print(f"y_mean: {ckpt['y_mean']}, y_std: {ckpt['y_std']}")
```

---

## Сценарий D: добавить новый год после 2024

**Цель:** когда появятся данные за 2025, 2026, добавить их в slider.

### Шаг D.1: GEE экспорт нового года

Адаптировать `gee_export_2024.py` под нужный год — поменять `YEARS_NEW = [2025]`. Скачать 4 файла с Drive в `data/gee/`.

### Шаг D.2: Обновить process_2024.py под новый год

В файле `scripts/process_2024.py` поменять константу:

```python
YEAR = 2025   # было 2024
```

И запустить:

```bash
python3 scripts/process_2024.py
# → расширяет тензор до 23 лет (2003–2025)
# → добавляет 2025 в p2_yearly_maps.npz
```

### Шаг D.3: Перерисовать графики

```bash
python3 scripts/plot_yearly_trend_p2.py        # тренд теперь по 19 годам
python3 scripts/build_interactive_map.py       # slider теперь 2007–2025
```

**Внимание:** MODIS Terra MOD11A1.061 заканчивается 15 октября 2025 года.
Для полного года 2025 нужно либо ждать MODIS v7.0, либо переходить на VIIRS.

---

## Что делать если что-то пошло не так

| Проблема | Решение |
|---|---|
| `ImportError: cannot import name X from src.*` | Перезапустить kernel Jupyter (Cmd+Shift+P → Restart Kernel) |
| `Earth Engine client library not initialized` | `earthengine authenticate` |
| `SSL: CERTIFICATE_VERIFY_FAILED` на Mac | `/Applications/Python 3.11/Install Certificates.command` |
| GEE задача FAILED | Проверь https://code.earthengine.google.com/tasks, типичная причина — превышение quota |
| Дубликаты годов в p2_yearly_maps.npz | `python3 scripts/dedupe_p2_yearly_maps.py` |
| Перед коммитом нужна очистка | `python3 scripts/cleanup_repo.py --execute` |
| Smoke test упал | `python3 scripts/smoke_test.py 2>&1 \| tail -20` и посмотри first FAIL |

---

## Зависимости между скриптами

```
gee_export_2003_2009.py  ─┐
gee_export_2024.py       ─┤
                          ↓
         merge_bands_extended.py
                          ↓
         rasterize_extended.py  →  data/tensor_01deg_extended.npz
                          ↓                    ↓
         compute_target_extended.py            ↓
                          ↓                    ↓
         data/y_new_rk_landcover_extended.npz  ↓
                          ↓                    ↓
                  train_extended_21years.py ───┘
                          ↓
            models/convlstm_ttop_extended_21years.pt
                          ↓
         inference_p2_all_years.py
                          ↓
            results/maps/p2_yearly_maps.npz (17 лет)
                          ↓
         process_2024.py  →  p2_yearly_maps.npz (18 лет)
                          ↓
              ┌───────────┴───────────┐
              ↓                       ↓
   plot_yearly_trend_p2.py    build_interactive_map.py
              ↓                       ↓
         PNG графики          HTML slider
```

---

## Ключевые принципы pipeline

1. **Идемпотентность.** Большинство скриптов проверяют существующие файлы и пропускают шаги:
   - `merge_bands_extended.py` пропускает годы с уже существующим merged файлом
   - `process_2024.py` пропускает шаги если выход уже на диске

2. **Sanity checks.** `compute_target_extended.py` проверяет что для общих годов
   (2010–2023) новый target математически идентичен старому (max abs diff < 0.001°C).

3. **Воспроизводимость.** Каждый скрипт начинается с описания вход/выход, чтобы можно было запустить любой шаг отдельно.

4. **Никаких heredoc/python << EOF.** Вся логика — в файлах скриптов, чтобы рецензент мог открыть и прочитать что именно делалось.
