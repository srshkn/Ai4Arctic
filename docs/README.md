# MAGT ConvLSTM Model — Overview

## Кратко

ConvLSTM-модель для карт **MAGT** (температура верхней границы мерзлоты) по России
(30–180°E, 55–78°N, сетка 0.1°).

- **Архитектура:** ConvLSTM + регрессионная голова (6 climate_core признаков в финальной модели)
- **Target:** TTOP с `rk(landcover)` → MAGT с поправкой на латентное тепло и bias correction
- **Данные:** 23 года признаков (2003–2025), прогноз 2026–2035
- **Валидация:** Obu 2019, ESA CCI, 33 GTN-P бура

## Быстрый старт

### 1. Зависимости

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Проверка окружения

```bash
python3 model/scripts/smoke_test.py
```

### 3. Запуск pipeline v3

**Полная инструкция:** [PIPELINE.md](PIPELINE.md) — 8 этапов от GEE export до графиков.

Минимальный путь (если данные и модели уже в репозитории):

```bash
python3 model/scripts/projection_2026_2035.py
python3 model/scripts/plot_delta_vs_baseline.py
```

Полный путь с нуля:

```bash
earthengine authenticate
python3 model/scripts/gee_export.py --years 2003-2025 --project YOUR_GCP_PROJECT
# скачать выгрузку в model/data/gee/

python3 model/scripts/merge_bands.py --years 2003-2025
python3 model/scripts/rasterize_extended.py
python3 model/scripts/rasterize_2025.py
python3 model/scripts/compute_target_23y.py
python3 model/scripts/train_p3_model_A_ensemble.py
python3 model/scripts/train_p3_model_B_ensemble.py
python3 model/scripts/projection_2026_2035.py
```

### 4. Экспорт GeoTIFF для бэкенда / базы данных

```bash
pip install rasterio
python3 model/scripts/export_geotiff.py
```

Создаст 29 файлов `model/results/geotiff/magt_YYYY.tif` (2007–2035), EPSG:4326,
float32, COG-совместимые (tiled + deflate). Готовы к загрузке в PostGIS / titiler / любой GIS-бэкенд.

## Структура репозитория

```
Ai4Arctic/
├── README.md                       # Общий README репо (development)
├── requirements.txt
├── docs/
│   ├── MODEL_OVERVIEW.md           # ⭐ Этот файл — про ConvLSTM-модель
│   ├── PIPELINE.md                 # Полная инструкция v3 (этапы 1–8)
│   └── model_architecture.md       # Детали архитектуры
└── model/                          # Всё для воспроизведения / Docker
    ├── src/                        # Модули: model, data, landcover, ablation, …
    ├── scripts/                    # Воспроизводимый pipeline (GEE → прогноз 2035)
    ├── notebooks/                  # см. /notebooks/ в корне
    ├── models/                     # Чекпоинты ConvLSTM (P2, P3, ablation)
    ├── data/                       # Данные (малые в git, большие — см. external_links)
    └── results/
        ├── maps/                   # .npz карты прогнозов
        ├── figures/                # графики, интерактивные HTML
        └── geotiff/                # GeoTIFF для бэкенда (генерируется)
```

## Данные

| Категория             | Файлы                                         | Где описано                                                     |
| --------------------- | --------------------------------------------- | --------------------------------------------------------------- |
| Статические (в git)   | `y_new_rk_landcover.npz`, `maps_lh_v3.npz`, … | [model/data/README.md](../model/data/README.md)                 |
| Генерируются pipeline | `tensor_01deg_extended_23y.npz`, target 23y   | [PIPELINE.md](PIPELINE.md)                                      |
| Скачивание / GEE      | большие тензоры, эталоны                      | [model/data/external_links.md](../model/data/external_links.md) |
| Выходные GeoTIFF      | `magt_2007.tif` … `magt_2035.tif`             | генерируются `export_geotiff.py`                                |

## Два пути в проекте

| Путь             | Для чего                                | Точка входа                                   |
| ---------------- | --------------------------------------- | --------------------------------------------- |
| **v3 scripts**   | Прогноз 2026–2035, защита               | [PIPELINE.md](PIPELINE.md)                    |
| **v2 notebooks** | Baseline 2010–2023, ablation, валидация | [notebooks/README.md](../notebooks/README.md) |

## Дополнительно

- Скрипты (детали): [model/scripts/README.md](../model/scripts/README.md)
- Архитектура модели: [model_architecture.md](model_architecture.md)
- Модели и метрики: [model/models/README.md](../model/models/README.md)
