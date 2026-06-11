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
python3 scripts/smoke_test.py
```

### 3. Запуск pipeline v3

**Полная инструкция:** [PIPELINE.md](PIPELINE.md) — 8 этапов от GEE export до графиков.

Минимальный путь (если данные и модели уже в репозитории):

```bash
python3 scripts/projection_2026_2035.py
python3 scripts/plot_delta_vs_baseline.py
```

Полный путь с нуля:

```bash
earthengine authenticate
python3 scripts/gee_export.py --years 2003-2025 --project YOUR_GCP_PROJECT
# скачать выгрузку в data/gee/

python3 scripts/merge_bands.py --years 2003-2025
python3 scripts/rasterize_extended.py
python3 scripts/rasterize_2025.py
python3 scripts/compute_target_23y.py
python3 scripts/train_p3_model_A_ensemble.py
python3 scripts/train_p3_model_B_ensemble.py
python3 scripts/projection_2026_2035.py
```

## Структура репозитория

```
Ai4Arctic/
├── PIPELINE.md              # ⭐ Главная инструкция v3 (этапы 1–8)
├── README.md                # Этот файл
├── requirements.txt
├── src/                     # Модули: model, data, landcover, ablation, …
├── scripts/                 # Воспроизводимый pipeline (GEE → прогноз 2035)
├── notebooks/               # Baseline v2 (2010–2023): обучение, валидация, ablation
├── models/                  # Чекпоинты ConvLSTM (P2, P3, ablation)
├── data/                    # Данные (малые в git, большие — см. external_links)
├── results/                 # Карты, метрики, фигуры
├── docs/                    # Архитектура, воспроизведение v2
└── archive/                 # История исследований (не использовать для v3)
```

## Данные

| Категория | Файлы | Где описано |
|-----------|-------|-------------|
| Статические (в git) | `y_new_rk_landcover.npz`, `maps_lh_v3.npz`, … | [data/README.md](data/README.md) |
| Генерируются pipeline | `tensor_01deg_extended_23y.npz`, target 23y | [PIPELINE.md](PIPELINE.md) |
| Скачивание / GEE | большие тензоры, эталоны | [data/external_links.md](data/external_links.md) |

## Два пути в проекте

| Путь | Для чего | Точка входа |
|------|----------|-------------|
| **v3 scripts** | Прогноз 2026–2035, защита | [PIPELINE.md](PIPELINE.md) |
| **v2 notebooks** | Baseline 2010–2023, ablation, валидация | [notebooks/README.md](notebooks/README.md) |

## Дополнительно

- Скрипты (детали): [scripts/README.md](scripts/README.md)
- Архитектура модели: [docs/model_architecture.md](docs/model_architecture.md)
- Модели и метрики: [models/README.md](models/README.md)
