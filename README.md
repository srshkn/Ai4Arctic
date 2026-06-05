## Кратко

- **Архитектура:** ConvLSTM v2 — стек ConvLSTM-ячеек + регрессионная голова
- **Цель обучения:** TTOP (температура на верхней границе мерзлоты) с поправкой на латентное тепло → MAGT
- **Признаки:** до 20 каналов (LST, температуры воздуха, осадки, индексы вегетации, снежный покров, почвенные параметры)
- **Сетка:** 0.1° lat × 0.1° lon, территория России 30-180°E × 55-78°N
- **Валидация:** Obu 2019, ESA CCI Permafrost, in-situ бурения (33 точки)

## Метрики модели

| Метрика            | v2 baseline | v3 (после bias correction) |
| ------------------ | ----------- | -------------------------- |
| RMSE vs Obu 2019   | 2.07 °C     | будет                      |
| RMSE vs 33 бурения | ~2.0 °C     | будет                      |
| Pearson r          | 0.943       | будет                      |
| R²                 | 0.97        | будет                      |
| MC Dropout σ       | 0.22 °C     | будет                      |

## Структура репозитория

```
Ai4Arctic/
├── README.md                Этот файл
├── requirements.txt         Python зависимости
├── .gitignore               Исключает большие данные и временные файлы
├── src/                     Переиспользуемые модули
│   ├── model.py             ConvLSTM (нужно перенести из старой работы)
│   ├── data.py              Dataset, нормализация
│   ├── inference.py         Inference utils + latent heat correction
│   ├── metrics.py           RMSE, R², per-zone метрики
│   ├── landcover.py         rk(landcover), ΔT, ALT через Стефана
│   ├── bias_correction.py   ⭐ P1: калибровка по бурениям
│   ├── ablation.py          ⭐ P4: feature ablation
│   ├── data_extended.py     ⭐ P2: sliding windows 2001-2020
│   └── esa_cci.py           ⭐ P3: загрузка ESA CCI target
├── scripts/                 Подготовка данных и батч-задачи
│   ├── compute_ttop_target.py    Расчёт TTOP target (старый)
│   ├── rasterize_to_tensor.py    Сборка тензора из GeoJSON тайлов
│   ├── gee_export_2000_2020.py   ⭐ P2: GEE экспорт за 20 лет
│   └── download_esa_cci.py       ⭐ P3: скачивание ESA CCI с CEDA
├── notebooks/               Запускаемые ноутбуки
│   ├── 01_data_preparation.ipynb  Подготовка данных
│   ├── 02_train_model.ipynb       Обучение v2
│   ├── 03_inference.ipynb         Inference + LH correction
│   ├── 04_validation.ipynb        Валидация на Obu/ESA/бурениях
│   ├── 05_bias_correction.ipynb   ⭐ P1
│   ├── 06_train_extended.ipynb    ⭐ P2
│   ├── 07_feature_ablation.ipynb  ⭐ P4
│   └── 08_esa_cci_training.ipynb  ⭐ P3
└── data/
    ├── README.md                  Описание данных
    └── external_links.md          Ссылки на большие файлы (Drive/Zenodo)
```

⭐ = новые компоненты v3 для исправлений руководителя.

## Быстрый старт

### 1. Зависимости

```bash
python -m venv .venv
source .venv/bin/activate   # на Mac/Linux
# или: .venv\Scripts\activate на Windows

pip install -r requirements.txt
```

### 2. Данные

Большие файлы (тензор ~200 МБ, Obu TIFF ~1.8 ГБ, ESA CCI ~1 ГБ) **не входят** в репозиторий. Скачать по ссылкам в [data/external_links.md](data/external_links.md) и положить в `data/`.

### 3. Запуск

В Colab или локально открой ноутбуки по порядку:

```
01 → 02 → 03 → 04          # v2 baseline (если ещё не обучено)
05 → 07 → 06 → 08          # v3 upgrade (P1, P4, P2, P3)
```

Все ноутбуки автодетектят Colab vs локальный запуск — первая ячейка определяет `BASE_DIR`.
