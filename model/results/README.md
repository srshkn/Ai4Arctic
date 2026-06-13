# Результаты

Финальные результаты работы модели — карты, метрики, визуализации.

## Структура

```
results/
├── metrics/   # CSV-таблицы с метриками (~5 файлов, < 1 МБ)
├── maps/      # Бинарные карты в формате .npz (~30 МБ)
└── figures/   # Ключевые визуализации (PNG, ~5 МБ)
```

## metrics/ — таблицы метрик

| Файл | Описание |
|---|---|
| `metrics_main_table.csv` | **Главная сводная таблица** всех валидаций (vs Obu, ESA, боры) |
| `all_metrics_summary.csv` | Расширенная сводка с дополнительными метриками (KGE, MedAE) |
| `convlstm_per_zone_metrics.csv` | Per-zone валидация (континуальная / прерывистая / спорадическая) |
| `feature_importance.csv` | Permutation Feature Importance: ΔMAE и ΔRMSE для 20 признаков |

## maps/ — бинарные карты

| Файл | Описание | Размер |
|---|---|---|
| `FINAL_maps.npz` | pred_2023, pred_2024, pred_2023_lh, pred_2024_lh, real_2023, lons, lats | ~11 МБ |
| `ALT_2023.npz` | ALT_2023, E_map, TDD_2023, lons, lats | ~4 МБ |
| `bonus_results.npz` | delta_MAGT, trend_per_year, vulnerability, categories | ~7 МБ |
| `uncertainty_mc_dropout.npz` | mean_pred, std_pred, ci_low, ci_high (30 прогонов) | ~4 МБ |

### Загрузка карт

```python
import numpy as np

final = np.load('results/maps/FINAL_maps.npz')
pred_2023_lh = final['pred_2023_lh']    # (231, 1501) — финальная карта MAGT
lons = final['lons']
lats = final['lats']

alt = np.load('results/maps/ALT_2023.npz')['ALT_2023']
```

## figures/ — визуализации

### Главные слайды для защиты

| Файл | Назначение |
|---|---|
| `three_way_comparison.png` | **Главный 6-панельный слайд:** Наша / Obu / ESA + три разности |
| `feature_importance.png` | Важность 20 признаков (горизонтальная диаграмма) |
| `permafrost_vulnerability.png` | Карта уязвимости (4 категории) |
| `uncertainty_mc_dropout.png` | Карта неопределённости σ |
| `MAGT_change_2010_2023.png` | Изменение MAGT за 14 лет |
| `ALT_2023_map.png` | Карта ALT за 2023 |
| `convlstm_yearly_validation.png` | Yearly валидация real vs pred |

### Региональные карты (5 регионов)

4-панельные карты для каждого региона: MAGT, ALT, σ, уязвимость.

| Файл | Регион |
|---|---|
| `region_Якутия_континуальная.png` | Якутия |
| `region_Ямал_и_ЯНАО.png` | Ямал и ЯНАО |
| `region_Воркута__Северный_Урал.png` | **Воркута** (тёплая мерзлота) |
| `region_Чукотка.png` | Чукотка |
| `region_Прибайкалье.png` | Прибайкалье (южная граница) |

## Главные числа работы

```
RMSE vs Obu 2019      :  2.07 °C   ← главный результат
Pearson r vs Obu      : +0.943
R² на target          : +0.97
σ (MC Dropout)        :  0.22 °C
% территории с потеплением (2010→2023) : 99.4%
Средний тренд         : +0.088 °C/год
```

## Воспроизведение результатов

Все эти файлы генерируются ноутбуками:

| Файл | Создаётся в |
|---|---|
| `FINAL_maps.npz`, `pred_*.npz` | `notebooks/03_inference.ipynb` |
| `ALT_2023.npz` | `notebooks/03_inference.ipynb` |
| `bonus_results.npz`, `*_vulnerability*` | `notebooks/04_validation.ipynb` |
| `uncertainty_mc_dropout.npz` | `notebooks/04_validation.ipynb` |
| `metrics_main_table.csv`, прочие CSV | `notebooks/04_validation.ipynb` |
| Все PNG | `notebooks/04_validation.ipynb` |
