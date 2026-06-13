# Ноутбуки

Основные ноутбуки проекта, разбитые по этапам pipeline.

## Список ноутбуков

| Ноутбук | Назначение | Время выполнения |
|---|---|---|
| `01_data_preparation.ipynb` | Загрузка тензора, построение landcover и rk_map, пересчёт target | ~5 мин |
| `02_train_model.ipynb` | Обучение ConvLSTM (80 эпох, lr=1e-4) | ~30 мин на T4 GPU |
| `03_inference.ipynb` | Прогноз 2014–2024, latent heat correction, расчёт ALT | ~10 мин |
| `04_validation.ipynb` | Валидация: Obu, ESA, боры, per-zone, MC Dropout, feature importance, региональные карты | ~40 мин |

## Порядок запуска

Ноутбуки должны запускаться **строго по порядку**, потому что каждый следующий использует результаты предыдущего.

```
01_data_preparation → 02_train_model → 03_inference → 04_validation
```

## Зависимости

Все ноутбуки импортируют функции из модуля `src/`:

```python
import sys
sys.path.append('..')

from src.model import ConvLSTMNet, load_checkpoint
from src.data import load_tensor, normalize_features, make_pairs, TileDataset
from src.inference import predict_full_map, mc_dropout_predict
from src.metrics import all_metrics, per_zone_metrics
from src.landcover import classify_landcover, build_rk_map, build_delta_t_map, compute_alt
```

## Пути к данным

Все ноутбуки используют **относительные пути** от папки `notebooks/`:

| Что | Путь |
|---|---|
| Входной тензор | `../data/tensor_01deg_v2.npz` |
| Эталон Obu (для валидации) | `../data/obu_2019_reprojected.npz` |
| Эталон ESA | `../data/comparison_esacci_2023.npz` |
| Буровые станции | `../data/boreholes/clean_boreholes_33.csv` |
| Финальная модель | `../models/convlstm_ttop_rk_v2.pt` |
| Финальные карты | `../results/maps/FINAL_maps.npz` |
| Метрики (CSV) | `../results/metrics/...` |
| Графики (PNG) | `../results/figures/...` |

Если каких-то файлов нет — см. [`../data/external_links.md`](../data/external_links.md) для скачивания больших данных.

## Системные требования

- **Минимум:** 16 ГБ RAM, CPU (медленнее)
- **Рекомендуется:** GPU с ≥ 8 ГБ VRAM (T4, V100, A100, RTX 3060+)
- **Сильно рекомендуется:** SSD для быстрого чтения тензора 200 МБ
