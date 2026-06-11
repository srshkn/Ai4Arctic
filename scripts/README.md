# Скрипты

Воспроизводимый pipeline v3: данные 2003–2025 → прогноз MAGT 2026–2035.

**Главная инструкция по этапам:** [../PIPELINE.md](../PIPELINE.md)

Каждый скрипт автономен: в начале файла описаны входы и выходы. После любого шага можно остановиться и проверить артефакт.

---

## Скрипты в `scripts/` (актуальные)

### Pipeline v3 — этапы 1–8

| Этап | Скрипт | Выход | Время |
|------|--------|-------|-------|
| — | `smoke_test.py` | проверка импортов | <1 мин |
| 1 | `gee_export.py` | 92 geojson на Drive → `data/gee/` | 2–4 ч (фон) |
| 2 | `merge_bands.py` | 23 годовых geojson | ~5 мин |
| 3a | `rasterize_extended.py` | `tensor_01deg_extended_22y.npz` | ~30 сек |
| 3b | `rasterize_2025.py` | `tensor_01deg_extended_23y.npz` | ~30 сек |
| 4 | `compute_target_23y.py` | `y_new_rk_landcover_extended_23y.npz` | ~10 сек |
| 5 | `inference_p2_history.py` | `p2_yearly_maps.npz` + `p2_real_2025.npz` | ~2 мин |
| 5a | `inference_p2_yearly_maps.py` | только `p2_yearly_maps.npz` (2007–2024) | ~1 мин |
| 5b | `inference_p2_2025.py` | только `p2_real_2025.npz` | ~1 мин |
| 6 | `train_p3_model_A_ensemble.py` | Model A (3 seed + BEST) | ~15 мин |
| 6 | `train_p3_model_B_ensemble.py` | Model B (3 seed + BEST) | ~15 мин |
| 6+ | `train_and_ensemble_7models.py` | +4 seed к Model A | ~20 мин |
| 7 | `projection_2026_2035.py` | карты 2026–2035, synthetic tensor | ~10 мин |
| 8 | `plot_ensemble6_final.py` | тренд 2007–2035, uncertainty, HTML | ~5 мин |
| 8 | `plot_delta_vs_baseline.py` | ΔMAGT vs baseline 2018–2024 | ~3 мин |
| 8 | `plot_delta_permafrost_only.py` | потепление в зоне мерзлоты | ~3 мин |

### Вспомогательные

| Скрипт | Назначение |
|--------|------------|
| `train_extended_21years.py` | Обучение P2 на 2003–2023 (нужно только если нет `convlstm_ttop_extended_21years.pt`) |
| `reproject_obu.py` | Obu TIFF → `obu_2019_reprojected.npz` (валидация v2) |

Устаревшие аналоги (`gee_export_2003_2009.py`, `process_2024.py`, …) — в `archive/research_history/scripts/`, **не использовать**.

---

## Prerequisites

До этапа 4 (не создаются скриптами 1–3):

| Файл | Источник |
|------|----------|
| `data/y_new_rk_landcover.npz` | В git / v2 baseline |
| `data/maps_lh_v3.npz` | В git / notebook 03 |
| `results/metrics/ablation_table.csv` | `notebooks/07_feature_ablation.ipynb` |

Для этапа 5:

| Файл | Источник |
|------|----------|
| `models/convlstm_ttop_extended_21years.pt` | В git **или** `train_extended_21years.py` |

Подробнее: [../data/external_links.md](../data/external_links.md)

---

## Сценарии запуска

### Сценарий 1: полный v3 pipeline с нуля

```bash
earthengine authenticate

python3 scripts/gee_export.py --years 2003-2025 --project YOUR_GCP_PROJECT
# скачать data/gee/ с Drive

python3 scripts/merge_bands.py --years 2003-2025
python3 scripts/rasterize_extended.py
python3 scripts/rasterize_2025.py
python3 scripts/compute_target_23y.py

python3 scripts/train_p3_model_A_ensemble.py
python3 scripts/train_p3_model_B_ensemble.py
python3 scripts/projection_2026_2035.py

python3 scripts/inference_p2_history.py          # опционально
python3 scripts/plot_ensemble6_final.py
python3 scripts/plot_delta_vs_baseline.py
python3 scripts/plot_delta_permafrost_only.py
```

### Сценарий 2: только прогноз (данные и модели в репо)

```bash
python3 scripts/smoke_test.py
python3 scripts/projection_2026_2035.py
python3 scripts/plot_delta_vs_baseline.py
```

### Сценарий 3: baseline v2 (ноутбуки, не скрипты)

P4 ablation, bias correction, валидация на Obu/ESA/бурах:

```
notebooks/01 → 02 → 03 → 04 → 05 → 07
```

См. [../notebooks/README.md](../notebooks/README.md).

### Сценарий 4: добавить новый год после 2025

```bash
python3 scripts/gee_export.py --years 2026 --project YOUR_GCP_PROJECT
# скачать 4 файла в data/gee/

python3 scripts/merge_bands.py --years 2026
# адаптировать rasterize_2025.py под новый год (скопировать логику)
python3 scripts/compute_target_23y.py   # после обновления тензора
```

---

## Зависимости между скриптами

```
gee_export.py
      ↓
merge_bands.py
      ↓
rasterize_extended.py  →  tensor_01deg_extended_22y.npz
      ↓
rasterize_2025.py      →  tensor_01deg_extended_23y.npz
      ↓
y_new_rk_landcover.npz → compute_target_23y.py
      ↓
y_new_rk_landcover_extended_23y.npz
      ↓
train_p3_model_A_ensemble.py ─┐
train_p3_model_B_ensemble.py ─┤
train_and_ensemble_7models.py ┘ (опционально)
      ↓
maps_lh_v3.npz → projection_2026_2035.py
      ↓
p3_projection_*.npz, tensor_01deg_synthetic_2026_2035.npz
      ↓
plot_delta_vs_baseline.py / plot_delta_permafrost_only.py

(параллельная ветка для тренда)
convlstm_ttop_extended_21years.pt → inference_p2_history.py
      ↓
p2_yearly_maps.npz + p2_real_2025.npz
      ↓
plot_ensemble6_final.py
```

---

## Загрузка модели в Python

```python
import torch
import sys
sys.path.insert(0, '/path/to/Ai4Arctic')

from src.model import ConvLSTMNet

ckpt = torch.load('models/convlstm_ttop_p3_model_A_BEST.pt',
                  map_location='cpu', weights_only=False)
model = ConvLSTMNet(in_ch=6)
model.load_state_dict(ckpt['state_dict'])
model.eval()

print(f"Features: {ckpt['feature_names']}")
print(f"Val RMSE: {ckpt['val_rmse']:.3f}°C")
```

---

## Troubleshooting

| Проблема | Решение |
|----------|---------|
| `ImportError` из `src.*` | `python3 scripts/smoke_test.py`, перезапустить venv |
| GEE не инициализируется | `earthengine authenticate` |
| GEE задача FAILED | https://code.earthengine.google.com/tasks (quota, timeout) |
| `SSL: CERTIFICATE_VERIFY_FAILED` (Mac) | `/Applications/Python 3.11/Install Certificates.command` |
| Нет `tensor_23y` | Скачать из `data/external_links.md` или пройти этапы 1–3 |
| Нет `ablation_table.csv` | Запустить `notebooks/07_feature_ablation.ipynb` |
| Нет P2-модели для этапа 5 | Использовать чекпоинт из `models/` или `train_extended_21years.py` |
| Smoke test 16/18 | Нормально: `src.data_extended` и `src.esa_cci` только в `archive/` |

---

## Принципы

1. **Идемпотентность** — `merge_bands.py` пропускает уже склеенные годы (без `--overwrite`).
2. **Sanity checks** — `compute_target_23y.py` сверяет target с baseline (2010–2023).
3. **Разделение target и inference** — этап 4 считает TTOP по формуле; нейросеть — на этапах 5–7.
4. **Один GEE-скрипт** — `gee_export.py` заменяет старые `gee_export_2003_2009.py` и `gee_export_2025.py`.
