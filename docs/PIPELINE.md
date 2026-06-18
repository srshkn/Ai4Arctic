# Pipeline v3 — прогноз MAGT 2003–2035

Пошаговая инструкция для воспроизведения результатов дипломной работы.
Период данных: **23 года признаков (2003–2025)**, прогноз **2026–2035**.

Полный путь: **~3–5 часов** (большая часть — ожидание GEE export на Google Drive).

---

## Prerequisites (до этапа 4)

Эти файлы **не создаются** скриптами этапов 1–3. Они из v2 baseline и включены в репозиторий (или скачиваются — см. [data/external_links.md](data/external_links.md)).

| Файл | Зачем |
|------|-------|
| `data/y_new_rk_landcover.npz` | Статичные `landcover` + `rk_map` для формулы TTOP |
| `data/maps_lh_v3.npz` | `delta_t` — поправка на латентное тепло при inference |
| `results/metrics/ablation_table.csv` | Нужен для обучения P2/P3 (winner из P4 ablation) |

Для **этапа 5** (графики тренда) дополнительно нужна P2-модель:

| Файл | Как получить |
|------|--------------|
| `models/convlstm_ttop_extended_21years.pt` | В репозитории **или** `scripts/train_extended_21years.py` (обучена на 2003–2023) |

---

## Сводная таблица этапов

| Этап | Скрипт | Выход | Время | Обязательно |
|------|--------|-------|-------|-------------|
| 1 | `gee_export.py` | `data/gee/` (92 geojson) | 2–4 ч (фон) | Да* |
| 2 | `merge_bands.py` | 23 годовых geojson | ~5 мин | Да* |
| 3a | `rasterize_extended.py` | `tensor_01deg_extended_22y.npz` | ~30 сек | Да* |
| 3b | `rasterize_2025.py` | `tensor_01deg_extended_23y.npz` | ~30 сек | Да* |
| 4 | `compute_target_23y.py` | `y_new_rk_landcover_extended_23y.npz` | ~10 сек | **Да** |
| 5 | `inference_p2_history.py` | `p2_yearly_maps.npz`, `p2_real_2025.npz` | ~2 мин | Нет** |
| 6 | `train_p3_model_A_ensemble.py` | Model A (3 seed + BEST) | ~15 мин | **Да** |
| 6 | `train_p3_model_B_ensemble.py` | Model B (3 seed + BEST) | ~15 мин | **Да** |
| 6+ | `train_and_ensemble_7models.py` | +4 seed к Model A | ~20 мин | Нет |
| 7 | `projection_2026_2035.py` | карты 2026–2035, synthetic tensor | ~10 мин | **Да** |
| 8 | `plot_ensemble6_final.py` | тренд 2007–2035, HTML slider | ~5 мин | Для защиты |
| 8 | `plot_delta_vs_baseline.py` | ΔMAGT vs baseline 2018–2024 | ~3 мин | Для защиты |
| 8 | `plot_delta_permafrost_only.py` | потепление в зоне мерзлоты | ~3 мин | Для защиты |
| 9 | `export_gis_all_years.py` | MAGT 2007–2035 → `.npz` + GeoTIFF | ~1 мин | Для GIS |

\* Этапы 1–3 можно пропустить, если скачать готовый `tensor_01deg_extended_23y.npz` (см. external_links).  
\** Этап 5 нужен только для графика тренда 2007–2035 (`plot_ensemble6_final.py`).

---

## Этап 1. GEE Export

Выгрузка 20 спутниковых признаков из Google Earth Engine.

```bash
earthengine authenticate

# Замените YOUR_PROJECT на свой Google Cloud project ID
python3 scripts/gee_export.py --years 2003-2025 --project YOUR_PROJECT
```

- **Выход на Drive:** папка `GEE_exports_features_01deg_v2_extended/` (92 файла = 23 года × 4 поддиапазона)
- **Мониторинг:** https://code.earthengine.google.com/tasks
- После COMPLETED — скачать папку в `data/gee/`

---

## Этап 2. Merge bands

Склейка 4 поддиапазонов каждого года в один geojson.

```bash
python3 scripts/merge_bands.py --years 2003-2025
```

**Выход:** `data/gee/RussiaGrid_0.1deg_v2_<year>.geojson` — 23 файла.

---

## Этап 3. Rasterize

### 3a. 2003–2024 (22 года)

```bash
python3 scripts/rasterize_extended.py
```

**Выход:** `data/tensor_01deg_extended_22y.npz` — shape `(22, 231, 1501, 20)`.

### 3b. +2025 → 23 года

```bash
python3 scripts/rasterize_2025.py
```

**Выход:** `data/tensor_01deg_extended_23y.npz` — shape `(23, 231, 1501, 20)`.

---

## Этап 4. Compute target

Расчёт TTOP target по формуле (не inference нейросети):

```
y = (rk_map · TDD − FDD) / 365
```

```bash
python3 scripts/compute_target_23y.py
```

| Вход | Выход |
|------|-------|
| `tensor_01deg_extended_23y.npz` | `y_new_rk_landcover_extended_23y.npz` (~18 МБ) |
| `y_new_rk_landcover.npz` (landcover) | |

Sanity check: для 2010–2023 diff с baseline < 0.001°C.

---

## Этап 5. P2 inference (опционально)

Карты MAGT на **реальных** данных для графиков истории 2007–2025.
Использует P2-модель, обученную на 2003–2023 (`convlstm_ttop_extended_21years.pt`).

```bash
python3 scripts/inference_p2_history.py
```

| Выход | Содержимое |
|-------|------------|
| `results/maps/p2_yearly_maps.npz` | 18 карт, 2007–2024 |
| `results/maps/p2_real_2025.npz` | карта 2025 |

Отдельно: `inference_p2_yearly_maps.py` (только 2007–2024) или `inference_p2_2025.py` (только 2025).

> **Не путать с этапом 7:** здесь P2 на реальных данных; в этапе 7 — P3 на synthetic features 2026–2035.

---

## Этап 6. Train P3 ensemble

### Model A — научная (val на 2023–2025)

```bash
python3 scripts/train_p3_model_A_ensemble.py
```

- Train: 2007–2022, Val: 2023, 2024, 2025
- 3 seed → `convlstm_ttop_p3_model_A_seed_*.pt` + `model_A_BEST.pt`

### Model B — production (все данные)

```bash
python3 scripts/train_p3_model_B_ensemble.py
```

- Train: 2007–2025 (без val)
- Для прогноза 2026–2035

### Расширение до 7 моделей (опционально)

```bash
python3 scripts/train_and_ensemble_7models.py
```

Добавляет seeds 7, 99, 777, 2024 к Model A.

---

## Этап 7. Projection 2026–2035

Линейная экстраполяция climate_core признаков + inference P3.

```bash
python3 scripts/projection_2026_2035.py
```

| Выход | Описание |
|-------|----------|
| `data/tensor_01deg_synthetic_2026_2035.npz` | synthetic features |
| `results/maps/p3_projection_2026_2035_modelA.npz` | прогноз Model A |
| `results/maps/p3_projection_2026_2035_modelB.npz` | прогноз Model B |
| `results/metrics/p3_projection_comparison.csv` | сравнение |

---

## Этап 8. Визуализация

```bash
python3 scripts/plot_ensemble6_final.py      # тренд 2007–2035, uncertainty, HTML
python3 scripts/plot_delta_vs_baseline.py    # ΔMAGT vs mean(2018–2024)
python3 scripts/plot_delta_permafrost_only.py
```

**Выход:** `results/figures/p3_ensemble6_*.png`, `results/figures/p3_ensemble6_interactive_2007_2035.html`

---

## Минимальный запуск (без GEE, с готовыми данными)

Если в репозитории уже есть `tensor_23y`, `target_23y`, модели:

```bash
pip install -r requirements.txt
python3 scripts/smoke_test.py

python3 scripts/projection_2026_2035.py
python3 scripts/plot_delta_vs_baseline.py
```

## Полный запуск с нуля

```bash
pip install -r requirements.txt
earthengine authenticate

python3 scripts/gee_export.py --years 2003-2025 --project YOUR_PROJECT
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
```

---

## Baseline v2 (ноутбуки)

Для воспроизведения исходной модели 2010–2023 (P4 ablation, bias correction):

```
notebooks/01 → 02 → 03 → 04 → 05 → 07
```

См. [notebooks/README.md](notebooks/README.md) и [docs/reproducing_results.md](docs/reproducing_results.md).

---

## Диаграмма зависимостей

```
gee_export → merge_bands → rasterize_extended → rasterize_2025
                                    ↓
                         tensor_01deg_extended_23y.npz
                                    ↓
              y_new_rk_landcover.npz → compute_target_23y
                                    ↓
                         y_new_rk_landcover_extended_23y.npz
                                    ↓
                    train_p3_model_A/B_ensemble
                                    ↓
                    maps_lh_v3.npz → projection_2026_2035
                                    ↓
                         p3_projection_*.npz
                                    ↓
              (опционально) inference_p2_history → plot_ensemble6_final
```

---

## Этап 9. Экспорт для GIS (все годы 2007–2035)

Склеивает историю (P2) и прогноз (P3) в один файл и опционально — GeoTIFF на каждый год.

```bash
python3 scripts/export_gis_all_years.py --export-tiff
```

| Выход | Содержимое |
|-------|------------|
| `results/maps/magt_all_years_2007_2035.npz` | `magt` (29, 231, 1501), `years`, `lats`, `lons` |
| `results/gis/MAGT_<year>.tif` | GeoTIFF EPSG:4326 для QGIS / ArcGIS |

По умолчанию прогноз 2026–2035 — **Model B** (`--projection modelA` для научной модели).

---

## Проверка

```bash
python3 scripts/smoke_test.py
```

Ожидается 16/18 PASS (модули `data_extended` и `esa_cci` — только в `archive/`).
