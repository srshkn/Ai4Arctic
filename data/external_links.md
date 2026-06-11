# Данные — что в репозитории, что скачать, что генерировать

Большие файлы (>100 МБ) не помещаются в GitHub. Ниже — три категории.

---

## 1. Уже в репозитории (v3 pipeline)

Эти файлы включены в git (см. исключения в `.gitignore`). **Скачивать не нужно.**

| Файл | Размер | Назначение |
|------|--------|------------|
| `data/y_new_rk_landcover.npz` | ~20 МБ | `landcover`, `rk_map` (baseline v2) |
| `data/maps_lh_v3.npz` | ~4 МБ | `delta_t` для lh-коррекции |
| `data/obu_2019_reprojected.npz` | ~1.3 МБ | Эталон Obu на сетке 0.1° |
| `data/y_new_rk_landcover_extended_23y.npz` | ~18 МБ | TTOP target 2003–2025 |
| `data/tensor_01deg_extended_23y.npz` | ~312 МБ | Входной тензор X (23 года) |
| `data/boreholes/clean_boreholes_33.csv` | <1 МБ | 33 бура GTN-P |

Модели в `models/` (чекпоинты `.pt`) — см. [models/README.md](../models/README.md).

---

## 2. Генерируются скриптами (не скачивать)

Запускайте [PIPELINE.md](../PIPELINE.md). Эти файлы создаются локально:

| Файл | Этап | Скрипт |
|------|------|--------|
| `data/gee/*.geojson` | 1–2 | `gee_export.py`, `merge_bands.py` |
| `data/tensor_01deg_extended_22y.npz` | 3a | `rasterize_extended.py` |
| `data/tensor_01deg_extended_23y.npz` | 3b | `rasterize_2025.py` |
| `data/y_new_rk_landcover_extended_23y.npz` | 4 | `compute_target_23y.py` |
| `data/tensor_01deg_synthetic_2026_2035.npz` | 7 | `projection_2026_2035.py` |
| `results/maps/p2_yearly_maps.npz` | 5 | `inference_p2_yearly_maps.py` |
| `results/maps/p2_real_2025.npz` | 5 | `inference_p2_2025.py` |
| `results/maps/p3_projection_*.npz` | 7 | `projection_2026_2035.py` |

---

## 3. Опционально: скачать вместо GEE export

Если нет доступа к Google Earth Engine, можно скачать готовый тензор.
**Разместите архив на Zenodo / Google Drive и добавьте ссылку ниже.**

### Входной тензор v3 (23 года)

| | |
|---|---|
| **Файл** | `tensor_01deg_extended_23y.npz` (~312 МБ) |
| **Содержимое** | X (23, 231, 1501, 20), years 2003–2025 |
| **Положить в** | `data/tensor_01deg_extended_23y.npz` |
| **Скачать** | *TODO: добавить ссылку Zenodo / Drive после публикации* |

```bash
mkdir -p data
# wget -O data/tensor_01deg_extended_23y.npz "ССЫЛКА"
```

### TTOP target v3 (если нет в git)

| | |
|---|---|
| **Файл** | `y_new_rk_landcover_extended_23y.npz` (~18 МБ) |
| **Положить в** | `data/y_new_rk_landcover_extended_23y.npz` |
| **Скачать** | *TODO: добавить ссылку* |
| **Альтернатива** | `python3 scripts/compute_target_23y.py` (нужен тензор 23y + `y_new_rk_landcover.npz`) |

---

## 4. Baseline v2 (ноутбуки 01–07)

Нужны только для воспроизведения **исходной** модели 2010–2023.

### Входной тензор v2

| | |
|---|---|
| **Файл** | `tensor_01deg_v2.npz` (~200 МБ) |
| **Содержимое** | X (14, 231, 1501, 20), years 2010–2023 |
| **Положить в** | `data/tensor_01deg_v2.npz` |
| **Скачать** | *TODO: добавить ссылку Zenodo / Drive* |
| **Альтернатива** | Собрать через GEE + растеризацию (см. `docs/data_pipeline.md`) |

### Эталон ESA CCI (на нашей сетке)

| | |
|---|---|
| **Файл** | `comparison_esacci_2023.npz` (~4 МБ) |
| **Положить в** | `data/comparison_esacci_2023.npz` |
| **Скачать** | *TODO: добавить ссылку* |

### Модель v2 / P4

| | |
|---|---|
| **Файл** | `models/convlstm_ttop_rk_v2_climate_core_6.pt` |
| **В репозитории** | Да (P4 ablation winner) |
| **Старое имя в ноутбуках** | `convlstm_ttop_rk_v2.pt` — тот же чекпоинт, другое имя |

---

## 5. Исходники (воспроизведение эталонов с нуля)

### Obu et al. 2019 MAGT — оригинальный TIFF

| | |
|---|---|
| **Скачать** | https://store.pangaea.de/Publications/ObuJ-etal_2018/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.zip |
| **Положить в** | `data/raw/Obu/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif` |
| **Перепроекция** | `python3 scripts/reproject_obu.py` → `obu_2019_reprojected.npz` |
| **Лицензия** | CC-BY 4.0 |

### ESA CCI Permafrost v05.0

| | |
|---|---|
| **Каталог** | https://catalogue.ceda.ac.uk/ |
| **Поиск** | «ESA CCI Permafrost» / CryoGrid Area 4 |
| **Положить в** | `data/raw/ESA_CCI/` |
| **Лицензия** | ESA CCI Data Policy |

### GEE raw exports (если не запускаете `gee_export.py`)

| | |
|---|---|
| **Папка на Drive** | `GEE_exports_features_01deg_v2_extended/` |
| **Скачать** | *TODO: ссылка на Drive после публикации* |
| **Положить в** | `data/gee/` |
| **Далее** | `merge_bands.py` → `rasterize_extended.py` |

---

## Проверка после настройки

### Pipeline v3 (скрипты)

```bash
ls -lh data/tensor_01deg_extended_23y.npz
ls -lh data/y_new_rk_landcover_extended_23y.npz
ls -lh data/y_new_rk_landcover.npz
ls -lh data/maps_lh_v3.npz
ls -lh models/convlstm_ttop_p3_model_A_BEST.pt
ls -lh models/convlstm_ttop_p3_model_B_BEST.pt
```

Если всё на месте:

```bash
python3 scripts/projection_2026_2035.py
```

### Baseline v2 (ноутбуки)

```bash
ls -lh data/tensor_01deg_v2.npz                    # или пропустить, если только v3
ls -lh data/obu_2019_reprojected.npz
ls -lh data/boreholes/clean_boreholes_33.csv
ls -lh models/convlstm_ttop_rk_v2_climate_core_6.pt
```

---

## Публикация данных (для авторов репозитория)

Рекомендуется Zenodo (DOI для диплома):

1. Загрузить `tensor_01deg_extended_23y.npz` (+ опционально v2 тензор)
2. Вставить прямую ссылку в раздел 3 выше
3. Указать DOI в README и PIPELINE.md

Альтернативы: Google Drive (доступ по ссылке), Yandex.Disk, Hugging Face Datasets.
