# Данные

## Что лежит в репозитории

В `data/` хранятся только **малые файлы** — преимущественно справочные таблицы:

| Файл/папка | Размер | Описание |
|---|---|---|
| `boreholes/clean_boreholes_33.csv` | < 1 МБ | 33 валидных GTN-P бура с MAGT и координатами |
| `boreholes/lst_44_stations.csv` | < 1 МБ | Расширенный список 44 станций (future work) |
| `external_links.md` | — | Ссылки на скачивание больших данных |

## Большие данные (НЕ в репозитории)

Из-за размера в репозиторий не входят:

| Файл | Размер | Источник |
|---|---|---|
| `tensor_01deg_v2.npz` | 200 МБ | Входной тензор (генерируется скриптами или скачивается) |
| `UiO_PEX_MAGTM_5.0_..._NH.tif` | 1.8 ГБ | Obu et al. 2019 MAGT (оригинал TIFF) |
| `obu_2019_reprojected.npz` | 1.3 МБ | Obu MAGT на нашей сетке (генерируется из TIFF) |
| `ESACCI-...-2023-fv05.0.nc` | 344 МБ | ESA CCI Permafrost v05.0 NetCDF |
| `comparison_esacci_2023.npz` | 4 МБ | ESA T2m на нашей сетке |
| `GEE_exports/` | ~2 ГБ | Папка с 56 .geojson из Google Earth Engine |

**Ссылки на скачивание этих файлов** — в [`external_links.md`](external_links.md).

## Структура входного тензора `tensor_01deg_v2.npz`

```python
import numpy as np
data = np.load('data/tensor_01deg_v2.npz', allow_pickle=True)

data['X']             # (14, 231, 1501, 20)  — признаки
data['y']             # (14, 231, 1501)      — target MAGT по TTOP
data['years']         # [2010, 2011, ..., 2023]
data['lons']          # (1501,)              — массив долгот
data['lats']          # (231,)               — массив широт
data['feature_names'] # ['NDVI', 'NDWI', ..., 'era5_precip']  (20 имён)
```

## 20 признаков

| # | Признак | Источник | Единицы |
|---|---|---|---|
| 0 | NDVI | MODIS MOD13Q1 | безразм. |
| 1 | NDWI | MODIS MOD13Q1 | безразм. |
| 2 | NDMI | MODIS MOD13Q1 | безразм. |
| 3 | SAVI | MODIS MOD13Q1 | безразм. |
| 4 | LST_summer | MODIS MOD11A1 | °C |
| 5 | LST_winter | MODIS MOD11A1 | °C |
| 6 | LST_annual | MODIS MOD11A1 | °C |
| 7 | FDD | MODIS MOD11A1 | К·сут |
| 8 | TDD | MODIS MOD11A1 | К·сут |
| 9 | snow_days | MOD10A1 | дней |
| 10 | elevation | GMTED2010 | м |
| 11 | slope | GMTED2010 | градусы |
| 12 | aspect | GMTED2010 | градусы |
| 13 | soil_oc | SoilGrids 2.0 | кг/м² |
| 14 | soil_bd | SoilGrids 2.0 | г/см³ |
| 15 | soil_clay | SoilGrids 2.0 | % |
| 16 | MAAT | WorldClim v2.1 | °C |
| 17 | MAP | WorldClim v2.1 | мм/год |
| 18 | era5_temp | ERA5-Land | °C |
| 19 | era5_precip | ERA5-Land | мм/год |

**Важно:** FDD и TDD в тензоре лежат в **исходных К·сут**, без нормализации. Это
позволяет напрямую применять формулу TTOP:

```python
y = (rk * X[..., 8] - X[..., 7]) / 365
```

## Параметры сетки

- **Долгота:** 30.0° E → 180.0° E с шагом 0.1° → 1501 точка
- **Широта:** 55.0° N → 78.0° N с шагом 0.1° → 231 точка
- **CRS:** EPSG:4326 (WGS84)
- **Размер пикселя** на 65° N: ≈ 5 × 11 км

## Файл буров `boreholes/clean_boreholes_33.csv`

Колонки:
- `borehole_id` — идентификатор GTN-P
- `name`, `site_name` — название бура
- `lat`, `lon` — координаты
- `max_depth_m`, `depth_used_m` — максимальная глубина измерения
- `magt_c` — измеренная MAGT в °C (на год измерений)
- `year_min`, `year_max`, `mid_year` — годы измерений
- `n_measurements` — количество измерений
- `magt_adjusted` — MAGT с временной коррекцией на тренд +0.085 °C/год до 2023 года

## Проверка структуры

```python
from src.data import load_tensor

data = load_tensor('data/tensor_01deg_v2.npz')
# Выведет shape, годы, имена признаков
```
