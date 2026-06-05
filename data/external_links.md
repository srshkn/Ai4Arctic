# Скачивание больших данных

Большие файлы не входят в репозиторий из-за лимитов GitHub (100 МБ на файл).
Скачайте их по ссылкам ниже и положите в указанные папки.

---

## Обязательно (для запуска ноутбуков)

### Входной тензор

**Файл:** `tensor_01deg_v2.npz` (~200 МБ)
**Скачать:** [ВСТАВИТЬ_ССЫЛКУ_GOOGLE_DRIVE_ИЛИ_ZENODO]
**Положить в:** `data/tensor_01deg_v2.npz`

Содержит признаки X (14×231×1501×20), target y, метаданные.

```bash
# Пример загрузки через wget (если ссылка прямая):
mkdir -p data
wget -O data/tensor_01deg_v2.npz "[ВСТАВИТЬ_ССЫЛКУ]"
```

### Эталон Obu 2019 (на нашей сетке)

**Файл:** `obu_2019_reprojected.npz` (1.3 МБ)
**Скачать:** [ВСТАВИТЬ_ССЫЛКУ]
**Положить в:** `data/obu_2019_reprojected.npz`

Уже перепроецированная карта Obu MAGT на сетку 0.1°. Файл малый — теоретически
может быть включён в репозиторий, но логически принадлежит к категории «данные».

### Эталон ESA CCI (на нашей сетке)

**Файл:** `comparison_esacci_2023.npz` (~4 МБ)
**Скачать:** [ВСТАВИТЬ_ССЫЛКУ]
**Положить в:** `data/comparison_esacci_2023.npz`

ESA CCI T2m 2023, переинтерполированный на нашу сетку 0.1°.

---

## Опционально (для полного воспроизведения с нуля)

Эти файлы нужны, только если вы хотите **с нуля** построить тензор и эталоны
(а не использовать готовые).

### Obu et al. 2019 MAGT — оригинальный TIFF

**Файл:** `UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif` (1.8 ГБ)
**Скачать:** https://store.pangaea.de/Publications/ObuJ-etal_2018/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.zip
**Положить в:** `data/raw/Obu/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif`
**Лицензия:** CC-BY 4.0
**Цитирование:** Obu et al. (2019), *Earth-Science Reviews*, 193, 299–316.

### ESA CCI Permafrost — оригинальный NetCDF

**Файл:** `ESACCI-PERMAFROST-L4-GTD-MODISLST_CRYOGRID-AREA4_PP-2023-fv05.0.nc` (344 МБ)
**Скачать:** https://catalogue.ceda.ac.uk/uuid/[нужен_конкретный_UUID]
**Положить в:** `data/raw/ESA_CCI/`
**Лицензия:** ESA CCI Data Policy

### Выгрузки из Google Earth Engine

**Папка:** `GEE_exports_features_01deg_v2/` (~2 ГБ, 56 файлов .geojson)
**Скачать:** [ВСТАВИТЬ_ССЫЛКУ_DRIVE]
**Положить в:** `data/raw/GEE_exports/`

Эти файлы получены через ноутбук Google Earth Engine. Скрипт выгрузки (для повторения)
будет добавлен отдельно.

---

## Проверка после скачивания

После размещения файлов проверьте структуру:

```bash
cd permafrost-russia-convlstm

# Проверка минимально необходимых файлов
ls -lh data/tensor_01deg_v2.npz                # 200 МБ
ls -lh data/obu_2019_reprojected.npz           # 1.3 МБ
ls -lh data/comparison_esacci_2023.npz         # 4 МБ
ls -lh data/boreholes/clean_boreholes_33.csv   # < 1 МБ
ls -lh models/convlstm_ttop_rk_v2.pt           # 0.5 МБ
```

Если всё на месте — можно запускать `notebooks/01_data_preparation.ipynb`.

---

## Альтернативы хостингам

Большие файлы могут быть размещены на:

- **Zenodo** (рекомендуется для научных работ): https://zenodo.org/ — выдаёт DOI
- **Google Drive** — простой вариант, требует «доступ по ссылке»
- **Yandex.Disk** — для российских пользователей
- **Hugging Face Datasets** — для ML-датасетов

При размещении укажите соответствующие ссылки в этом файле.
