"""
Сэмплирование ESA CCI v5.0 (T10m) как target к разбросанным точкам.

Что делает:
  1. Читает 2198 точек с фичами (RussiaScatter).
  2. Открывает ESA CCI NetCDF, сэмплирует T10m в координаты точек.
  3. Фильтрует: убирает точки без мерзлоты (NaN) и с высокой неопределённостью.
  4. Сохраняет обучающий датасет: фичи + target_magt (T10m) + uncertainty.

ESA CCI lat/lon — обычная регулярная сетка, сэмплируем через xarray .sel(nearest).

Запуск:
    python3 sample_esacci_target.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr

POINTS_GEOJSON = Path("data/features/RussiaScatter_3000pts_2021.geojson")
ESACCI_DIR = Path("data/esa_cci")
OUT = Path("data/features/scatter_training_dataset.parquet")

# Глубина target: T10m — стабильная зона нулевой годовой амплитуды
TARGET_VAR = "T10m"
UNCERT_VAR = "T10m_uncertainty"
MAX_UNCERTAINTY = 3.0   # °C — выкидываем точки где ESA CCI сама не уверена

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_points():
    with open(POINTS_GEOJSON) as f:
        gj = json.load(f)
    rows = []
    for ft in gj["features"]:
        props = dict(ft["properties"])
        geom = ft.get("geometry")
        if geom is None:
            continue
        lon, lat = geom["coordinates"]
        props["lon"] = lon
        props["lat"] = lat
        rows.append(props)
    df = pd.DataFrame(rows)
    print(f"Точек загружено: {len(df)}")
    return df


def sample_esacci(df):
    files = glob.glob(str(ESACCI_DIR / "*.nc"))
    if not files:
        raise FileNotFoundError(f"Нет .nc в {ESACCI_DIR}")
    print(f"Открываю: {files[0]}")
    ds = xr.open_dataset(files[0])

    # Выбираем нужные переменные, убираем измерение time (оно = 1)
    t10 = ds[TARGET_VAR].isel(time=0)
    unc = ds[UNCERT_VAR].isel(time=0)

    # Векторное сэмплирование nearest neighbor
    lats = xr.DataArray(df["lat"].values, dims="points")
    lons = xr.DataArray(df["lon"].values, dims="points")

    print("Сэмплирую T10m в точках...")
    df["target_magt"] = t10.sel(lat=lats, lon=lons, method="nearest").values
    df["target_uncertainty"] = unc.sel(lat=lats, lon=lons, method="nearest").values

    return df


def main():
    df = load_points()
    df = sample_esacci(df)

    n0 = len(df)
    # 1. Убираем точки без мерзлоты (NaN в target)
    df = df.dropna(subset=["target_magt"])
    print(f"После удаления NaN target (нет мерзлоты): {len(df)}/{n0}")

    # 2. Убираем точки с высокой неопределённостью ESA CCI
    n1 = len(df)
    df = df[df["target_uncertainty"].fillna(99) <= MAX_UNCERTAINTY]
    print(f"После фильтра uncertainty <= {MAX_UNCERTAINTY}: {len(df)}/{n1}")

    # 3. Убираем точки с NaN в фичах (на всякий)
    n2 = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"После удаления NaN в фичах: {len(df)}/{n2}")

    print(f"\nИТОГО обучающих точек: {len(df)}")
    print(f"Target (T10m) диапазон: {df['target_magt'].min():.2f} ... {df['target_magt'].max():.2f} °C")
    print(f"Target среднее: {df['target_magt'].mean():.2f} °C, std: {df['target_magt'].std():.2f}")

    # Распределение по классам (для информации)
    def to_class(v):
        if v > 0: return 0
        if v > -1.5: return 1
        return 2
    classes = df["target_magt"].apply(to_class)
    print("\nБаланс классов:")
    for c, name in [(0, "нет/тёплая >0"), (1, "деградирующая -1.5..0"), (2, "устойчивая <-1.5")]:
        n = (classes == c).sum()
        print(f"  Класс {c} ({name}): {n} ({100*n/len(df):.1f}%)")

    df.to_parquet(OUT)
    print(f"\nСохранено: {OUT}")


if __name__ == "__main__":
    main()