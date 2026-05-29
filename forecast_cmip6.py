"""
ПРОГНОЗ МЕРЗЛОТЫ на 2030/2050 через CMIP6.

Логика:
  1. Берём фичи сетки 2021 (реальные).
  2. Прибавляем ΔT (из CMIP6) к ТЕМПЕРАТУРНЫМ фичам:
     MAAT, era5_temp, LST_summer/winter/annual.
     (FDD/TDD пересчитываем приближённо со сдвигом.)
  3. Предсказываем MAGT мультигодовой моделью для базы(2021)/2030/2050.
  4. Карты прогноза + изменение относительно 2021.

ВАЖНЫЕ ОГОВОРКИ (для отчёта):
  - Модель даёт РАВНОВЕСНЫЙ отклик на сдвиг климата, без учёта тепловой
    инерции грунта и латентного тепла таяния льда (замедляет нагрев около 0°C).
  - Прогноз = "если климат потеплеет по SSP2-4.5 на ΔT, мерзлота станет такой".
  - Это сценарная оценка, не точное предсказание.

Запуск:
    python3 forecast_cmip6.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import joblib

REPORTS = Path("reports")
FEAT_DIR = Path("data/features")
CMIP6 = glob.glob(str(FEAT_DIR / "CMIP6_dT_*.geojson"))

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]
# Температурные фичи, к которым прибавляем ΔT
TEMP_SHIFT = ["MAAT", "era5_temp", "LST_summer", "LST_winter", "LST_annual"]


def load_grid_2021():
    with open(FEAT_DIR / "RussiaGrid_0.25deg_2021.geojson") as fh:
        gj = json.load(fh)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lo, la = ft["geometry"]["coordinates"]
        lons.append(lo); lats.append(la)
    df = pd.DataFrame(rows)
    df["lon"] = lons; df["lat"] = lats
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def load_cmip6():
    with open(CMIP6[0]) as fh:
        gj = json.load(fh)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lo, la = ft["geometry"]["coordinates"]
        lons.append(lo); lats.append(la)
    df = pd.DataFrame(rows)
    df["lon"] = lons; df["lat"] = lats
    return df


def attach_dt(grid, cmip):
    """Привязываем ΔT к точкам сетки по ближайшему соседу."""
    tree = cKDTree(cmip[["lon", "lat"]].values)
    d, idx = tree.query(grid[["lon", "lat"]].values, k=1)
    grid["dT_2030"] = cmip["dT_2030"].values[idx]
    grid["dT_2050"] = cmip["dT_2050"].values[idx]
    return grid


def predict_with_shift(df, model, dT_col):
    """Прибавляем ΔT к температурным фичам, предсказываем MAGT."""
    X = df[FEATURE_COLS].copy()
    dT = df[dT_col].values
    for c in TEMP_SHIFT:
        X[c] = X[c] + dT
    # FDD/TDD приближённо: потепление уменьшает FDD, увеличивает TDD
    # грубо: сдвиг на dT градусов * ~半 года в днях (180) — порядковая поправка
    X["FDD"] = np.maximum(X["FDD"] - dT * 180, 0)
    X["TDD"] = X["TDD"] + dT * 180
    return model.predict(X[FEATURE_COLS].values)


def raster(sub, col, res=0.25, md=0.4, ref_magt=None):
    lon_g = np.arange(30, 180, res); lat_g = np.arange(55, 78, res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    ss = sub.dropna(subset=[col])
    Z = griddata((ss["lon"], ss["lat"]), ss[col], (LON, LAT), method="linear")
    tree = cKDTree(np.column_stack([ss["lon"], ss["lat"]]))
    d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
    Z = np.where(d.reshape(LON.shape) <= md, Z, np.nan)
    return LON, LAT, Z


def main():
    if not CMIP6:
        print("Нет файла CMIP6_dT_*.geojson в data/features/")
        return
    model = joblib.load("models_multiyear_xgb.joblib")
    grid = load_grid_2021()
    cmip = load_cmip6()
    print(f"Сетка: {len(grid)}, CMIP6: {len(cmip)}")
    grid = attach_dt(grid, cmip)
    print(f"ΔT привязан. Среднее dT_2030={grid['dT_2030'].mean():.2f}, dT_2050={grid['dT_2050'].mean():.2f}")

    # Предсказания
    grid["magt_2021"] = model.predict(grid[FEATURE_COLS].values)
    grid["magt_2030"] = predict_with_shift(grid, model, "dT_2030")
    grid["magt_2050"] = predict_with_shift(grid, model, "dT_2050")
    grid["change_2030"] = grid["magt_2030"] - grid["magt_2021"]
    grid["change_2050"] = grid["magt_2050"] - grid["magt_2021"]

    # Статистика
    print("\n=== ПРОГНОЗ МЕРЗЛОТЫ (SSP2-4.5) ===")
    print(f"MAGT 2021 (база): среднее {grid['magt_2021'].mean():.2f}°C")
    print(f"MAGT 2030: среднее {grid['magt_2030'].mean():.2f}°C (потепление {grid['change_2030'].mean():+.2f}°C)")
    print(f"MAGT 2050: среднее {grid['magt_2050'].mean():.2f}°C (потепление {grid['change_2050'].mean():+.2f}°C)")

    # Площадь мерзлоты (MAGT < 0)
    pf_2021 = (grid["magt_2021"] < 0).mean() * 100
    pf_2030 = (grid["magt_2030"] < 0).mean() * 100
    pf_2050 = (grid["magt_2050"] < 0).mean() * 100
    print(f"\nДоля территории с мерзлотой (MAGT<0):")
    print(f"  2021: {pf_2021:.1f}%")
    print(f"  2030: {pf_2030:.1f}%")
    print(f"  2050: {pf_2050:.1f}%")
    print(f"  Потеря мерзлоты к 2050: {pf_2021-pf_2050:.1f} проц. пунктов")

    # Карты прогноза изменения
    for yr, col, vmax in [("2030", "change_2030", 1.5), ("2050", "change_2050", 3.0)]:
        LON, LAT, Z = raster(grid, col)
        fig, ax = plt.subplots(figsize=(16, 7))
        im = ax.pcolormesh(LON, LAT, Z, cmap="Reds", vmin=0, vmax=vmax, shading="auto")
        ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
        ax.set_title(f"Прогноз потепления мерзлоты к {yr} (CMIP6 SSP2-4.5)\nΔMAGT относительно 2021, °C")
        plt.colorbar(im, ax=ax, label="ΔMAGT, °C", shrink=0.7)
        ax.set_aspect(2.0); ax.grid(alpha=0.15)
        plt.tight_layout()
        plt.savefig(REPORTS / f"forecast_{yr}.png", dpi=140)
        plt.close()
        print(f"Сохранено: forecast_{yr}.png")

    # Карта MAGT 2050 (абсолютная)
    LON, LAT, Z = raster(grid, "magt_2050")
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.pcolormesh(LON, LAT, Z, cmap="coolwarm_r", vmin=-12, vmax=2, shading="auto")
    ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
    ax.set_title("Прогноз MAGT на 2050 (CMIP6 SSP2-4.5)")
    plt.colorbar(im, ax=ax, label="MAGT, °C", shrink=0.7)
    ax.set_aspect(2.0); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(REPORTS / "forecast_magt_2050.png", dpi=140)
    plt.close()
    print("Сохранено: forecast_magt_2050.png")

    grid[["lon", "lat", "magt_2021", "magt_2030", "magt_2050",
          "change_2030", "change_2050"]].to_csv(REPORTS / "forecast_data.csv", index=False)
    print("\nДанные: forecast_data.csv")


if __name__ == "__main__":
    main()