"""
Карта тренда на МУЛЬТИГОДОВОЙ модели (models_multiyear_xgb.joblib).

Эта модель даёт пространственно достоверный паттерн тренда (корреляция с
ESA CCI r=0.62 против 0.02 у статичной). Строим итоговую карту тренда +
карту изменения 2010->2023.

Запуск:
    python3 build_trend_multiyear.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
import joblib

REPORTS = Path("reports")
FEAT_DIR = Path("data/features")
ALL_YEARS = [2010, 2013, 2015, 2017, 2019, 2021, 2023]

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_grid(year):
    with open(FEAT_DIR / f"RussiaGrid_0.25deg_{year}.geojson") as fh:
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
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    df["k"] = df["lon"].round(2).astype(str) + "_" + df["lat"].round(2).astype(str)
    return df.drop_duplicates(subset="k")


def raster(sub, col, res=0.25, md=0.4, smooth=True, permafrost_ref=None):
    lon_g = np.arange(30, 180, res); lat_g = np.arange(55, 78, res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    ss = sub.dropna(subset=[col])
    Z = griddata((ss["lon"], ss["lat"]), ss[col], (LON, LAT), method="linear")
    tree = cKDTree(np.column_stack([ss["lon"], ss["lat"]]))
    d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
    Z = np.where(d.reshape(LON.shape) <= md, Z, np.nan)
    if smooth:
        m = ~np.isnan(Z)
        num = uniform_filter(np.where(m, Z, 0), size=3)
        cnt = uniform_filter(m.astype(float), size=3)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = np.where(m, num/cnt, np.nan)
    if permafrost_ref is not None:
        Z = np.where(permafrost_ref < 0, Z, np.nan)
    return LON, LAT, Z


def main():
    model = joblib.load("models_multiyear_xgb.joblib")
    print("Мультигодовая модель загружена")

    preds = {}
    for y in ALL_YEARS:
        df = load_grid(y)
        df[f"magt_{y}"] = model.predict(df[FEATURE_COLS].values)
        preds[y] = df[["k", "lon", "lat", f"magt_{y}"]].set_index("k")
        print(f"{y}: {len(df)} точек")

    # OUTER join — берём ВСЕ точки, где есть хотя бы несколько лет
    common = preds[ALL_YEARS[0]]
    for y in ALL_YEARS[1:]:
        common = common.join(preds[y].drop(columns=["lon", "lat"]), how="outer")
    if common["lon"].isna().any():
        for y in ALL_YEARS:
            common["lon"] = common["lon"].fillna(preds[y]["lon"])
            common["lat"] = common["lat"].fillna(preds[y]["lat"])
    print(f"Всего точек (outer): {len(common)}")

    magt_cols = [f"magt_{y}" for y in ALL_YEARS]
    yrs = np.array(ALL_YEARS, dtype=float)

    # Тренд по ДОСТУПНЫМ годам в каждой точке (минимум 4 года)
    M = common[magt_cols].values
    trends = np.full(len(common), np.nan)
    for i in range(len(common)):
        mask = ~np.isnan(M[i])
        if mask.sum() >= 4:
            trends[i] = np.polyfit(yrs[mask], M[i][mask], 1)[0]
    common["trend"] = trends
    common["dmagt"] = common["magt_2023"] - common["magt_2010"]
    common["magt_mean"] = np.nanmean(M, axis=1)
    common = common[~np.isnan(common["trend"])]
    print(f"Точек с трендом (>=4 года): {len(common)}")

    print(f"\nСредний тренд: {common['trend'].mean()*10:+.3f} °C/декада")
    print(f"Изменение 2010-2023: {common['dmagt'].mean():+.3f} °C")
    print(f"Доля потепления: {100*(common['trend']>0).mean():.0f}%")

    # Карта тренда (сглаженная, достоверная)
    LON, LAT, TR = raster(common, "trend")
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.pcolormesh(LON, LAT, TR*10, cmap="RdBu_r", vmin=-0.8, vmax=0.8, shading="auto")
    ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
    ax.set_title("Тренд температуры мерзлоты 2010-2023 (°C/декада)\nмультигодовая модель, паттерн согласован с ESA CCI (r=0.62)")
    plt.colorbar(im, ax=ax, label="°C/декада", shrink=0.7)
    ax.set_aspect(2.0); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(REPORTS / "trend_final.png", dpi=140)
    plt.close()
    print("\nСохранено: trend_final.png")

    # Карта изменения 2010->2023 (в зоне мерзлоты)
    ref = common["magt_mean"].values  # для маски мерзлоты
    LON, LAT, DZ = raster(common, "dmagt")
    # маска мерзлоты по среднему MAGT
    LONm, LATm, MEANMAGT = raster(common, "magt_mean", smooth=False)
    DZ = np.where(MEANMAGT < 1, DZ, np.nan)  # показываем где мерзлота/около
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.pcolormesh(LON, LAT, DZ, cmap="RdBu_r", vmin=-1.5, vmax=1.5, shading="auto")
    ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
    ax.set_title("Изменение температуры мерзлоты 2010→2023 (°C)\nмультигодовая модель")
    plt.colorbar(im, ax=ax, label="ΔMAGT, °C", shrink=0.7)
    ax.set_aspect(2.0); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(REPORTS / "change_final.png", dpi=140)
    plt.close()
    print("Сохранено: change_final.png")

    common.reset_index()[["lon", "lat", "trend", "dmagt", "magt_mean"]].to_csv(
        REPORTS / "trend_final_data.csv", index=False)
    print("Данные: trend_final_data.csv")


if __name__ == "__main__":
    main()