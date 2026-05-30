"""
КАЧЕСТВЕННАЯ КАРТА мерзлоты России.

Превращает разбросанные предсказания в сплошную карту (интерполяция на регулярную
сетку), строит несколько слоёв:
  1. MAGT (среднегодовая температура грунта)
  2. ALT (глубина протаивания)
  3. Классы мерзлоты (нет/деградирующая/устойчивая)
  4. Зоны по вероятности (континуальная/прерывистая/островная — по порогам MAGT)

Работает с cartopy (если установлен) ИЛИ без него (geopandas fallback).

Вход: reports/russia_map_data.csv (lon,lat,pred_magt,pred_class) — от build_map.py
Плюс пересчитывает ALT, если есть models_alt_xgb.joblib и сетка фич.

Запуск:
    python3 build_quality_map.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
MAP_CSV = REPORTS / "russia_map_data.csv"
GRID_GEOJSON = glob.glob("data/features/RussiaGrid_*.geojson")

# Пытаемся подключить cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def add_alt(df):
    """Если есть ALT-модель и фичи в сетке — досчитываем ALT."""
    alt_model = Path("models_alt_xgb.joblib")
    if not alt_model.exists() or not GRID_GEOJSON:
        df["pred_alt"] = np.nan
        return df
    import joblib
    with open(GRID_GEOJSON[0]) as f:
        gj = json.load(f)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lon, lat = ft["geometry"]["coordinates"]
        lons.append(lon); lats.append(lat)
    g = pd.DataFrame(rows); g["lon"] = lons; g["lat"] = lats
    g = g.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    model = joblib.load(alt_model)
    g["pred_alt"] = model.predict(g[FEATURE_COLS].values)
    # мерджим ALT к df по координатам
    df = df.merge(g[["lon", "lat", "pred_alt"]], on=["lon", "lat"], how="left")
    return df


def interpolate_to_grid(df, value_col, res=0.25):
    """Интерполирует разбросанные значения в регулярную сетку для сплошной заливки."""
    lon_grid = np.arange(df["lon"].min(), df["lon"].max(), res)
    lat_grid = np.arange(df["lat"].min(), df["lat"].max(), res)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    sub = df.dropna(subset=[value_col])
    Z = griddata(
        (sub["lon"], sub["lat"]), sub[value_col],
        (LON, LAT), method="linear"
    )
    return LON, LAT, Z


def setup_ax(ax, title):
    if HAS_CARTOPY:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_extent([28, 182, 53, 79], crs=ccrs.PlateCarree())
    ax.set_title(title, fontsize=13)


def main():
    df = pd.read_csv(MAP_CSV)
    print(f"Точек: {len(df)}")
    df = add_alt(df)
    has_alt = df["pred_alt"].notna().any()
    print(f"ALT досчитана: {has_alt}")
    print(f"cartopy: {'есть' if HAS_CARTOPY else 'нет (geopandas fallback)'}")

    proj = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}

    # === КАРТА 1: MAGT (сплошная) ===
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, **proj)
    LON, LAT, Z = interpolate_to_grid(df, "pred_magt")
    im = ax.pcolormesh(LON, LAT, Z, cmap="coolwarm_r", vmin=-12, vmax=2, shading="auto",
                       **({"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}))
    setup_ax(ax, "Среднегодовая температура грунта (MAGT, 10 м) — Россия, 2021")
    plt.colorbar(im, ax=ax, label="MAGT, °C", shrink=0.7)
    if not HAS_CARTOPY:
        ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
    plt.tight_layout()
    plt.savefig(REPORTS / "map_magt_quality.png", dpi=150)
    plt.close()
    print("Сохранено: map_magt_quality.png")

    # === КАРТА 2: классы (зоны мерзлоты) ===
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1, **proj)
    cmap3 = mcolors.ListedColormap(["#d7191c", "#fdae61", "#2c7bb6"])  # тёплая->устойчивая
    LON, LAT, Zc = interpolate_to_grid(df, "pred_class")
    im = ax.pcolormesh(LON, LAT, Zc, cmap=cmap3, vmin=-0.5, vmax=2.5, shading="auto",
                       **({"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}))
    setup_ax(ax, "Классы состояния мерзлоты — Россия, 2021")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.7)
    cbar.ax.set_yticklabels(["нет/тёплая", "деградирующая", "устойчивая"])
    if not HAS_CARTOPY:
        ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
    plt.tight_layout()
    plt.savefig(REPORTS / "map_classes_quality.png", dpi=150)
    plt.close()
    print("Сохранено: map_classes_quality.png")

    # === КАРТА 3: ALT (если есть) ===
    if has_alt:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 1, 1, **proj)
        LON, LAT, Za = interpolate_to_grid(df, "pred_alt")
        im = ax.pcolormesh(LON, LAT, Za, cmap="YlOrBr", vmin=0, vmax=3, shading="auto",
                           **({"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}))
        setup_ax(ax, "Глубина протаивания (ALT) — Россия, 2021")
        plt.colorbar(im, ax=ax, label="ALT, м", shrink=0.7)
        if not HAS_CARTOPY:
            ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
        plt.tight_layout()
        plt.savefig(REPORTS / "map_alt_quality.png", dpi=150)
        plt.close()
        print("Сохранено: map_alt_quality.png")

    # === Экспорт для Kepler.gl / QGIS ===
    out = df[["lon", "lat", "pred_magt", "pred_class"]].copy()
    if has_alt:
        out["pred_alt"] = df["pred_alt"]
    out.to_csv(REPORTS / "russia_map_full.csv", index=False)
    print(f"\nДанные для Kepler.gl/QGIS: {REPORTS / 'russia_map_full.csv'}")
    print("(загрузите этот CSV в kepler.gl для интерактивной карты)")


if __name__ == "__main__":
    main()