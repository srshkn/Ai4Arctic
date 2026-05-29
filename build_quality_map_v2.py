"""
КАЧЕСТВЕННАЯ КАРТА мерзлоты России (надёжная версия).

Не зависит от cartopy и скачивания контуров (которые падали по SSL).
Контур/маска суши строится ПО САМИМ ДАННЫМ: оставляем только пиксели сетки,
рядом с которыми есть исходные точки -> убираем артефакты интерполяции в океане.

Слои: MAGT, классы, ALT.

Запуск:
    python3 build_quality_map_v2.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
MAP_CSV = REPORTS / "russia_map_data.csv"
GRID_GEOJSON = glob.glob("data/features/RussiaGrid_*.geojson")

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def add_alt(df):
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
    df = df.merge(g[["lon", "lat", "pred_alt"]], on=["lon", "lat"], how="left")
    return df


def interp_masked(df, value_col, res=0.2, mask_dist_deg=0.45,
                  permafrost_only=False):
    """
    Интерполирует в регулярную сетку И маскирует:
      1) ячейки, далёкие от исходных точек (mask_dist_deg) — артефакты в океане;
      2) если permafrost_only=True — ячейки БЕЗ мерзлоты (pred_magt >= 0),
         нужно для ALT: глубина протаивания определена только в мерзлоте.
    mask_dist_deg увеличен до 0.45 — убирает вертикальные белые полосы (разрывы сетки).
    """
    sub = df.dropna(subset=[value_col])
    lon_g = np.arange(df["lon"].min(), df["lon"].max(), res)
    lat_g = np.arange(df["lat"].min(), df["lat"].max(), res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    Z = griddata((sub["lon"], sub["lat"]), sub[value_col], (LON, LAT), method="linear")

    # Маска 1: расстояние до ближайшей исходной точки
    tree = cKDTree(np.column_stack([sub["lon"], sub["lat"]]))
    dist, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
    dist = dist.reshape(LON.shape)
    Z = np.where(dist <= mask_dist_deg, Z, np.nan)

    # Маска 2 (для ALT): только там, где есть мерзлота (MAGT < 0)
    if permafrost_only and "pred_magt" in df.columns:
        magt_sub = df.dropna(subset=["pred_magt"])
        MAGT = griddata((magt_sub["lon"], magt_sub["lat"]), magt_sub["pred_magt"],
                        (LON, LAT), method="linear")
        Z = np.where(MAGT < 0, Z, np.nan)  # нет мерзлоты -> ALT не показываем

    return LON, LAT, Z


def draw_map(LON, LAT, Z, title, cmap, vmin, vmax, fname,
             cbar_label=None, class_ticks=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.pcolormesh(LON, LAT, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
    ax.set_title(title, fontsize=13)
    ax.set_aspect(2.0)  # широты сжаты — компенсируем для похожести на карту
    ax.grid(alpha=0.15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    if class_ticks:
        cbar.set_ticks(list(class_ticks.keys()))
        cbar.ax.set_yticklabels(list(class_ticks.values()))
    elif cbar_label:
        cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.savefig(REPORTS / fname, dpi=150)
    plt.close()
    print(f"Сохранено: {fname}")


def main():
    df = pd.read_csv(MAP_CSV)
    print(f"Точек: {len(df)}")
    df = add_alt(df)
    has_alt = df["pred_alt"].notna().any()
    print(f"ALT досчитана: {has_alt}")

    # MAGT
    LON, LAT, Z = interp_masked(df, "pred_magt")
    draw_map(LON, LAT, Z,
             "Среднегодовая температура грунта (MAGT, 10 м) — Россия, 2021",
             "coolwarm_r", -12, 2, "map_magt_quality.png", cbar_label="MAGT, °C")

    # Классы
    LON, LAT, Zc = interp_masked(df, "pred_class")
    cmap3 = mcolors.ListedColormap(["#d7191c", "#fdae61", "#2c7bb6"])
    draw_map(LON, LAT, Zc,
             "Классы состояния мерзлоты — Россия, 2021",
             cmap3, -0.5, 2.5, "map_classes_quality.png",
             class_ticks={0: "нет/тёплая", 1: "деградирующая", 2: "устойчивая"})

    # ALT (только в зоне мерзлоты — где MAGT < 0)
    if has_alt:
        LON, LAT, Za = interp_masked(df, "pred_alt", permafrost_only=True)
        draw_map(LON, LAT, Za,
                 "Глубина протаивания (ALT) в зоне мерзлоты — Россия, 2021",
                 "YlOrBr", 0, 3, "map_alt_quality.png", cbar_label="ALT, м")

    # Экспорт для Kepler.gl / QGIS
    out = df[["lon", "lat", "pred_magt", "pred_class"]].copy()
    if has_alt:
        out["pred_alt"] = df["pred_alt"]
    out.to_csv(REPORTS / "russia_map_full.csv", index=False)
    print(f"\nДанные для Kepler.gl/QGIS: {REPORTS / 'russia_map_full.csv'}")


if __name__ == "__main__":
    main()