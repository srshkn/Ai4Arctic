"""
ГИС-КАРТЫ публикационного качества.

Сплошная заливка (интерполяция) + настоящий контур России поверх + светлый фон.
Строит слои: MAGT, зоны мерзлоты, уязвимость, прогноз 2050.

Требует: data/russia_border.geojson (скачан), reports/classification_data.csv,
         reports/forecast_data.csv

Запуск:
    python3 build_gis_maps.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

REPORTS = Path("reports")
BORDER = Path("data/russia_border.geojson")


def load_border_paths():
    """Загружает полигоны России как список путей (для контура и маски)."""
    with open(BORDER) as f:
        gj = json.load(f)
    polys = []
    for ft in gj["features"]:
        geom = ft["geometry"]
        if geom["type"] == "Polygon":
            polys.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            for p in geom["coordinates"]:
                polys.append(p[0])
    return polys


def make_mask(LON, LAT, polys):
    """Маска: точки внутри полигонов России."""
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        # нормализуем долготу полигона в [ -180,180 ] уже ок
        path = MplPath(np.array(poly))
        inside |= path.contains_points(pts)
    return inside.reshape(LON.shape)


def interp(df, col, res=0.15, mask_dist=0.4):
    sub = df.dropna(subset=[col])
    lon_g = np.arange(28, 182, res)
    lat_g = np.arange(53, 79, res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    Z = griddata((sub["lon"], sub["lat"]), sub[col], (LON, LAT), method="linear")
    tree = cKDTree(np.column_stack([sub["lon"], sub["lat"]]))
    d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
    Z = np.where(d.reshape(LON.shape) <= mask_dist, Z, np.nan)
    return LON, LAT, Z


def draw_gis(df, col, title, fname, cmap, vmin, vmax, label="",
             polys=None, discrete=None):
    LON, LAT, Z = interp(df, col)
    # маска по контуру России (убирает заезды в океан/соседей)
    if polys is not None:
        m = make_mask(LON, LAT, polys)
        Z = np.where(m, Z, np.nan)

    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_facecolor("#f7f9fb")

    im = ax.pcolormesh(LON, LAT, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    # Контур России поверх
    if polys is not None:
        for poly in polys:
            arr = np.array(poly)
            ax.plot(arr[:, 0], arr[:, 1], color="#333", lw=0.6, alpha=0.7)

    ax.set_xlim(28, 182); ax.set_ylim(53, 79)
    ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_aspect(1.9)
    ax.grid(alpha=0.2, linestyle=":")

    if discrete:
        cbar = plt.colorbar(im, ax=ax, ticks=list(discrete.keys()), shrink=0.65, pad=0.02)
        cbar.ax.set_yticklabels(list(discrete.values()))
    else:
        plt.colorbar(im, ax=ax, label=label, shrink=0.65, pad=0.02)

    plt.tight_layout()
    plt.savefig(REPORTS / fname, dpi=160, facecolor="white")
    plt.close()
    print(f"Сохранено: {fname}")


def main():
    polys = load_border_paths()
    print(f"Полигонов России: {len(polys)}")

    df = pd.read_csv(REPORTS / "classification_data.csv")  # lon,lat,pred_magt,pred_alt,zone,vulnerability
    print(f"Точек: {len(df)}")

    # 1. MAGT
    draw_gis(df, "pred_magt",
             "Среднегодовая температура грунта (MAGT, 10 м) — Россия, 2021",
             "gis_magt.png", "RdBu_r", -12, 2, "MAGT, °C", polys=polys)

    # 2. Зоны мерзлоты
    cmap_zone = mcolors.ListedColormap(["#e8e8e8", "#fdae61", "#74add1", "#313695"])
    draw_gis(df, "zone",
             "Зоны мерзлоты России (по MAGT)",
             "gis_zones.png", cmap_zone, -0.5, 3.5, polys=polys,
             discrete={0: "нет", 1: "спорадическая", 2: "прерывистая", 3: "сплошная"})

    # 3. Уязвимость
    draw_gis(df.assign(vuln=df["vulnerability"]), "vuln",
             "Индекс уязвимости мерзлоты (риск деградации)",
             "gis_vulnerability.png", "YlOrRd", 0, 1, "индекс (0-1)", polys=polys)

    # 4. Прогноз 2050 (если есть)
    try:
        fc = pd.read_csv(REPORTS / "forecast_data.csv")
        draw_gis(fc, "change_2050",
                 "Прогноз потепления мерзлоты к 2050 (CMIP6 SSP2-4.5)",
                 "gis_forecast_2050.png", "Reds", 0, 3, "ΔMAGT, °C", polys=polys)
        draw_gis(fc, "magt_2050",
                 "Прогноз MAGT на 2050 (CMIP6 SSP2-4.5)",
                 "gis_magt_2050.png", "RdBu_r", -12, 2, "MAGT, °C", polys=polys)
    except Exception as e:
        print(f"Прогноз пропущен: {e}")

    print("\nГотово. ГИС-карты в reports/gis_*.png")


if __name__ == "__main__":
    main()