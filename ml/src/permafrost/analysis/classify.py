"""
НАУЧНАЯ КЛАССИФИКАЦИЯ МЕРЗЛОТЫ (зоны + уязвимость).

Основано на литературе (IPA, Brown et al. 1997):
  - Зоны мерзлоты определяются по MAGT (континуальность связана с темп. грунта)
  - ALT немонотонна: максимальна в ПЕРЕХОДНОЙ зоне, мала в холодной (Smith,
    "Beyond MAGT"; Wikipedia: Элсмир 10см vs Якутск 2.5м)
  - Поэтому ALT используем как индикатор УЯЗВИМОСТИ, не для деления на зоны

Строит:
  1. Зоны мерзлоты по MAGT: сплошная / прерывистая / спорадическая / нет
  2. Карту ALT с акцентом на переходную зону
  3. Индекс УЯЗВИМОСТИ = тёплая мерзлота (MAGT近0) + глубокая ALT = риск деградации

Вход: reports/russia_map_data.csv (lon,lat,pred_magt,pred_class)
       + ALT досчитывается из модели на сетке

Запуск:
    python3 classify_permafrost.py
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
import joblib

REPORTS = Path("reports")
MAP_CSV = REPORTS / "russia_map_data.csv"
GRID_GEOJSON = glob.glob("data/features/RussiaGrid_0.25deg_2021.geojson")

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def add_alt(df):
    model = joblib.load("models_alt_xgb.joblib")
    with open(GRID_GEOJSON[0]) as f:
        gj = json.load(f)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lo, la = ft["geometry"]["coordinates"]
        lons.append(lo); lats.append(la)
    g = pd.DataFrame(rows); g["lon"] = lons; g["lat"] = lats
    g = g.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    g["pred_alt"] = model.predict(g[FEATURE_COLS].values)
    return df.merge(g[["lon", "lat", "pred_alt"]], on=["lon", "lat"], how="left")


def magt_to_zone(magt):
    """Зоны мерзлоты по MAGT (пороги из литературы IPA/Brown)."""
    # 3 = сплошная (continuous, холодная, <-5)
    # 2 = прерывистая (discontinuous, -5..-2)
    # 1 = спорадическая/переходная (sporadic, -2..0)
    # 0 = нет мерзлоты (>0)
    z = np.zeros(len(magt), dtype=int)
    z[magt <= 0] = 1
    z[magt <= -2] = 2
    z[magt <= -5] = 3
    return z


def rasterize(df, col, res=0.2, mask_dist=0.4, permafrost_only=False):
    sub = df.dropna(subset=[col])
    lon_g = np.arange(df["lon"].min(), df["lon"].max(), res)
    lat_g = np.arange(df["lat"].min(), df["lat"].max(), res)
    LON, LAT = np.meshgrid(lon_g, lat_g)
    Z = griddata((sub["lon"], sub["lat"]), sub[col], (LON, LAT), method="linear")
    tree = cKDTree(np.column_stack([sub["lon"], sub["lat"]]))
    d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
    Z = np.where(d.reshape(LON.shape) <= mask_dist, Z, np.nan)
    if permafrost_only:
        magt_sub = df.dropna(subset=["pred_magt"])
        M = griddata((magt_sub["lon"], magt_sub["lat"]), magt_sub["pred_magt"],
                     (LON, LAT), method="linear")
        Z = np.where(M < 0, Z, np.nan)
    return LON, LAT, Z


def main():
    df = pd.read_csv(MAP_CSV)
    df = add_alt(df)
    print(f"Точек: {len(df)}, ALT досчитана: {df['pred_alt'].notna().sum()}")

    # 1. Зоны по MAGT
    df["zone"] = magt_to_zone(df["pred_magt"].values)
    print("\nЗоны мерзлоты (по MAGT):")
    names = {0: "нет", 1: "спорадическая", 2: "прерывистая", 3: "сплошная"}
    for z in [3, 2, 1, 0]:
        n = (df["zone"] == z).sum()
        print(f"  {names[z]}: {n} ({100*n/len(df):.1f}%)")

    # Проверка: ALT по зонам (должна быть НЕМОНОТОННА — макс в переходной)
    print("\nСредняя ALT по зонам (проверка немонотонности):")
    for z in [3, 2, 1]:
        sub = df[(df["zone"] == z) & df["pred_alt"].notna()]
        if len(sub):
            print(f"  {names[z]}: ALT = {sub['pred_alt'].mean():.2f} м")

    # 2. Индекс уязвимости: тёплая мерзлота (MAGT近0) + глубокая ALT
    # нормируем: чем ближе MAGT к 0 (снизу) и чем глубже ALT, тем выше риск
    pf = df[df["pred_magt"] < 0].copy()
    # риск растёт когда MAGT -> 0 (от -5 до 0) и ALT большой
    magt_risk = np.clip((pf["pred_magt"] + 5) / 5, 0, 1)  # 0 при -5, 1 при 0
    alt_risk = np.clip(pf["pred_alt"] / 2.0, 0, 1)         # 0..1, насыщение к 2м
    pf["vulnerability"] = (magt_risk * alt_risk)            # оба высоки = риск
    df = df.merge(pf[["lon", "lat", "vulnerability"]], on=["lon", "lat"], how="left")
    print(f"\nИндекс уязвимости посчитан для {df['vulnerability'].notna().sum()} мерзлотных точек")
    print(f"Зон высокого риска (vuln>0.5): {(df['vulnerability']>0.5).sum()}")

    # === КАРТА 1: Зоны мерзлоты ===
    LON, LAT, Z = rasterize(df, "zone")
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = mcolors.ListedColormap(["#f0f0f0", "#fdae61", "#74add1", "#313695"])
    im = ax.pcolormesh(LON, LAT, Z, cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto")
    ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
    ax.set_title("Зоны мерзлоты России (по MAGT)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3], shrink=0.7)
    cbar.ax.set_yticklabels(["нет", "спорадическая", "прерывистая", "сплошная"])
    ax.set_aspect(2.0); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(REPORTS / "zones_permafrost.png", dpi=140)
    plt.close()
    print("\nСохранено: zones_permafrost.png")

    # === КАРТА 2: Индекс уязвимости ===
    LON, LAT, V = rasterize(df, "vulnerability", permafrost_only=True)
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.pcolormesh(LON, LAT, V, cmap="YlOrRd", vmin=0, vmax=1, shading="auto")
    ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
    ax.set_title("Индекс уязвимости мерзлоты (тёплая + глубокое протаивание)\nкрасное = высокий риск деградации")
    plt.colorbar(im, ax=ax, label="индекс уязвимости (0-1)", shrink=0.7)
    ax.set_aspect(2.0); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(REPORTS / "vulnerability_index.png", dpi=140)
    plt.close()
    print("Сохранено: vulnerability_index.png")

    # === График: ALT по зонам (доказательство немонотонности) ===
    fig, ax = plt.subplots(figsize=(8, 5))
    zone_data = [df[(df["zone"]==z) & df["pred_alt"].notna()]["pred_alt"] for z in [3,2,1]]
    ax.boxplot(zone_data, labels=["сплошная\n(холодная)", "прерывистая", "спорадическая\n(тёплая)"])
    ax.set_ylabel("ALT, м")
    ax.set_title("Глубина протаивания по зонам мерзлоты\n(макс в переходной зоне — Smith, 'Beyond MAGT')")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(REPORTS / "alt_by_zone.png", dpi=130)
    plt.close()
    print("Сохранено: alt_by_zone.png")

    df[["lon", "lat", "pred_magt", "pred_alt", "zone", "vulnerability"]].to_csv(
        REPORTS / "classification_data.csv", index=False)
    print(f"\nДанные: classification_data.csv")


if __name__ == "__main__":
    main()