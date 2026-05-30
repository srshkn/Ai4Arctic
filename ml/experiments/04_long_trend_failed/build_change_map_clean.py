"""
ШАГ 1: Чистая карта изменения мерзлоты 2010->2023.

Устраняет артефакты:
  1.1 Только точки, присутствующие во ВСЕ годы (2010,2017,2021,2023) -> нет блоков
  1.2 Пространственное сглаживание (медианный фильтр) -> нет пиксельного шума
  1.3 Маска достоверности: показываем тренд только где он согласован
      (знак изменения 2010->2023 совпадает со знаком линейного тренда,
       и |тренд| выше порога шума)

Запуск:
    python3 build_change_map_clean.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter, generic_filter
import joblib

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)
FEAT_DIR = Path("data/features")
YEARS = [2010, 2017, 2021, 2023]

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_year(year):
    with open(FEAT_DIR / f"RussiaGrid_0.25deg_{year}.geojson") as fh:
        gj = json.load(fh)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lon, lat = ft["geometry"]["coordinates"]
        lons.append(lon); lats.append(lat)
    df = pd.DataFrame(rows)
    df["lon"] = lons; df["lat"] = lats
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    df["k"] = df["lon"].round(2).astype(str) + "_" + df["lat"].round(2).astype(str)
    df = df.drop_duplicates(subset="k")
    return df


def main():
    model_magt = joblib.load("models_scatter_xgb.joblib")
    model_alt = joblib.load("models_alt_xgb.joblib")

    # Предсказания по годам, индексируем по ключу координат
    preds = {}
    for y in YEARS:
        df = load_year(y)
        df[f"magt_{y}"] = model_magt.predict(df[FEATURE_COLS].values)
        df[f"alt_{y}"] = model_alt.predict(df[FEATURE_COLS].values)
        preds[y] = df[["k", "lon", "lat", f"magt_{y}", f"alt_{y}"]].set_index("k")
        print(f"{y}: {len(df)} точек")

    # 1.1 ТОЛЬКО общие точки всех лет
    common = preds[YEARS[0]]
    for y in YEARS[1:]:
        common = common.join(preds[y].drop(columns=["lon", "lat"]), how="inner")
    print(f"\nОбщих точек во ВСЕ годы: {len(common)}")

    magt_cols = [f"magt_{y}" for y in YEARS]
    alt_cols = [f"alt_{y}" for y in YEARS]

    # Изменение и тренд
    common["dmagt"] = common["magt_2023"] - common["magt_2010"]
    common["dalt"] = common["alt_2023"] - common["alt_2010"]
    yrs = np.array(YEARS, dtype=float)
    common["trend"] = np.polyfit(yrs, common[magt_cols].values.T, 1)[0]  # °C/год

    # 1.3 Маска достоверности:
    #   - знак dmagt совпадает со знаком тренда (согласованность)
    #   - |тренд| выше порога шума (0.02°C/год = 0.2°C/декада)
    consistent = np.sign(common["dmagt"]) == np.sign(common["trend"])
    significant = np.abs(common["trend"]) > 0.02
    common["reliable"] = consistent & significant
    print(f"Согласованных и значимых точек: {common['reliable'].sum()} "
          f"({100*common['reliable'].mean():.0f}%)")

    print(f"\nСреднее изменение MAGT: {common['dmagt'].mean():+.2f}°C за 13 лет "
          f"({common['dmagt'].mean()/13*10:+.2f}°C/декада)")
    print(f"Средний тренд: {common['trend'].mean()*10:+.2f}°C/декада")
    print(f"Доля потепления: {100*(common['dmagt']>0).mean():.0f}%")

    # === Функция растеризации со сглаживанием ===
    def rasterize(sub, col, res=0.25, mask_dist=0.4, smooth=True):
        lon_g = np.arange(30, 180, res)
        lat_g = np.arange(55, 78, res)
        LON, LAT = np.meshgrid(lon_g, lat_g)
        ss = sub.dropna(subset=[col])
        Z = griddata((ss["lon"], ss["lat"]), ss[col], (LON, LAT), method="linear")
        # маска по расстоянию
        tree = cKDTree(np.column_stack([ss["lon"], ss["lat"]]))
        d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
        Z = np.where(d.reshape(LON.shape) <= mask_dist, Z, np.nan)
        # 1.2 сглаживание (медианный фильтр, игнорируя NaN)
        if smooth:
            Zf = Z.copy()
            mask = ~np.isnan(Z)
            Zfilled = np.where(mask, Z, 0)
            from scipy.ndimage import uniform_filter
            num = uniform_filter(Zfilled, size=3)
            cnt = uniform_filter(mask.astype(float), size=3)
            with np.errstate(invalid="ignore", divide="ignore"):
                Zsmooth = num / cnt
            Z = np.where(mask, Zsmooth, np.nan)
        return LON, LAT, Z

    def draw(LON, LAT, Z, title, fname, vmin, vmax, cmap="RdBu_r", label=""):
        fig, ax = plt.subplots(figsize=(16, 7))
        im = ax.pcolormesh(LON, LAT, Z, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label=label, shrink=0.7)
        ax.set_aspect(2.0); ax.grid(alpha=0.15)
        plt.tight_layout()
        plt.savefig(REPORTS / fname, dpi=140)
        plt.close()
        print(f"Сохранено: {fname}")

    # Карта изменения MAGT — только надёжные точки, сглажено
    rel = common[common["reliable"]]
    LON, LAT, DZ = rasterize(rel, "dmagt")
    draw(LON, LAT, DZ,
         "Изменение MAGT 2010→2023 (°C), только достоверные зоны\nкрасное = потепление",
         "change_magt_clean.png", -2, 2, label="ΔMAGT, °C")

    # Карта тренда — все общие точки, сглажено
    LON, LAT, TR = rasterize(common, "trend")
    draw(LON, LAT, TR * 10,
         "Тренд MAGT 2010-2023 (°C/декада), сглажено",
         "trend_magt_clean.png", -1, 1, label="°C/декада")

    # Карта изменения ALT — надёжные, сглажено
    LON, LAT, DA = rasterize(rel, "dalt")
    draw(LON, LAT, DA,
         "Изменение ALT 2010→2023 (м), достоверные зоны\nкрасное = глубже протаивание",
         "change_alt_clean.png", -0.4, 0.4, label="ΔALT, м")

    common.reset_index()[["lon", "lat", "dmagt", "dalt", "trend", "reliable"]].to_csv(
        REPORTS / "change_clean_data.csv", index=False)
    print(f"\nДанные: change_clean_data.csv")
    print(f"\nДля отчёта: {len(common)} общих точек, "
          f"{100*common['reliable'].mean():.0f}% достоверных, "
          f"тренд {common['trend'].mean()*10:+.2f}°C/декада")


if __name__ == "__main__":
    main()