"""
НЕЗАВИСИМАЯ ПРОВЕРКА на реальных измерениях GTN-P.

Это единственная проверка соответствия РЕАЛЬНОСТИ (не другой модели).

Что делает:
  1. Берёт 50 boreholes с реальными измерениями MAGT.
  2. Для координат каждой скважины нужны фичи -> выгружаем их из уже
     имеющихся региональных файлов (Nadym/Vorkuta/Chara/Yakutsk/Tiksi),
     находя ближайшую ячейку. Если скважина не покрыта регионом - пропускаем.
  3. Прогоняем нашу модель -> pred_magt.
  4. Сэмплируем ESA CCI T10m в тех же точках.
  5. Сравниваем: модель vs реальность, ESA CCI vs реальность.

ВАЖНО про интерпретацию (обсуждали в Шаге диагностики boreholes):
  Скважина измеряет ТОЧКУ, модель/ESA CCI - КВАДРАТ 1 км. Поэтому
  идеального совпадения не ждём. Ждём, что наша модель не сильно хуже ESA CCI.

Запуск:
    python3 validate_on_boreholes.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import joblib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

GTNP_CSV = Path("data/labels/gtnp_boreholes_v2.csv")
MODEL = "models_scatter_xgb.joblib"
ESACCI = glob.glob("data/esa_cci/*.nc")[0]
REGION_FILES = glob.glob("data/features/*_2021_features.geojson")  # региональные фичи
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_all_region_cells():
    """Собирает все региональные ячейки в один GeoDataFrame с центроидами."""
    gdfs = []
    for f in REGION_FILES:
        # пропускаем grid/scatter файлы — берём только региональные _features
        name = Path(f).stem
        if "Grid" in name or "Scatter" in name:
            continue
        gdf = gpd.read_file(f).set_crs("EPSG:3857", allow_override=True)
        cen = gdf.geometry.centroid.to_crs("EPSG:4326")
        gdf2 = gpd.GeoDataFrame(
            gdf[[c for c in FEATURE_COLS if c in gdf.columns]].copy(),
            geometry=[Point(x, y) for x, y in zip(cen.x, cen.y)],
            crs="EPSG:4326",
        )
        gdfs.append(gdf2)
    allcells = pd.concat(gdfs, ignore_index=True)
    return gpd.GeoDataFrame(allcells, geometry="geometry", crs="EPSG:4326")


def main():
    # 1. Boreholes
    bh = pd.read_csv(GTNP_CSV).dropna(subset=["lat", "lon", "magt_c"])
    bh_gdf = gpd.GeoDataFrame(
        bh, geometry=[Point(x, y) for x, y in zip(bh.lon, bh.lat)], crs="EPSG:4326")
    print(f"Скважин: {len(bh_gdf)}")

    # 2. Фичи в точках скважин — ближайшая региональная ячейка
    print("Ищу фичи для скважин в региональных файлах...")
    cells = load_all_region_cells()
    cells_m = cells.to_crs("EPSG:3857")
    bh_m = bh_gdf.to_crs("EPSG:3857")
    joined = gpd.sjoin_nearest(bh_m, cells_m, how="left", max_distance=2000, distance_col="dist")
    joined = joined.dropna(subset=["dist"]).reset_index(drop=True)
    print(f"Скважин с найденными фичами (в пределах 2 км): {len(joined)}")

    if len(joined) < 5:
        print("Слишком мало скважин покрыто региональными файлами.")
        print("Эти регионы (Nadym/Vorkuta/Chara/Yakutsk/Tiksi) покрывают только")
        print("свои скважины. Остальные скважины вне выгруженных регионов.")
        return

    # Удаляем дубли (несколько скважин -> одна ячейка)
    feat_cols_present = [c for c in FEATURE_COLS if c in joined.columns]
    joined = joined.dropna(subset=feat_cols_present)
    print(f"После очистки: {len(joined)} скважин")

    # 3. Предсказание нашей моделью
    model = joblib.load(MODEL)
    joined["pred_magt"] = model.predict(joined[FEATURE_COLS].values)

    # 4. ESA CCI T10m в точках скважин
    ds = xr.open_dataset(ESACCI)
    t10 = ds["T10m"].isel(time=0)
    lats = xr.DataArray(joined["lat"].values, dims="p")
    lons = xr.DataArray(joined["lon"].values, dims="p")
    joined["esacci_magt"] = t10.sel(lat=lats, lon=lons, method="nearest").values

    # Оставляем где все три есть
    v = joined.dropna(subset=["magt_c", "pred_magt", "esacci_magt"]).copy()
    print(f"\nИтоговое сравнение по {len(v)} скважинам\n")

    # 5. Метрики
    def metrics(true, pred, label):
        res = pred - true
        rmse = np.sqrt((res**2).mean())
        bias = res.mean()
        mae = np.abs(res).mean()
        if len(true) > 2:
            r, _ = pearsonr(true, pred)
        else:
            r = np.nan
        print(f"  [{label}]  RMSE={rmse:.2f}  MAE={mae:.2f}  bias={bias:+.2f}  corr={r:.3f}")
        return rmse, r

    print("Сравнение с РЕАЛЬНЫМИ измерениями скважин:")
    print("(скважина = точка, модель = квадрат 1км, идеала не ждём)")
    metrics(v["magt_c"].values, v["pred_magt"].values, "Наша модель vs реальность")
    metrics(v["magt_c"].values, v["esacci_magt"].values, "ESA CCI    vs реальность")
    metrics(v["esacci_magt"].values, v["pred_magt"].values, "Наша модель vs ESA CCI")

    print("\nИнтерпретация:")
    print("  Если 'наша модель vs реальность' близка к 'ESA CCI vs реальность' —")
    print("  наша модель так же близка к правде, как физическая модель. Это успех эмуляции.")

    # График
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(v["magt_c"], v["pred_magt"], label="наша модель", alpha=0.7, s=50)
    axes[0].scatter(v["magt_c"], v["esacci_magt"], label="ESA CCI", alpha=0.7, s=50, marker="^")
    lims = [-12, 4]
    axes[0].plot(lims, lims, "r--", lw=1.2)
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("Реальное MAGT (скважина) °C")
    axes[0].set_ylabel("Предсказанное °C")
    axes[0].set_title("vs реальные измерения")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].scatter(v["esacci_magt"], v["pred_magt"], alpha=0.7, s=50, color="green")
    axes[1].plot(lims, lims, "r--", lw=1.2)
    axes[1].set_xlim(lims); axes[1].set_ylim(lims)
    axes[1].set_xlabel("ESA CCI T10m °C")
    axes[1].set_ylabel("Наша модель °C")
    axes[1].set_title("Наша модель vs ESA CCI (что эмулируем)")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS / "validate_boreholes.png", dpi=120)
    print(f"\nГрафик: {REPORTS / 'validate_boreholes.png'}")
    v[["name", "site_name", "lat", "lon", "magt_c", "pred_magt", "esacci_magt"]].to_csv(
        REPORTS / "validate_boreholes.csv", index=False)
    print(f"Таблица: {REPORTS / 'validate_boreholes.csv'}")


if __name__ == "__main__":
    main()