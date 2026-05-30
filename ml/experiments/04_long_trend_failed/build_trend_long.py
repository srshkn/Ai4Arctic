"""
ДЛИННЫЙ ТРЕНД по 7 годам (2010-2023) — цель "максимума".

Больше временных точек -> устойчивее тренд -> меньше погодного шума.
Сравниваем с трендом ESA CCI: ожидаем, что корреляция паттерна ВЫРАСТЕТ
относительно версии на 4 годах (r было -0.11).

Запуск:
    python3 build_trend_long.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr
import joblib

REPORTS = Path("reports")
FEAT_DIR = Path("data/features")
YEARS = [2010, 2013, 2015, 2017, 2019, 2021, 2023]

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
        lo, la = ft["geometry"]["coordinates"]
        lons.append(lo); lats.append(la)
    df = pd.DataFrame(rows)
    df["lon"] = lons; df["lat"] = lats
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    df["k"] = df["lon"].round(2).astype(str) + "_" + df["lat"].round(2).astype(str)
    return df.drop_duplicates(subset="k")


def main():
    model = joblib.load("models_scatter_xgb.joblib")

    preds = {}
    for y in YEARS:
        df = load_year(y)
        df[f"magt_{y}"] = model.predict(df[FEATURE_COLS].values)
        preds[y] = df[["k", "lon", "lat", f"magt_{y}"]].set_index("k")
        print(f"{y}: {len(df)} точек")

    # Общие точки во ВСЕ 7 лет
    common = preds[YEARS[0]]
    for y in YEARS[1:]:
        common = common.join(preds[y].drop(columns=["lon", "lat"]), how="inner")
    print(f"\nОбщих точек во все 7 лет: {len(common)}")

    magt_cols = [f"magt_{y}" for y in YEARS]
    yrs = np.array(YEARS, dtype=float)

    # Тренд по 7 точкам + R² линейной аппроксимации (мера устойчивости)
    Y = common[magt_cols].values  # (n, 7)
    fit = np.polyfit(yrs, Y.T, 1)
    slopes = fit[0]
    common["trend"] = slopes
    # R² тренда: насколько линейна динамика (высокий = устойчивый тренд, низкий = шум)
    pred_lin = np.outer(slopes, yrs) + fit[1][:, None]
    ss_res = ((Y - pred_lin) ** 2).sum(axis=1)
    ss_tot = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    common["trend_r2"] = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)

    print(f"Средний тренд (7 лет): {common['trend'].mean()*10:+.3f} °C/декада")
    print(f"Медиана R² линейности тренда: {common['trend_r2'].median():.2f}")
    print(f"Точек с устойчивым трендом (R²>0.5): {(common['trend_r2']>0.5).mean()*100:.0f}%")

    # Сравнение с ESA CCI (2010, 2021, 2023 — что есть)
    print("\nСравнение с ESA CCI...")
    esa_years = [2010, 2021, 2023]
    for y in esa_years:
        cands = [f for f in glob.glob("data/esa_cci/*.nc") if "GTD" in f.upper() and str(y) in f]
        if not cands:
            print(f"  ESA CCI {y} не найден"); continue
        ds = xr.open_dataset(cands[0])
        t10 = ds["T10m"].isel(time=0)
        la = xr.DataArray(common["lat"].values, dims="p")
        lo = xr.DataArray(common["lon"].values, dims="p")
        common[f"esa_{y}"] = t10.sel(lat=la, lon=lo, method="nearest").values

    esa_cols = [f"esa_{y}" for y in esa_years if f"esa_{y}" in common.columns]
    if len(esa_cols) >= 2:
        v = common.dropna(subset=esa_cols)
        ey = np.array([int(c.split("_")[1]) for c in esa_cols], dtype=float)
        v_esa_trend = np.polyfit(ey, v[esa_cols].values.T, 1)[0]
        v = v.assign(esa_trend=v_esa_trend)
        r, _ = pearsonr(v["trend"], v["esa_trend"])
        same = (np.sign(v["trend"]) == np.sign(v["esa_trend"])).mean()
        print(f"  Наш тренд (7 лет):    {v['trend'].mean()*10:+.3f} °C/декада")
        print(f"  ESA CCI тренд:        {v['esa_trend'].mean()*10:+.3f} °C/декада")
        print(f"  Корреляция паттерна:  r={r:.3f}  (на 4 годах было -0.11)")
        print(f"  Совпадение знака:     {100*same:.0f}%")

    # === КАРТА тренда (сглаженная) ===
    def raster(sub, col, res=0.25, md=0.4, smooth=True):
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
        return LON, LAT, Z

    LON, LAT, TR = raster(common, "trend")
    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.pcolormesh(LON, LAT, TR*10, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    ax.set_xlabel("Долгота °E"); ax.set_ylabel("Широта °N")
    ax.set_title("Тренд MAGT по 7 годам (2010-2023), °C/декада\nкрасное = потепление")
    plt.colorbar(im, ax=ax, label="°C/декада", shrink=0.7)
    ax.set_aspect(2.0); ax.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(REPORTS / "trend_magt_long.png", dpi=140)
    plt.close()
    print(f"\nСохранено: trend_magt_long.png")

    common.reset_index()[["lon", "lat", "trend", "trend_r2"]].to_csv(
        REPORTS / "trend_long_data.csv", index=False)
    print("Данные: trend_long_data.csv")


if __name__ == "__main__":
    main()