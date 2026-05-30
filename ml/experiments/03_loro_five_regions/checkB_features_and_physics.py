"""
ПРОВЕРКА B. Два в одном:
  1. АУДИТ ЕДИНИЦ всех фич — проверяем физическую вменяемость каждой.
  2. Тест "физика без климата" — обобщается ли модель только на переносимых
     физических фичах (LST, FDD, snow, DEM), без географических ярлыков (MAP, MAAT).

Запуск:
    python3 checkB_features_and_physics.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform as warp_transform
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

REGIONS = ["Nadym", "Vorkuta", "Chara", "Yakutsk", "Tiksi"]
YEAR = 2021
FEAT_DIR = Path("data/features")
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")

# Ожидаемые физические диапазоны (грубо) для аудита
EXPECTED_RANGES = {
    "NDVI":       (-0.2, 1.0,  "безразмерный, тундра 0.1-0.5"),
    "NDWI":       (-1.0, 1.0,  "безразмерный"),
    "NDMI":       (-1.0, 1.0,  "безразмерный"),
    "SAVI":       (-1.0, 1.5,  "безразмерный"),
    "LST_summer": (0,    35,   "°C, лето поверхность"),
    "LST_winter": (-50, -5,    "°C, зима поверхность"),
    "LST_annual": (-25,  10,   "°C, годовое среднее поверхность"),
    "FDD":        (1000, 8000, "°C·дни замораживания"),
    "TDD":        (500,  3500, "°C·дни оттаивания"),
    "snow_days":  (0,    250,  "дней (но MODIS видит только ясные!)"),
    "elevation":  (-50,  3000, "м"),
    "slope":      (0,    60,   "градусы"),
    "aspect":     (0,    360,  "градусы"),
    "soil_oc":    (0,    500,  "г/кг ×? — проверить"),
    "soil_bd":    (0,    250,  "кг/м³ ×? — проверить"),
    "soil_clay":  (0,    100,  "% — проверить"),
    "MAAT":       (-200, 200,  "°C ×10 в WorldClim! т.е. -150=-15°C"),
    "MAP":        (0,    2000, "мм"),
    "era5_temp":  (-30,  10,   "°C"),
    "era5_precip":(0,    1,    "м/мес? — проверить единицы"),
}

SATELLITE_PHYS = ["LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
                  "snow_days", "elevation", "slope", "aspect"]  # чистая физика
VEG_SOIL = ["NDVI", "NDWI", "NDMI", "SAVI", "soil_oc", "soil_bd", "soil_clay"]
CLIMATE = ["MAAT", "MAP", "era5_temp", "era5_precip"]
ALL_FEATS = SATELLITE_PHYS + VEG_SOIL + CLIMATE


def load_region(name):
    gdf = gpd.read_file(FEAT_DIR / f"{name}_{YEAR}_features.geojson").set_crs("EPSG:3857", allow_override=True)
    cen = gdf.geometry.centroid.to_crs("EPSG:4326")
    lons, lats = cen.x.values, cen.y.values
    with rasterio.open(OBU_TIF) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        magt = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        magt[magt == src.nodata] = np.nan
        magt[magt < -100] = np.nan
    df = pd.DataFrame({c: gdf[c].values for c in ALL_FEATS if c in gdf.columns})
    df["target_magt"] = magt
    df["region"] = name
    return df.dropna(subset=ALL_FEATS + ["target_magt"]).reset_index(drop=True)


def audit_units(full):
    print("=" * 80)
    print("АУДИТ ЕДИНИЦ ФИЧЕЙ")
    print("=" * 80)
    print(f"{'Фича':<12}{'min':>9}{'max':>9}{'mean':>9}  {'ожидание':<35}{'?':>4}")
    print("-" * 80)
    for f in ALL_FEATS:
        if f not in full.columns:
            print(f"{f:<12} ОТСУТСТВУЕТ")
            continue
        mn, mx, mean = full[f].min(), full[f].max(), full[f].mean()
        lo, hi, desc = EXPECTED_RANGES[f]
        ok = "OK" if (mn >= lo - abs(lo)*0.5 - 1 and mx <= hi + abs(hi)*0.5 + 1) else "!!!"
        print(f"{f:<12}{mn:>9.1f}{mx:>9.1f}{mean:>9.1f}  {desc:<35}{ok:>4}")
    print()


def fit_eval(train, test, feats):
    m = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         tree_method="hist", random_state=42, n_jobs=-1)
    m.fit(train[feats].values, train["target_magt"].values)
    yp = m.predict(test[feats].values)
    yt = test["target_magt"].values
    return r2_score(yt, yp), np.sqrt(mean_squared_error(yt, yp))


def loro_test(data, feats, label):
    print(f"\n--- LORO: {label} ({len(feats)} фич) ---")
    r2s = []
    for holdout in REGIONS:
        train = pd.concat([data[r] for r in REGIONS if r != holdout], ignore_index=True)
        r2, rmse = fit_eval(train, data[holdout], feats)
        r2s.append(r2)
        print(f"  {holdout:<10} R²={r2:>8.3f}  RMSE={rmse:.2f}")
    print(f"  СРЕДНЕЕ R²={np.mean(r2s):.3f}")
    return np.mean(r2s)


def main():
    print("Загружаю регионы...\n")
    data = {r: load_region(r) for r in REGIONS}
    full = pd.concat(data.values(), ignore_index=True)

    # 1. Аудит
    audit_units(full)

    # 2. LORO на разных наборах
    print("=" * 80)
    print("ТЕСТ B: обобщается ли ФИЗИКА без климата?")
    print("=" * 80)
    res = {}
    res["phys"] = loro_test(data, SATELLITE_PHYS, "ТОЛЬКО ФИЗИКА (LST/FDD/snow/DEM)")
    res["phys+veg"] = loro_test(data, SATELLITE_PHYS + VEG_SOIL, "ФИЗИКА + вегетация/почва")
    res["climate"] = loro_test(data, CLIMATE, "ТОЛЬКО КЛИМАТ")
    res["all"] = loro_test(data, ALL_FEATS, "ВСЁ")

    print("\n" + "=" * 80)
    print("ИТОГ")
    print("=" * 80)
    for k, v in res.items():
        print(f"  {k:<12} средний LORO R² = {v:.3f}")
    best = max(res, key=res.get)
    print(f"\n  Лучший набор: {best} (R²={res[best]:.3f})")
    if res[best] > 0:
        print("  Хоть какой-то набор даёт положительный LORO — есть надежда.")
    else:
        print("  Все наборы дают отрицательный LORO.")
        print("  => проблема НЕ в выборе фич, а в покрытии пространства признаков.")
        print("  => нужен Способ A: разбросанная выборка точек по всей мерзлотной зоне.")


if __name__ == "__main__":
    main()