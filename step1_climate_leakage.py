"""
ШАГ 1. Проверка climate leakage.

Обучаем три модели на одних и тех же данных Надыма, но с разными наборами фич:
  A. FULL      — все фичи (как было)
  B. SATELLITE — только настоящие спутниковые наблюдения (без климата)
  C. CLIMATE   — только климатические поля (на которых построен Obu)

Сравниваем R²/RMSE. Интерпретация:
  - Если B почти как A  -> спутник несёт сигнал о мерзлоте (хорошо)
  - Если C почти как A  -> модель в основном восстанавливает климат-вход Obu (leakage)

Запуск:
    python3 step1_climate_leakage.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

FEATURES_GEOJSON = Path("data/features/Nadym_2022_features.geojson")
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Наборы фич
SATELLITE_FEATURES = [
    "NDVI", "NDWI", "NDMI", "SAVI",          # вегетация (Landsat)
    "LST_summer", "LST_winter", "LST_annual",  # тепло (MODIS)
    "FDD", "TDD",                              # градусо-дни (MODIS)
    "snow_days",                               # снег (MODIS)
    "soil_oc", "soil_bd", "soil_clay",         # почва (OpenLandMap)
]
CLIMATE_FEATURES = [
    "MAAT", "MAP", "era5_temp", "era5_precip",  # климат (WorldClim/ERA5)
]
FULL_FEATURES = SATELLITE_FEATURES + CLIMATE_FEATURES


def load_and_label():
    print("Читаю фичи и сэмплирую Obu...")
    gdf = gpd.read_file(FEATURES_GEOJSON)
    gdf = gdf.set_crs("EPSG:3857", allow_override=True)
    cen = gdf.geometry.centroid.to_crs("EPSG:4326")
    lons, lats = cen.x.values, cen.y.values

    with rasterio.open(OBU_TIF) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        sampled = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        sampled[sampled == src.nodata] = np.nan
        sampled[sampled < -100] = np.nan

    df = pd.DataFrame({c: gdf[c].values for c in FULL_FEATURES if c in gdf.columns})
    df["target_magt"] = sampled
    df = df.dropna(subset=FULL_FEATURES + ["target_magt"]).reset_index(drop=True)
    print(f"Ячеек после чистки: {len(df)}")
    return df


def train_one(df, features, name):
    X = df[features].values
    y = df["target_magt"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    print(f"\n[{name}]  фич: {len(features)}")
    print(f"  RMSE: {rmse:.3f} °C   MAE: {mae:.3f} °C   R²: {r2:.3f}")
    return {"name": name, "n_features": len(features), "rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = load_and_label()

    print("\n" + "=" * 55)
    print("СРАВНЕНИЕ ТРЁХ НАБОРОВ ФИЧ")
    print("=" * 55)

    res_full = train_one(df, FULL_FEATURES, "A. FULL (все фичи)")
    res_sat = train_one(df, SATELLITE_FEATURES, "B. SATELLITE (только спутник)")
    res_clim = train_one(df, CLIMATE_FEATURES, "C. CLIMATE (только климат)")

    # Интерпретация
    print("\n" + "=" * 55)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 55)
    drop_sat = res_full["r2"] - res_sat["r2"]
    print(f"\nПадение R² при удалении климата: {drop_sat:+.3f}")
    print(f"  FULL R²={res_full['r2']:.3f}  ->  SATELLITE R²={res_sat['r2']:.3f}")

    if res_sat["r2"] > 0.9 * res_full["r2"]:
        print("  ВЫВОД: спутник почти полностью объясняет Obu без климата.")
        print("         -> спутниковый сигнал реален, leakage невелик. ХОРОШО.")
    elif res_sat["r2"] > 0.6 * res_full["r2"]:
        print("  ВЫВОД: спутник несёт существенный сигнал, но климат добавляет много.")
        print("         -> частичный leakage. Приемлемо, но отметить в отчёте.")
    else:
        print("  ВЫВОД: без климата модель резко слабеет.")
        print("         -> сильный climate leakage. Модель в основном восстанавливает")
        print("            климат-вход Obu, а не учит мерзлоту по спутнику.")

    print(f"\nКлимат в одиночку: R²={res_clim['r2']:.3f}")
    if res_clim["r2"] > res_full["r2"] * 0.9:
        print("  Климат сам по себе почти так же хорош как всё вместе ->")
        print("  подтверждает, что Obu в этом регионе определяется климатом.")

    # График
    results = [res_full, res_sat, res_clim]
    names = [r["name"] for r in results]
    r2s = [r["r2"] for r in results]
    rmses = [r["rmse"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(3), r2s, color=["steelblue", "green", "orange"])
    axes[0].set_xticks(range(3))
    axes[0].set_xticklabels(["FULL", "SATELLITE", "CLIMATE"], rotation=0)
    axes[0].set_ylabel("R²")
    axes[0].set_title("R² по наборам фич")
    axes[0].grid(alpha=0.3, axis="y")
    for i, v in enumerate(r2s):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    axes[1].bar(range(3), rmses, color=["steelblue", "green", "orange"])
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["FULL", "SATELLITE", "CLIMATE"], rotation=0)
    axes[1].set_ylabel("RMSE °C")
    axes[1].set_title("RMSE по наборам фич")
    axes[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "step1_climate_leakage.png", dpi=120)
    print(f"\nГрафик сохранён: {REPORTS_DIR / 'step1_climate_leakage.png'}")

    # Таблица
    pd.DataFrame(results).to_csv(REPORTS_DIR / "step1_results.csv", index=False)
    print(f"Таблица: {REPORTS_DIR / 'step1_results.csv'}")


if __name__ == "__main__":
    main()