"""
ШАГ 2. Объединение двух регионов (Надым + Якутск) и переобучение.

Что делает:
  1. Читает фичи обоих регионов, сэмплирует Obu MAGT.
  2. Объединяет в один датасет.
  3. Обучает XGBoost на объединённых данных.
  4. Сравнивает метрики и баланс классов с одним Надымом.
  5. Повторяет тест climate leakage на расширенных данных.
  6. Карты по обоим регионам.

Топография (elevation/slope/aspect) НЕ используется: в Надыме её нет
(SRTM не покрыл 65N). Используем общий набор без топографии.

Запуск:
    python3 step2_two_regions.py
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
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    f1_score, cohen_kappa_score, classification_report, confusion_matrix
)

FEATURES = {
    "Nadym":   Path("data/features/Nadym_2022_features.geojson"),
    "Yakutsk": Path("data/features/Yakutsk_2022_features.geojson"),
}
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Общий набор фич (без топографии — её нет в Надыме)
SATELLITE_FEATURES = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "soil_oc", "soil_bd", "soil_clay",
]
CLIMATE_FEATURES = ["MAAT", "MAP", "era5_temp", "era5_precip"]
FULL_FEATURES = SATELLITE_FEATURES + CLIMATE_FEATURES


def magt_to_class(magt):
    cls = np.zeros(len(magt), dtype=int)
    cls[magt <= 0.0] = 1
    cls[magt <= -1.5] = 2
    return cls


def load_region(name, path):
    gdf = gpd.read_file(path)
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
    df["lon"] = lons
    df["lat"] = lats
    df["region"] = name
    df = df.dropna(subset=FULL_FEATURES + ["target_magt"]).reset_index(drop=True)
    print(f"  {name}: {len(df)} ячеек, MAGT {df['target_magt'].min():.2f}...{df['target_magt'].max():.2f} °C")
    return df


def train(df, features, label):
    X = df[features].values
    y = df["target_magt"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    yp = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, yp))
    r2 = r2_score(y_te, yp)
    mae = mean_absolute_error(y_te, yp)
    print(f"  [{label}] фич={len(features)}  RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")
    return model, {"label": label, "rmse": rmse, "mae": mae, "r2": r2}, (y_te, yp)


def main():
    print("[1] Загружаю регионы...")
    dfs = [load_region(name, path) for name, path in FEATURES.items()]
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nОбъединённый датасет: {len(df)} ячеек")
    print(f"MAGT диапазон: {df['target_magt'].min():.2f} ... {df['target_magt'].max():.2f} °C")

    # Баланс классов
    cls = magt_to_class(df["target_magt"].values)
    print("\nБаланс классов (объединённый):")
    for c in [0, 1, 2]:
        n = (cls == c).sum()
        names = {0: "нет мерзлоты", 1: "деградирующая", 2: "устойчивая"}
        print(f"  Класс {c} ({names[c]}): {n} ({100*n/len(cls):.1f}%)")

    print("\n[2] Обучение на объединённых данных:")
    model_full, res_full, (y_te, yp) = train(df, FULL_FEATURES, "FULL (2 региона)")
    model_sat, res_sat, _ = train(df, SATELLITE_FEATURES, "SATELLITE")
    model_clim, res_clim, _ = train(df, CLIMATE_FEATURES, "CLIMATE")

    # Метрики классификации для FULL
    print("\n[3] Классификация (FULL):")
    y_te_cls = magt_to_class(y_te)
    yp_cls = magt_to_class(yp)
    f1m = f1_score(y_te_cls, yp_cls, average="macro")
    kappa = cohen_kappa_score(y_te_cls, yp_cls)
    print(f"  macro-F1={f1m:.3f}  kappa={kappa:.3f}")
    print(classification_report(y_te_cls, yp_cls,
          target_names=["0:none", "1:degr", "2:stable"], digits=3, zero_division=0))
    print("  Confusion matrix:")
    print(confusion_matrix(y_te_cls, yp_cls))

    # Сравнение climate leakage
    print("\n[4] Climate leakage на 2 регионах:")
    print(f"  FULL R²={res_full['r2']:.3f}  SATELLITE R²={res_sat['r2']:.3f}  CLIMATE R²={res_clim['r2']:.3f}")
    print(f"  Спутник объясняет {100*res_sat['r2']/res_full['r2']:.0f}% от полной модели")
    print("  (на одном Надыме было: спутник 83% от полной)")

    # Важность фич
    print("\n[5] Важность фич (FULL):")
    imp = pd.Series(model_full.feature_importances_, index=FULL_FEATURES).sort_values(ascending=False)
    print(imp.to_string())

    # Карты
    df["pred"] = model_full.predict(df[FULL_FEATURES].values)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for i, rname in enumerate(["Nadym", "Yakutsk"]):
        sub = df[df["region"] == rname]
        s1 = axes[i,0].scatter(sub["lon"], sub["lat"], c=sub["target_magt"], cmap="coolwarm_r", s=2, vmin=-10, vmax=2)
        axes[i,0].set_title(f"{rname}: Obu (target)")
        plt.colorbar(s1, ax=axes[i,0], label="MAGT °C")
        s2 = axes[i,1].scatter(sub["lon"], sub["lat"], c=sub["pred"], cmap="coolwarm_r", s=2, vmin=-10, vmax=2)
        axes[i,1].set_title(f"{rname}: Predicted")
        plt.colorbar(s2, ax=axes[i,1], label="MAGT °C")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "step2_two_regions_maps.png", dpi=110)
    print(f"\nКарты: {REPORTS_DIR / 'step2_two_regions_maps.png'}")

    # Важность — график
    plt.figure(figsize=(8, 7))
    imp.sort_values().plot.barh()
    plt.title("Feature importance (2 региона)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "step2_importance.png", dpi=120)
    print(f"Важность: {REPORTS_DIR / 'step2_importance.png'}")


if __name__ == "__main__":
    main()