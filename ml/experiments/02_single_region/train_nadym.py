"""
ОБУЧЕНИЕ МОДЕЛИ: спутниковые фичи -> MAGT (target из Obu 2019).

Что делает:
  1. Читает фичи Надыма (GeoJSON из GEE).
  2. Сэмплирует MAGT из Obu GeoTIFF в центр каждой ячейки -> target.
  3. Чистит данные (убирает NaN, ячейки вне области мерзлоты).
  4. Обучает XGBoost: фичи -> MAGT.
  5. Метрики на честном train/test split (RMSE, R2, MAE).
  6. Важность фич.
  7. 3-классовая классификация через пороги.
  8. Карта предсказанного MAGT.

Запуск:
    python3 train_nadym.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    f1_score, cohen_kappa_score, classification_report, confusion_matrix
)

# ============================================================
# КОНФИГ
# ============================================================
FEATURES_GEOJSON = Path("data/features/Nadym_2022_features.geojson")
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def magt_to_class(magt):
    """3 класса по Ran 2022."""
    cls = np.zeros(len(magt), dtype=int)
    cls[magt <= 0.0] = 1   # деградирующая
    cls[magt <= -1.5] = 2  # устойчивая
    return cls


# ============================================================
# 1. Загрузка фичей + сэмплирование Obu
# ============================================================
def load_and_label():
    print("[1/5] Читаю фичи Надыма...")
    gdf = gpd.read_file(FEATURES_GEOJSON)
    print(f"      Ячеек: {len(gdf)}")

    # ВАЖНО: координаты в файле -- метры EPSG:3857, но метка стоит 4326.
    # Принудительно переопределяем CRS на реальный (3857).
    gdf = gdf.set_crs("EPSG:3857", allow_override=True)

    # Центроиды в метрах (3857), потом конвертация в lat/lon (4326)
    centroids_3857 = gdf.geometry.centroid
    centroids_4326 = centroids_3857.to_crs("EPSG:4326")
    lons = centroids_4326.x.values
    lats = centroids_4326.y.values

    # Фильтр невалидных координат (NaN или вне диапазона)
    valid_coord = (
        np.isfinite(lons) & np.isfinite(lats)
        & (lats > -89.9) & (lats < 89.9)
        & (lons >= -180) & (lons <= 180)
    )
    n_bad = (~valid_coord).sum()
    if n_bad > 0:
        print(f"      Отброшено ячеек с битыми координатами: {n_bad}")

    df = pd.DataFrame({c: gdf[c].values for c in FEATURE_COLS if c in gdf.columns})
    df["lon"] = lons
    df["lat"] = lats
    df = df[valid_coord].reset_index(drop=True)

    print("[2/5] Сэмплирую Obu MAGT в центры ячеек...")
    with rasterio.open(OBU_TIF) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs,
                                df["lon"].tolist(), df["lat"].tolist())
        coords = list(zip(xs, ys))
        sampled = np.array([v[0] for v in src.sample(coords)], dtype=float)
        if src.nodata is not None:
            sampled[sampled == src.nodata] = np.nan
        sampled[sampled < -100] = np.nan

    df["target_magt"] = sampled

    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLS + ["target_magt"]).reset_index(drop=True)
    print(f"      После чистки NaN: {len(df)}/{n_before}")
    if len(df) > 0:
        print(f"      MAGT диапазон: {df['target_magt'].min():.2f} ... {df['target_magt'].max():.2f} °C")
    return df

# ============================================================
# 2. Обучение
# ============================================================
def train(df):
    X = df[FEATURE_COLS].values
    y = df["target_magt"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n[3/5] Обучаю XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # Метрики регрессии
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    print("\n" + "=" * 50)
    print("МЕТРИКИ РЕГРЕССИИ (target = Obu MAGT)")
    print("=" * 50)
    print(f"  RMSE: {rmse:.3f} °C")
    print(f"  MAE:  {mae:.3f} °C")
    print(f"  R²:   {r2:.3f}")

    # Метрики классификации
    y_te_cls = magt_to_class(y_te)
    y_pred_cls = magt_to_class(y_pred)
    f1m = f1_score(y_te_cls, y_pred_cls, average="macro")
    kappa = cohen_kappa_score(y_te_cls, y_pred_cls)
    print("\n" + "=" * 50)
    print("МЕТРИКИ КЛАССИФИКАЦИИ (3 класса через пороги)")
    print("=" * 50)
    print(f"  macro-F1: {f1m:.3f}")
    print(f"  kappa:    {kappa:.3f}")
    print("\n  Классы: 0=нет мерзлоты, 1=деградирующая, 2=устойчивая")
    print(classification_report(y_te_cls, y_pred_cls,
          target_names=["0:none", "1:degrading", "2:stable"], digits=3, zero_division=0))
    print("  Confusion matrix:")
    print(confusion_matrix(y_te_cls, y_pred_cls))

    # Важность фич
    print("\n[4/5] Важность фич:")
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importance.to_string())

    # График важности
    plt.figure(figsize=(8, 7))
    importance.sort_values().plot.barh()
    plt.title("Feature importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close()

    return model, importance


# ============================================================
# 3. Карта предсказаний
# ============================================================
def make_map(df, model):
    print("\n[5/5] Строю карту предсказаний...")
    df = df.copy()
    df["pred_magt"] = model.predict(df[FEATURE_COLS].values)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Карта Obu (target)
    sc1 = axes[0].scatter(df["lon"], df["lat"], c=df["target_magt"],
                          cmap="coolwarm_r", s=2, vmin=-8, vmax=2)
    axes[0].set_title("Target: Obu 2019 MAGT")
    axes[0].set_xlabel("lon"); axes[0].set_ylabel("lat")
    plt.colorbar(sc1, ax=axes[0], label="MAGT °C")

    # Карта предсказаний
    sc2 = axes[1].scatter(df["lon"], df["lat"], c=df["pred_magt"],
                          cmap="coolwarm_r", s=2, vmin=-8, vmax=2)
    axes[1].set_title("Predicted MAGT (XGBoost)")
    axes[1].set_xlabel("lon"); axes[1].set_ylabel("lat")
    plt.colorbar(sc2, ax=axes[1], label="MAGT °C")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "nadym_map.png", dpi=120)
    plt.close()
    print(f"      Карта сохранена: {REPORTS_DIR / 'nadym_map.png'}")


# ============================================================
# MAIN
# ============================================================
def main():
    df = load_and_label()
    if len(df) < 100:
        print("\nВНИМАНИЕ: слишком мало валидных ячеек. Возможно, Надым вне области Obu.")
        return
    model, importance = train(df)
    make_map(df, model)
    print("\n[OK] Готово! Смотри reports/feature_importance.png и reports/nadym_map.png")


if __name__ == "__main__":
    main()