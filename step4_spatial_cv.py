"""
ШАГ 4. Spatial cross-validation (leave-one-region-out).

Проверяет РЕАЛЬНУЮ обобщающую способность:
  - Обучаем на Надыме -> тестируем на Якутске
  - Обучаем на Якутске -> тестируем на Надыме
  - Сравниваем с random split (где train/test перемешаны)

Если R² на spatial CV резко падает vs random split ->
модель запоминает регионы, а не учит физику.

Дополнительно: тест с климатом и без, чтобы увидеть,
держится ли спутниковая модель лучше при переносе на новый регион.

Запуск:
    python3 step4_spatial_cv.py
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

FEATURES = {
    "Nadym":   Path("data/features/Nadym_2022_features.geojson"),
    "Yakutsk": Path("data/features/Yakutsk_2022_features.geojson"),
}
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

SATELLITE_FEATURES = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "soil_oc", "soil_bd", "soil_clay",
]
CLIMATE_FEATURES = ["MAAT", "MAP", "era5_temp", "era5_precip"]
FULL_FEATURES = SATELLITE_FEATURES + CLIMATE_FEATURES


def load_region(name, path):
    gdf = gpd.read_file(path).set_crs("EPSG:3857", allow_override=True)
    cen = gdf.geometry.centroid.to_crs("EPSG:4326")
    lons, lats = cen.x.values, cen.y.values
    with rasterio.open(OBU_TIF) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        sampled = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        sampled[sampled == src.nodata] = np.nan
        sampled[sampled < -100] = np.nan
    df = pd.DataFrame({c: gdf[c].values for c in FULL_FEATURES if c in gdf.columns})
    df["target_magt"] = sampled
    df["region"] = name
    df = df.dropna(subset=FULL_FEATURES + ["target_magt"]).reset_index(drop=True)
    return df


def fit_eval(train_df, test_df, features):
    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    model.fit(train_df[features].values, train_df["target_magt"].values)
    yp = model.predict(test_df[features].values)
    yt = test_df["target_magt"].values
    return {
        "rmse": np.sqrt(mean_squared_error(yt, yp)),
        "mae": mean_absolute_error(yt, yp),
        "r2": r2_score(yt, yp),
        "pred": yp, "true": yt,
    }


def main():
    print("Загружаю регионы...")
    nadym = load_region("Nadym", FEATURES["Nadym"])
    yakutsk = load_region("Yakutsk", FEATURES["Yakutsk"])
    full = pd.concat([nadym, yakutsk], ignore_index=True)
    print(f"  Nadym: {len(nadym)}  ({nadym['target_magt'].min():.1f}...{nadym['target_magt'].max():.1f}°C)")
    print(f"  Yakutsk: {len(yakutsk)}  ({yakutsk['target_magt'].min():.1f}...{yakutsk['target_magt'].max():.1f}°C)")

    results = {}

    # --- 1. RANDOM SPLIT (baseline, "оптимистичный") ---
    print("\n" + "=" * 60)
    print("1. RANDOM SPLIT (train/test перемешаны — оптимистичная оценка)")
    print("=" * 60)
    tr, te = train_test_split(full, test_size=0.2, random_state=42)
    for fname, feats in [("FULL", FULL_FEATURES), ("SATELLITE", SATELLITE_FEATURES)]:
        r = fit_eval(tr, te, feats)
        print(f"  [{fname}] RMSE={r['rmse']:.3f}  R²={r['r2']:.3f}")
        results[f"random_{fname}"] = r

    # --- 2. SPATIAL CV: train Nadym -> test Yakutsk ---
    print("\n" + "=" * 60)
    print("2. SPATIAL: обучение на НАДЫМЕ -> тест на ЯКУТСКЕ")
    print("=" * 60)
    print("   (модель видит только тёплую мерзлоту, предсказывает холодную)")
    for fname, feats in [("FULL", FULL_FEATURES), ("SATELLITE", SATELLITE_FEATURES)]:
        r = fit_eval(nadym, yakutsk, feats)
        print(f"  [{fname}] RMSE={r['rmse']:.3f}  R²={r['r2']:.3f}")
        results[f"n2y_{fname}"] = r

    # --- 3. SPATIAL CV: train Yakutsk -> test Nadym ---
    print("\n" + "=" * 60)
    print("3. SPATIAL: обучение на ЯКУТСКЕ -> тест на НАДЫМЕ")
    print("=" * 60)
    print("   (модель видит только холодную мерзлоту, предсказывает тёплую)")
    for fname, feats in [("FULL", FULL_FEATURES), ("SATELLITE", SATELLITE_FEATURES)]:
        r = fit_eval(yakutsk, nadym, feats)
        print(f"  [{fname}] RMSE={r['rmse']:.3f}  R²={r['r2']:.3f}")
        results[f"y2n_{fname}"] = r

    # --- ИНТЕРПРЕТАЦИЯ ---
    print("\n" + "=" * 60)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 60)
    print(f"\n  Random split FULL R²:        {results['random_FULL']['r2']:.3f}  (оптимистично)")
    print(f"  Nadym->Yakutsk FULL R²:      {results['n2y_FULL']['r2']:.3f}")
    print(f"  Yakutsk->Nadym FULL R²:      {results['y2n_FULL']['r2']:.3f}")
    print()
    avg_spatial = (results['n2y_FULL']['r2'] + results['y2n_FULL']['r2']) / 2
    print(f"  Средний spatial R²: {avg_spatial:.3f}")
    drop = results['random_FULL']['r2'] - avg_spatial
    print(f"  Падение vs random: {drop:.3f}")
    print()
    if avg_spatial > 0.7:
        print("  ВЫВОД: модель ХОРОШО обобщается на новый регион. Учит физику.")
    elif avg_spatial > 0.3:
        print("  ВЫВОД: модель ЧАСТИЧНО обобщается. Есть и физика, и запоминание.")
    elif avg_spatial > 0:
        print("  ВЫВОД: модель СЛАБО обобщается. В основном запоминала регионы.")
    else:
        print("  ВЫВОД: модель НЕ обобщается (R²<0). На новом регионе бесполезна.")
        print("         Высокий random R² был иллюзией от двухкластерности.")

    print(f"\n  Сравнение FULL vs SATELLITE при переносе:")
    print(f"    Nadym->Yakutsk: FULL={results['n2y_FULL']['r2']:.3f}  SAT={results['n2y_SATELLITE']['r2']:.3f}")
    print(f"    Yakutsk->Nadym: FULL={results['y2n_FULL']['r2']:.3f}  SAT={results['y2n_SATELLITE']['r2']:.3f}")

    # --- ГРАФИК ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    scenarios = [
        ("random_FULL", "Random split\n(оптимистично)"),
        ("n2y_FULL", "Nadym → Yakutsk"),
        ("y2n_FULL", "Yakutsk → Nadym"),
    ]
    for ax, (key, title) in zip(axes, scenarios):
        r = results[key]
        # подвыборка для скорости отрисовки
        idx = np.random.RandomState(42).choice(len(r["true"]), min(5000, len(r["true"])), replace=False)
        ax.scatter(r["true"][idx], r["pred"][idx], alpha=0.3, s=5)
        lims = [-10, 2]
        ax.plot(lims, lims, "r--", lw=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True MAGT (Obu) °C")
        ax.set_ylabel("Predicted °C")
        ax.set_title(f"{title}\nR²={r['r2']:.3f}  RMSE={r['rmse']:.2f}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "step4_spatial_cv.png", dpi=120)
    print(f"\nГрафик: {REPORTS_DIR / 'step4_spatial_cv.png'}")


if __name__ == "__main__":
    main()