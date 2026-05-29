"""
LORO Cross-Validation на 5 регионах.

Leave-One-Region-Out: обучаем на 4 регионах, тестируем на 5-м, по кругу.
Это ЧЕСТНАЯ оценка обобщающей способности.

С континуумом из 5 перекрывающихся регионов (Tiksi -10.9...-6.6, Yakutsk -9...-2.5,
Chara -9...+0.8, Vorkuta -2.8...+1.6, Nadym -2.2...+1.0) ожидаем положительный R²,
в отличие от Шага 4 (два изолированных региона дали R²=-18).

Сравниваем три набора фич: FULL, SATELLITE, CLIMATE.

Запуск:
    python3 loro_5regions.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

REGIONS = ["Nadym", "Vorkuta", "Chara", "Yakutsk", "Tiksi"]
YEAR = 2021
FEAT_DIR = Path("data/features")
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

SATELLITE_FEATURES = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
]
CLIMATE_FEATURES = ["MAAT", "MAP", "era5_temp", "era5_precip"]
FULL_FEATURES = SATELLITE_FEATURES + CLIMATE_FEATURES


def load_region(name):
    path = FEAT_DIR / f"{name}_{YEAR}_features.geojson"
    gdf = gpd.read_file(path).set_crs("EPSG:3857", allow_override=True)
    cen = gdf.geometry.centroid.to_crs("EPSG:4326")
    lons, lats = cen.x.values, cen.y.values
    with rasterio.open(OBU_TIF) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        magt = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        magt[magt == src.nodata] = np.nan
        magt[magt < -100] = np.nan
    df = pd.DataFrame({c: gdf[c].values for c in FULL_FEATURES if c in gdf.columns})
    df["target_magt"] = magt
    df["region"] = name
    df = df.dropna(subset=FULL_FEATURES + ["target_magt"]).reset_index(drop=True)
    return df


def fit_eval(train_df, test_df, features):
    m = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )
    m.fit(train_df[features].values, train_df["target_magt"].values)
    yp = m.predict(test_df[features].values)
    yt = test_df["target_magt"].values
    return {
        "rmse": np.sqrt(mean_squared_error(yt, yp)),
        "mae": mean_absolute_error(yt, yp),
        "r2": r2_score(yt, yp),
        "true": yt, "pred": yp,
    }


def main():
    print("Загружаю 5 регионов...")
    data = {r: load_region(r) for r in REGIONS}
    full = pd.concat(data.values(), ignore_index=True)
    print(f"Всего ячеек: {len(full)}")
    print(f"MAGT диапазон: {full['target_magt'].min():.2f} ... {full['target_magt'].max():.2f}\n")

    # LORO
    print("=" * 70)
    print("LEAVE-ONE-REGION-OUT (обучение на 4, тест на 1)")
    print("=" * 70)
    print(f"\n{'Holdout':<10} {'FULL R²':>9} {'FULL RMSE':>10} {'SAT R²':>9} {'CLIM R²':>9}")
    print("-" * 50)

    loro = {"FULL": [], "SATELLITE": [], "CLIMATE": []}
    preds_full = {}

    for holdout in REGIONS:
        train = pd.concat([data[r] for r in REGIONS if r != holdout], ignore_index=True)
        test = data[holdout]
        r_full = fit_eval(train, test, FULL_FEATURES)
        r_sat = fit_eval(train, test, SATELLITE_FEATURES)
        r_clim = fit_eval(train, test, CLIMATE_FEATURES)
        loro["FULL"].append(r_full["r2"])
        loro["SATELLITE"].append(r_sat["r2"])
        loro["CLIMATE"].append(r_clim["r2"])
        preds_full[holdout] = r_full
        print(f"{holdout:<10} {r_full['r2']:>9.3f} {r_full['rmse']:>10.3f} "
              f"{r_sat['r2']:>9.3f} {r_clim['r2']:>9.3f}")

    print("-" * 50)
    print(f"{'СРЕДНЕЕ':<10} {np.mean(loro['FULL']):>9.3f} {'':>10} "
          f"{np.mean(loro['SATELLITE']):>9.3f} {np.mean(loro['CLIMATE']):>9.3f}")

    # Интерпретация
    avg_full = np.mean(loro["FULL"])
    print("\n" + "=" * 70)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 70)
    print(f"\n  Средний LORO R² (FULL): {avg_full:.3f}")
    print(f"  (в Шаге 4 на 2 изолированных регионах было: -27.7)")
    if avg_full > 0.7:
        print("  ВЫВОД: модель ХОРОШО обобщается на новый регион. Континуум сработал!")
    elif avg_full > 0.5:
        print("  ВЫВОД: модель ПРИЛИЧНО обобщается. Континуум помог.")
    elif avg_full > 0.3:
        print("  ВЫВОД: умеренное обобщение. Возможно, нужны ещё регионы.")
    elif avg_full > 0:
        print("  ВЫВОД: слабое, но положительное обобщение.")
    else:
        print("  ВЫВОД: всё ещё не обобщается. Проблема глубже покрытия.")

    print(f"\n  FULL vs SATELLITE vs CLIMATE (средние):")
    print(f"    FULL={np.mean(loro['FULL']):.3f}  SAT={np.mean(loro['SATELLITE']):.3f}  CLIM={np.mean(loro['CLIMATE']):.3f}")
    if np.mean(loro["SATELLITE"]) > np.mean(loro["CLIMATE"]):
        print("    -> Спутник обобщается ЛУЧШЕ климата. Физический сигнал переносим!")
    else:
        print("    -> Климат обобщается лучше. Спутник пока слабее при переносе.")

    # График: предсказания по каждому holdout-региону
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, holdout in zip(axes, REGIONS):
        r = preds_full[holdout]
        idx = np.random.RandomState(42).choice(len(r["true"]), min(3000, len(r["true"])), replace=False)
        ax.scatter(r["true"][idx], r["pred"][idx], alpha=0.3, s=5)
        lims = [-12, 2]
        ax.plot(lims, lims, "r--", lw=1.2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("True (Obu)")
        ax.set_ylabel("Pred")
        ax.set_title(f"{holdout}\nR²={r['r2']:.2f} RMSE={r['rmse']:.2f}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "loro_5regions.png", dpi=110)
    print(f"\nГрафик: {REPORTS_DIR / 'loro_5regions.png'}")

    # Сохраняем сводку
    summary = pd.DataFrame({
        "holdout": REGIONS,
        "full_r2": loro["FULL"],
        "sat_r2": loro["SATELLITE"],
        "clim_r2": loro["CLIMATE"],
    })
    summary.to_csv(REPORTS_DIR / "loro_5regions.csv", index=False)
    print(f"Сводка: {REPORTS_DIR / 'loro_5regions.csv'}")


if __name__ == "__main__":
    main()