"""
БЛОК 1: ALT (active layer thickness) как вторая целевая переменная.

Делает то же, что для MAGT, но target = ALT (глубина протаивания, м).
ALT — прямой индикатор таяния, чувствительнее MAGT около 0°C ("Beyond MAGT").

Шаги:
  1. Берём те же 1744 точки с фичами (scatter_training_dataset.parquet),
     где уже есть координаты.
  2. Сэмплируем ALT из ESA CCI в этих точках.
  3. Обучаем вторую модель: фичи -> ALT.
  4. Spatial block CV (честная оценка обобщения).

Нюансы ALT vs MAGT:
  - ALT всегда положительна (это глубина), диапазон ~0.2-3 м.
  - ALT определена ТОЛЬКО где есть мерзлота; где мерзлоты нет -> NaN.
    Поэтому точек для ALT будет меньше или столько же, сколько для MAGT.

Запуск:
    python3 train_alt.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

POINTS_GEOJSON = Path("data/features/RussiaScatter_3000pts_2021.geojson")
ALT_NC = [f for f in glob.glob("data/esa_cci/*.nc") if "ALT" in f.upper()][0]
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

MAX_UNCERTAINTY = 1.0   # м — ALT uncertainty; ALT измеряется в метрах, порог меньше

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_points():
    with open(POINTS_GEOJSON) as f:
        gj = json.load(f)
    rows = []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        p = dict(ft["properties"])
        p["lon"], p["lat"] = ft["geometry"]["coordinates"]
        rows.append(p)
    return pd.DataFrame(rows)


def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )


def spatial_block_cv(df, target_col, block_deg=5.0, k=5):
    bx = (df["lon"] // block_deg).astype(int)
    by = (df["lat"] // block_deg).astype(int)
    blocks = (bx.astype(str) + "_" + by.astype(str)).values
    ub = np.array(sorted(set(blocks)))
    rng = np.random.RandomState(42); rng.shuffle(ub)
    fold_of = {b: i % k for i, b in enumerate(ub)}
    fold_id = np.array([fold_of[b] for b in blocks])
    X = df[FEATURE_COLS].values
    y = df[target_col].values
    r2s, rmses = [], []
    all_t, all_p = [], []
    for f in range(k):
        te = fold_id == f; tr = ~te
        if te.sum() < 10 or tr.sum() < 50:
            continue
        m = make_model(); m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        r2 = r2_score(y[te], yp); rmse = np.sqrt(mean_squared_error(y[te], yp))
        r2s.append(r2); rmses.append(rmse)
        all_t.append(y[te]); all_p.append(yp)
        print(f"  Fold {f}: train={tr.sum()} test={te.sum()}  R²={r2:.3f}  RMSE={rmse:.3f}")
    return np.mean(r2s), np.mean(rmses), np.concatenate(all_t), np.concatenate(all_p)


def main():
    df = load_points()
    print(f"Точек загружено: {len(df)}")

    # Сэмплируем ALT
    ds = xr.open_dataset(ALT_NC)
    alt = ds["ALT"].isel(time=0)
    unc = ds["ALT_uncertainty"].isel(time=0)
    lats = xr.DataArray(df["lat"].values, dims="p")
    lons = xr.DataArray(df["lon"].values, dims="p")
    df["target_alt"] = alt.sel(lat=lats, lon=lons, method="nearest").values
    df["alt_uncertainty"] = unc.sel(lat=lats, lon=lons, method="nearest").values

    n0 = len(df)
    df = df.dropna(subset=["target_alt"])
    print(f"После удаления NaN ALT (нет мерзлоты): {len(df)}/{n0}")
    df = df[df["alt_uncertainty"].fillna(99) <= MAX_UNCERTAINTY]
    print(f"После фильтра uncertainty <= {MAX_UNCERTAINTY}м: {len(df)}")
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"После NaN в фичах: {len(df)}")

    print(f"\nALT диапазон: {df['target_alt'].min():.2f} ... {df['target_alt'].max():.2f} м")
    print(f"ALT среднее: {df['target_alt'].mean():.2f} м, медиана {df['target_alt'].median():.2f}")

    # Spatial CV
    print("\n=== SPATIAL BLOCK CV для ALT ===")
    r2_sp, rmse_sp, st, sp = spatial_block_cv(df, "target_alt")
    print(f"  СРЕДНЕЕ: R²={r2_sp:.3f}  RMSE={rmse_sp:.3f} м")

    # Random CV для сравнения
    from sklearn.model_selection import KFold
    kf = KFold(5, shuffle=True, random_state=42)
    X = df[FEATURE_COLS].values; y = df["target_alt"].values
    rr2 = []
    for tr, te in kf.split(X):
        m = make_model(); m.fit(X[tr], y[tr])
        rr2.append(r2_score(y[te], m.predict(X[te])))
    print(f"\n  Random CV R²={np.mean(rr2):.3f} (для сравнения)")
    print(f"  Spatial CV R²={r2_sp:.3f}")
    if r2_sp > 0.5:
        print("  ВЫВОД: ALT-модель обобщается. Вторая цель работает.")
    elif r2_sp > 0.2:
        print("  ВЫВОД: ALT обобщается слабее MAGT (ожидаемо — ALT шумнее).")
    else:
        print("  ВЫВОД: ALT обобщается плохо. ALT действительно труднее MAGT.")

    # Финальная модель
    model = make_model()
    model.fit(X, y)
    joblib.dump(model, "models_alt_xgb.joblib")
    print("\nМодель ALT сохранена: models_alt_xgb.joblib")

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nВажность фич для ALT (топ-10):")
    print(imp.head(10).to_string())

    # График
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(st, sp, alpha=0.4, s=12, color="darkorange")
    lims = [0, df["target_alt"].max() * 1.1]
    axes[0].plot(lims, lims, "r--")
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("True ALT (ESA CCI) м"); axes[0].set_ylabel("Predicted ALT м")
    axes[0].set_title(f"ALT Spatial CV\nR²={r2_sp:.3f} RMSE={rmse_sp:.3f}м")
    axes[0].grid(alpha=0.3)
    imp.sort_values().plot.barh(ax=axes[1])
    axes[1].set_title("Важность фич (ALT)")
    plt.tight_layout()
    plt.savefig(REPORTS / "alt_spatial_cv.png", dpi=120)
    print(f"\nГрафик: {REPORTS / 'alt_spatial_cv.png'}")


if __name__ == "__main__":
    main()