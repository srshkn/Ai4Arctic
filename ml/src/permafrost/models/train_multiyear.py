"""
ВАРИАНТ A: Мультигодовое обучение (Путь 1, на сетке).

Идея: вместо обучения на 1 годе (2021) учим на ТРЁХ годах (2010,2021,2023)
вместе, каждая точка со своим target ESA CCI того года. Модель калибруется
на межгодовой изменчивости -> должна устойчивее ловить тренд.

Шаги:
  1. Для каждого года (2010,2021,2023): берём фичи сетки + сэмплируем ESA CCI T10m.
  2. Объединяем в один датасет (точки разных лет вместе).
  3. Обучаем XGBoost. Spatial CV (с учётом, что одна точка в разные годы — разные сэмплы).
  4. КЛЮЧЕВАЯ ПРОВЕРКА: прогоняем по 7 годам, считаем R² линейности тренда.
     Если вырос с 0.13 — мультигодовое обучение помогло.

Запуск:
    python3 train_multiyear.py
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
from sklearn.metrics import r2_score, mean_squared_error

FEAT_DIR = Path("data/features")
REPORTS = Path("reports")
TRAIN_YEARS = [2010, 2021, 2023]   # годы с ESA CCI target
ALL_YEARS = [2010, 2013, 2015, 2017, 2019, 2021, 2023]  # для проверки тренда

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_grid(year):
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
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def sample_esa(df, year):
    cands = [f for f in glob.glob("data/esa_cci/*.nc") if "GTD" in f.upper() and str(year) in f]
    if not cands:
        return None
    ds = xr.open_dataset(cands[0])
    t10 = ds["T10m"].isel(time=0)
    la = xr.DataArray(df["lat"].values, dims="p")
    lo = xr.DataArray(df["lon"].values, dims="p")
    return t10.sel(lat=la, lon=lo, method="nearest").values


def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1)


def spatial_block_cv(df, block_deg=5.0, k=5):
    bx = (df["lon"] // block_deg).astype(int)
    by = (df["lat"] // block_deg).astype(int)
    blocks = (bx.astype(str) + "_" + by.astype(str)).values
    ub = np.array(sorted(set(blocks)))
    rng = np.random.RandomState(42); rng.shuffle(ub)
    fold_of = {b: i % k for i, b in enumerate(ub)}
    fid = np.array([fold_of[b] for b in blocks])
    X = df[FEATURE_COLS].values; y = df["target"].values
    r2s = []
    for f in range(k):
        te = fid == f; tr = ~te
        if te.sum() < 20 or tr.sum() < 100: continue
        m = make_model(); m.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], m.predict(X[te])))
    return np.mean(r2s)


def main():
    # 1. Собираем мультигодовой датасет
    parts = []
    for y in TRAIN_YEARS:
        df = load_grid(y)
        tgt = sample_esa(df, y)
        if tgt is None:
            print(f"ESA CCI {y} нет, пропуск"); continue
        df["target"] = tgt
        df["year"] = y
        df = df.dropna(subset=["target"])
        # фильтр мерзлоты + разумный диапазон
        df = df[(df["target"] > -25) & (df["target"] < 10)]
        parts.append(df[FEATURE_COLS + ["lon", "lat", "target", "year"]])
        print(f"{y}: {len(df)} точек с target")

    data = pd.concat(parts, ignore_index=True)
    print(f"\nМультигодовой датасет: {len(data)} точек ({len(TRAIN_YEARS)} года)")

    # 2. Spatial CV (обобщение)
    print("\nSpatial block CV (мультигодовая модель)...")
    r2_sp = spatial_block_cv(data)
    print(f"  Spatial CV R² = {r2_sp:.3f}")

    # 3. Финальная модель на всех годах
    model = make_model()
    model.fit(data[FEATURE_COLS].values, data["target"].values)
    joblib.dump(model, "models_multiyear_xgb.joblib")
    print("  Модель сохранена: models_multiyear_xgb.joblib")

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n  Важность (топ-6):")
    print(imp.head(6).to_string())

    # 4. КЛЮЧЕВАЯ ПРОВЕРКА: R² линейности тренда по 7 годам
    print("\n" + "=" * 60)
    print("ПРОВЕРКА: устойчивость тренда (R² линейности)")
    print("=" * 60)
    preds = {}
    for y in ALL_YEARS:
        df = load_grid(y)
        df["k"] = df["lon"].round(2).astype(str) + "_" + df["lat"].round(2).astype(str)
        df = df.drop_duplicates(subset="k")
        df[f"magt_{y}"] = model.predict(df[FEATURE_COLS].values)
        preds[y] = df[["k", "lon", "lat", f"magt_{y}"]].set_index("k")

    common = preds[ALL_YEARS[0]]
    for y in ALL_YEARS[1:]:
        common = common.join(preds[y].drop(columns=["lon", "lat"]), how="inner")
    print(f"Общих точек: {len(common)}")

    magt_cols = [f"magt_{y}" for y in ALL_YEARS]
    yrs = np.array(ALL_YEARS, dtype=float)
    Y = common[magt_cols].values
    fit = np.polyfit(yrs, Y.T, 1)
    slopes = fit[0]
    pred_lin = np.outer(slopes, yrs) + fit[1][:, None]
    ss_res = ((Y - pred_lin) ** 2).sum(axis=1)
    ss_tot = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    r2_lin = 1 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)

    print(f"\n  Средний тренд: {np.nanmean(slopes)*10:+.3f} °C/декада")
    print(f"  Медиана R² линейности: {np.nanmedian(r2_lin):.3f}")
    print(f"  Точек с устойчивым трендом (R²>0.5): {np.nanmean(r2_lin>0.5)*100:.0f}%")
    print(f"\n  Было (модель на 1 годе): медиана R² линейности = 0.13")
    if np.nanmedian(r2_lin) > 0.3:
        print("  ВЫВОД: мультигодовое обучение ЗАМЕТНО улучшило устойчивость тренда!")
    elif np.nanmedian(r2_lin) > 0.18:
        print("  ВЫВОД: умеренное улучшение устойчивости тренда.")
    else:
        print("  ВЫВОД: улучшение слабое — шум фундаментален, нужна явная инерция (лаги/LSTM).")

    # Сравнение с ESA CCI трендом
    esa_cols = []
    for y in [2010, 2021, 2023]:
        common[f"esa_{y}"] = sample_esa(
            common.reset_index().rename(columns={"lon":"lon","lat":"lat"}), y)
        esa_cols.append(f"esa_{y}")
    v = common.dropna(subset=esa_cols)
    if len(v) > 100:
        ey = np.array([2010, 2021, 2023], dtype=float)
        esa_slopes = np.polyfit(ey, v[esa_cols].values.T, 1)[0]
        our_slopes_v = np.polyfit(yrs, v[magt_cols].values.T, 1)[0]
        from scipy.stats import pearsonr
        r, _ = pearsonr(our_slopes_v, esa_slopes)
        print(f"\n  Корреляция тренда с ESA CCI: r={r:.3f} (было 0.02 на статичной модели)")

    print(f"\nСохранено: models_multiyear_xgb.joblib")


if __name__ == "__main__":
    main()