"""
КРИВАЯ ОБУЧЕНИЯ (learning curve) — доказательство достаточности данных.

Идея: обучаем модель на подвыборках растущего размера (20%..100%) и смотрим
качество на отложенной части. Если кривая выходит на ПЛАТО — данных достаточно
(добавление новых точек не улучшает модель). Если ещё растёт — данных мало.

Это стандартный научный способ ответить на вопрос "хватает ли данных?".

Использует те же обучающие точки, что основная модель MAGT.

Запуск:
    python3 learning_curve.py
"""

from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_training_data():
    """Загружает обучающую выборку MAGT (точки + target).
    Пытается найти готовый файл; иначе собирает из сетки 2021 + ESA CCI.
    """
    # Вариант 1: если есть сохранённая обучающая таблица
    for cand in ["data/training_magt.csv", "reports/training_magt.csv"]:
        if Path(cand).exists():
            df = pd.read_csv(cand)
            if "target" in df.columns or "T10m" in df.columns:
                tcol = "target" if "target" in df.columns else "T10m"
                df = df.rename(columns={tcol: "target"})
                return df.dropna(subset=FEATURE_COLS + ["target"])

    # Вариант 2: собрать из сетки 2021 + ESA CCI (как train_scatter.py)
    print("Готовой таблицы нет — собираю из сетки 2021 + ESA CCI...")
    import xarray as xr
    with open("data/features/RussiaGrid_0.25deg_2021.geojson") as fh:
        gj = json.load(fh)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lo, la = ft["geometry"]["coordinates"]
        lons.append(lo); lats.append(la)
    df = pd.DataFrame(rows); df["lon"] = lons; df["lat"] = lats
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    # target из ESA CCI GTD 2021
    f = [x for x in glob.glob("data/esa_cci/*.nc") if "GTD" in x.upper() and "2021" in x][0]
    ds = xr.open_dataset(f)
    t10 = ds["T10m"].isel(time=0)
    la = xr.DataArray(df["lat"].values, dims="p")
    lo = xr.DataArray(df["lon"].values, dims="p")
    df["target"] = t10.sel(lat=la, lon=lo, method="nearest").values
    df = df.dropna(subset=["target"])
    df = df[(df["target"] > -25) & (df["target"] < 10)]
    return df


def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1)


def main():
    df = load_training_data()
    X = df[FEATURE_COLS].values
    y = df["target"].values
    n_total = len(df)
    print(f"Всего обучающих точек: {n_total}")

    # Доли выборки для кривой
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0]
    rng = np.random.RandomState(42)

    # 5-fold CV: для каждой доли усредняем R² по фолдам
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for frac in fractions:
        r2_folds, rmse_folds = [], []
        for tr_idx, te_idx in kf.split(X):
            # из обучающей части берём долю frac
            n_sub = max(50, int(len(tr_idx) * frac))
            sub = rng.choice(tr_idx, n_sub, replace=False)
            m = make_model()
            m.fit(X[sub], y[sub])
            pred = m.predict(X[te_idx])
            r2_folds.append(r2_score(y[te_idx], pred))
            rmse_folds.append(np.sqrt(mean_squared_error(y[te_idx], pred)))
        n_used = int(n_total * 0.8 * frac)  # примерное число использованных точек
        results.append({
            "fraction": frac,
            "n_points": n_used,
            "r2_mean": np.mean(r2_folds),
            "r2_std": np.std(r2_folds),
            "rmse_mean": np.mean(rmse_folds),
        })
        print(f"  доля {frac:.0%} (~{n_used} точек): "
              f"R² = {np.mean(r2_folds):.3f} ± {np.std(r2_folds):.3f}, "
              f"RMSE = {np.mean(rmse_folds):.2f}°C")

    res = pd.DataFrame(results)

    # Анализ плато: насколько выросло R² за последнюю половину
    mid = res[res["fraction"] >= 0.5]["r2_mean"].values
    gain_late = mid[-1] - mid[0]
    print(f"\nПрирост R² от 50% к 100% данных: {gain_late:+.3f}")
    if gain_late < 0.02:
        print("ВЫВОД: кривая вышла на ПЛАТО — данных ДОСТАТОЧНО.")
        print("Добавление новых точек не улучшает модель значимо.")
    elif gain_late < 0.05:
        print("ВЫВОД: близко к плато — данных в целом достаточно.")
    else:
        print("ВЫВОД: кривая ещё растёт — больше данных помогло бы.")

    # === График ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # R² с доверительным коридором
    ax1.plot(res["n_points"], res["r2_mean"], "o-", color="#185FA5", lw=2, markersize=7)
    ax1.fill_between(res["n_points"],
                     res["r2_mean"] - res["r2_std"],
                     res["r2_mean"] + res["r2_std"],
                     alpha=0.2, color="#378ADD")
    ax1.set_xlabel("Число обучающих точек")
    ax1.set_ylabel("R² (5-fold CV)")
    ax1.set_title("Кривая обучения: R² vs объём данных")
    ax1.grid(alpha=0.3)
    ax1.axhline(res["r2_mean"].iloc[-1], ls="--", color="gray", alpha=0.5)

    # RMSE
    ax2.plot(res["n_points"], res["rmse_mean"], "s-", color="#993C1D", lw=2, markersize=7)
    ax2.set_xlabel("Число обучающих точек")
    ax2.set_ylabel("RMSE, °C")
    ax2.set_title("Кривая обучения: ошибка vs объём данных")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS / "learning_curve.png", dpi=150)
    print(f"\nСохранено: {REPORTS / 'learning_curve.png'}")
    res.to_csv(REPORTS / "learning_curve_data.csv", index=False)
    print(f"Данные: {REPORTS / 'learning_curve_data.csv'}")


if __name__ == "__main__":
    main()