"""
СВОДНЫЙ ВАЛИДАЦИОННЫЙ РАЗДЕЛ.

Собирает ВСЕ проверки модели в единую таблицу и многопанельный рисунок:
  Панель A: Spatial block CV (MAGT) — обобщение
  Панель B: Spatial block CV (ALT) — обобщение второй цели
  Панель C: vs реальные boreholes — соответствие реальности
  Панель D: vs Obu 2019 — согласованность с независимой картой
  + итоговая таблица метрик
  + текстовый вывод для отчёта

Использует уже сохранённые результаты (reports/*.csv) и модели.

Запуск:
    python3 validation_summary.py
"""

from pathlib import Path
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

REPORTS = Path("reports")
DATA = Path("data/features/scatter_training_dataset.parquet")

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


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
    X = df[FEATURE_COLS].values; y = df[target_col].values
    all_t, all_p = [], []
    for f in range(k):
        te = fold_id == f; tr = ~te
        if te.sum() < 10 or tr.sum() < 50:
            continue
        m = make_model(); m.fit(X[tr], y[tr])
        all_t.append(y[te]); all_p.append(m.predict(X[te]))
    t = np.concatenate(all_t); p = np.concatenate(all_p)
    return t, p, r2_score(t, p), np.sqrt(mean_squared_error(t, p))


def main():
    rows = []  # для итоговой таблицы

    # === A. Spatial CV MAGT ===
    df = pd.read_parquet(DATA)
    t_magt, p_magt, r2_magt, rmse_magt = spatial_block_cv(df, "target_magt")
    rows.append(["Spatial CV (MAGT)", f"{r2_magt:.3f}", f"{rmse_magt:.2f} °C",
                 "обобщение на новые территории"])

    # === B. Spatial CV ALT ===
    # грузим ALT-датасет если есть, иначе пропускаем
    alt_t = alt_p = None
    try:
        # пересоберём ALT target из точек (как в train_alt)
        import xarray as xr
        POINTS = Path("data/features/RussiaScatter_3000pts_2021.geojson")
        ALT_NC = [f for f in glob.glob("data/esa_cci/*.nc") if "ALT" in f.upper()][0]
        with open(POINTS) as f:
            gj = json.load(f)
        prows = []
        for ft in gj["features"]:
            if ft.get("geometry") is None: continue
            pp = dict(ft["properties"])
            pp["lon"], pp["lat"] = ft["geometry"]["coordinates"]
            prows.append(pp)
        adf = pd.DataFrame(prows)
        ds = xr.open_dataset(ALT_NC)
        alt = ds["ALT"].isel(time=0); unc = ds["ALT_uncertainty"].isel(time=0)
        la = xr.DataArray(adf["lat"].values, dims="p"); lo = xr.DataArray(adf["lon"].values, dims="p")
        adf["target_alt"] = alt.sel(lat=la, lon=lo, method="nearest").values
        adf["alt_unc"] = unc.sel(lat=la, lon=lo, method="nearest").values
        adf = adf.dropna(subset=["target_alt"])
        adf = adf[adf["alt_unc"].fillna(99) <= 1.0]
        adf = adf.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        alt_t, alt_p, r2_alt, rmse_alt = spatial_block_cv(adf, "target_alt")
        rows.append(["Spatial CV (ALT)", f"{r2_alt:.3f}", f"{rmse_alt:.2f} м",
                     "обобщение второй цели"])
    except Exception as e:
        print(f"ALT пропущена: {e}")
        r2_alt = None

    # === C. vs boreholes (читаем сохранённое) ===
    bh = None
    bh_csv = REPORTS / "validate_boreholes.csv"
    if bh_csv.exists():
        bh = pd.read_csv(bh_csv)
        res = bh["pred_magt"] - bh["magt_c"]
        rmse_bh = np.sqrt((res**2).mean())
        r_bh, _ = pearsonr(bh["magt_c"], bh["pred_magt"])
        res_e = bh["esacci_magt"] - bh["magt_c"]
        rmse_bh_e = np.sqrt((res_e**2).mean())
        rows.append(["vs boreholes (наша)", f"corr={r_bh:.2f}", f"{rmse_bh:.2f} °C",
                     "соответствие реальности"])
        rows.append(["vs boreholes (ESA CCI)", "—", f"{rmse_bh_e:.2f} °C",
                     "эталон для сравнения"])

    # === D. vs Obu (читаем сохранённое) ===
    obu = None
    map_csv = REPORTS / "russia_map_data.csv"
    obu_diff_available = (REPORTS / "compare_vs_obu.png").exists()
    # пересчитать корреляцию vs Obu из russia_map_data + Obu tif
    try:
        import rasterio
        from rasterio.warp import transform as warp_transform
        mdf = pd.read_csv(map_csv)
        OBU_TIF = "data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif"
        with rasterio.open(OBU_TIF) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, mdf["lon"].tolist(), mdf["lat"].tolist())
            ob = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
            ob[ob == src.nodata] = np.nan; ob[ob < -100] = np.nan
        mdf["obu"] = ob
        vv = mdf.dropna(subset=["pred_magt", "obu"])
        r_obu, _ = pearsonr(vv["pred_magt"], vv["obu"])
        rmse_obu = np.sqrt(((vv["pred_magt"] - vv["obu"])**2).mean())
        bias_obu = (vv["pred_magt"] - vv["obu"]).mean()
        rows.append(["vs Obu 2019", f"R²={r_obu**2:.3f}", f"{rmse_obu:.2f} °C",
                     f"согласованность, bias {bias_obu:+.1f}"])
        obu = (vv["obu"].values, vv["pred_magt"].values)
    except Exception as e:
        print(f"Obu сравнение пропущено: {e}")

    # === ИТОГОВАЯ ТАБЛИЦА ===
    summary = pd.DataFrame(rows, columns=["Проверка", "Метрика", "Ошибка", "Что показывает"])
    print("\n" + "=" * 75)
    print("СВОДНАЯ ВАЛИДАЦИЯ МОДЕЛИ")
    print("=" * 75)
    print(summary.to_string(index=False))
    summary.to_csv(REPORTS / "validation_summary.csv", index=False)

    # === МНОГОПАНЕЛЬНЫЙ РИСУНОК ===
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))

    # A: MAGT spatial CV
    ax = axes[0, 0]
    ax.scatter(t_magt, p_magt, alpha=0.4, s=10)
    ax.plot([-13, 6], [-13, 6], "r--")
    ax.set_xlim(-13, 6); ax.set_ylim(-13, 6)
    ax.set_xlabel("ESA CCI T10m, °C"); ax.set_ylabel("Предсказание, °C")
    ax.set_title(f"A. Spatial CV — MAGT\nR²={r2_magt:.3f}, RMSE={rmse_magt:.2f}°C")
    ax.grid(alpha=0.3)

    # B: ALT spatial CV
    ax = axes[0, 1]
    if alt_t is not None:
        ax.scatter(alt_t, alt_p, alpha=0.4, s=10, color="darkorange")
        mx = max(alt_t.max(), alt_p.max())
        ax.plot([0, mx], [0, mx], "r--")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel("ESA CCI ALT, м"); ax.set_ylabel("Предсказание, м")
        ax.set_title(f"B. Spatial CV — ALT\nR²={r2_alt:.3f}, RMSE={rmse_alt:.2f}м")
        ax.grid(alpha=0.3)

    # C: vs boreholes
    ax = axes[1, 0]
    if bh is not None:
        ax.scatter(bh["magt_c"], bh["pred_magt"], alpha=0.7, s=50, label="наша модель")
        ax.scatter(bh["magt_c"], bh["esacci_magt"], alpha=0.7, s=50, marker="^", label="ESA CCI")
        ax.plot([-12, 4], [-12, 4], "r--")
        ax.set_xlim(-12, 4); ax.set_ylim(-12, 4)
        ax.set_xlabel("Реальное MAGT (скважина), °C"); ax.set_ylabel("Предсказание, °C")
        ax.set_title(f"C. vs наземные измерения (n={len(bh)})\nRMSE {rmse_bh:.1f}°C (ESA CCI: {rmse_bh_e:.1f}°C)")
        ax.legend(); ax.grid(alpha=0.3)

    # D: vs Obu
    ax = axes[1, 1]
    if obu is not None:
        idx = np.random.RandomState(42).choice(len(obu[0]), min(6000, len(obu[0])), replace=False)
        ax.scatter(obu[0][idx], obu[1][idx], alpha=0.3, s=4)
        ax.plot([-14, 4], [-14, 4], "r--")
        ax.set_xlim(-14, 4); ax.set_ylim(-14, 4)
        ax.set_xlabel("Obu 2019, °C"); ax.set_ylabel("Наша модель, °C")
        ax.set_title(f"D. vs Obu 2019\nR²={r_obu**2:.3f}, RMSE={rmse_obu:.2f}°C")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS / "validation_summary.png", dpi=130)
    print(f"\nРисунок: {REPORTS / 'validation_summary.png'}")
    print(f"Таблица: {REPORTS / 'validation_summary.csv'}")


if __name__ == "__main__":
    main()