"""
ГЛАВНАЯ ПРОВЕРКА: обучение на распределённой выборке (1744 точки)
со честной пространственной валидацией.

Сравниваем:
  A. Random CV  — точки перемешаны (оптимистично, может завышать)
  B. Spatial block CV — Россия делится на гео-блоки, train и test
     в РАЗНЫХ блоках (честно: соседние точки не "подсматривают")

Если spatial CV даёт положительный приличный R² — стратегия Способа A
сработала, модель ОБОБЩАЕТСЯ (в отличие от LORO=-26 на 5 регионах).

Запуск:
    python3 train_scatter.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    f1_score, cohen_kappa_score, classification_report, confusion_matrix
)

DATA = Path("data/features/scatter_training_dataset.parquet")
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def to_class(magt):
    c = np.zeros(len(magt), dtype=int)
    c[magt <= 0.0] = 1
    c[magt <= -1.5] = 2
    return c


def make_model():
    return xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=42, n_jobs=-1,
    )


def eval_fold(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    return {
        "rmse": np.sqrt(mean_squared_error(yte, yp)),
        "mae": mean_absolute_error(yte, yp),
        "r2": r2_score(yte, yp),
        "pred": yp, "true": yte,
    }


# ============================================================
# A. RANDOM CV
# ============================================================
def random_cv(df, k=5):
    X = df[FEATURE_COLS].values
    y = df["target_magt"].values
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2s, rmses = [], []
    for tr, te in kf.split(X):
        r = eval_fold(make_model(), X[tr], y[tr], X[te], y[te])
        r2s.append(r["r2"]); rmses.append(r["rmse"])
    return np.mean(r2s), np.mean(rmses)


# ============================================================
# B. SPATIAL BLOCK CV
# ============================================================
def assign_blocks(df, block_deg=5.0):
    """Делим пространство на гео-блоки block_deg x block_deg градусов."""
    bx = (df["lon"] // block_deg).astype(int)
    by = (df["lat"] // block_deg).astype(int)
    return (bx.astype(str) + "_" + by.astype(str)).values


def spatial_block_cv(df, block_deg=5.0, k=5):
    """
    Блоки группируем в k фолдов. Train и test — в разных блоках.
    """
    blocks = assign_blocks(df, block_deg)
    unique_blocks = np.array(sorted(set(blocks)))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_blocks)
    # раскидываем блоки по k фолдам
    fold_of_block = {b: i % k for i, b in enumerate(unique_blocks)}
    fold_id = np.array([fold_of_block[b] for b in blocks])

    X = df[FEATURE_COLS].values
    y = df["target_magt"].values
    results = []
    all_true, all_pred = [], []
    for f in range(k):
        te = fold_id == f
        tr = ~te
        if te.sum() < 10 or tr.sum() < 50:
            continue
        r = eval_fold(make_model(), X[tr], y[tr], X[te], y[te])
        results.append(r)
        all_true.append(r["true"]); all_pred.append(r["pred"])
        print(f"  Fold {f}: train={tr.sum()} test={te.sum()}  R²={r['r2']:.3f}  RMSE={r['rmse']:.2f}")
    r2_mean = np.mean([r["r2"] for r in results])
    rmse_mean = np.mean([r["rmse"] for r in results])
    return r2_mean, rmse_mean, np.concatenate(all_true), np.concatenate(all_pred)


# ============================================================
# MAIN
# ============================================================
def main():
    df = pd.read_parquet(DATA)
    print(f"Точек: {len(df)}")
    print(f"Блоков по 5°: {len(set(assign_blocks(df, 5.0)))}")
    print(f"Блоков по 10°: {len(set(assign_blocks(df, 10.0)))}\n")

    print("=" * 60)
    print("A. RANDOM CV (оптимистичная оценка)")
    print("=" * 60)
    r2_rand, rmse_rand = random_cv(df)
    print(f"  R²={r2_rand:.3f}  RMSE={rmse_rand:.3f}")

    print("\n" + "=" * 60)
    print("B. SPATIAL BLOCK CV (честная оценка, блоки 5°)")
    print("=" * 60)
    r2_sp, rmse_sp, st, sp = spatial_block_cv(df, block_deg=5.0, k=5)
    print(f"  СРЕДНЕЕ: R²={r2_sp:.3f}  RMSE={rmse_sp:.3f}")

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ И ВЫВОД")
    print("=" * 60)
    print(f"  Random CV:        R²={r2_rand:.3f}")
    print(f"  Spatial block CV: R²={r2_sp:.3f}")
    print(f"  (На 5 регионах LORO было: R²=-26)")
    print()
    if r2_sp > 0.7:
        print("  ВЫВОД: ОТЛИЧНОЕ обобщение! Способ A полностью сработал.")
    elif r2_sp > 0.5:
        print("  ВЫВОД: ХОРОШЕЕ обобщение. Распределённая выборка решила проблему.")
    elif r2_sp > 0.3:
        print("  ВЫВОД: умеренное обобщение — большой прогресс vs LORO=-26.")
    elif r2_sp > 0:
        print("  ВЫВОД: слабое, но ПОЛОЖИТЕЛЬНОЕ обобщение (vs -26 раньше).")
    else:
        print("  ВЫВОД: всё ещё не обобщается — нужно разбираться глубже.")

    # Финальная модель на всех данных + важность + классификация
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ МОДЕЛЬ (на всех точках)")
    print("=" * 60)
    model = make_model()
    X = df[FEATURE_COLS].values
    y = df["target_magt"].values
    model.fit(X, y)
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("Важность фич (топ-10):")
    print(imp.head(10).to_string())

    # классификация на spatial-предсказаниях
    print("\nКлассификация (на spatial-CV предсказаниях):")
    f1 = f1_score(to_class(st), to_class(sp), average="macro")
    kappa = cohen_kappa_score(to_class(st), to_class(sp))
    print(f"  macro-F1={f1:.3f}  kappa={kappa:.3f}")

    # графики
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].scatter(st, sp, alpha=0.4, s=12)
    axes[0].plot([-13, 6], [-13, 6], "r--")
    axes[0].set_xlim(-13, 6); axes[0].set_ylim(-13, 6)
    axes[0].set_xlabel("True T10m (ESA CCI) °C"); axes[0].set_ylabel("Predicted °C")
    axes[0].set_title(f"Spatial block CV\nR²={r2_sp:.3f} RMSE={rmse_sp:.2f}")
    axes[0].grid(alpha=0.3)

    imp.sort_values().plot.barh(ax=axes[1])
    axes[1].set_title("Feature importance")
    plt.tight_layout()
    plt.savefig(REPORTS / "scatter_spatial_cv.png", dpi=120)
    print(f"\nГрафик: {REPORTS / 'scatter_spatial_cv.png'}")

    import joblib
    joblib.dump(model, "models_scatter_xgb.joblib")
    print("Модель сохранена: models_scatter_xgb.joblib")


if __name__ == "__main__":
    main()