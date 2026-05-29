"""
Сравнение нашей карты (предсказания модели на сетке) с Obu 2019.

Obu 2019 — независимая карта мерзлоты (хоть и родственная ESA CCI).
Сравниваем наши предсказания MAGT с Obu MAGT в узлах сетки.

ВАЖНО: Obu и ESA CCI — РОДСТВЕННЫЕ продукты (та же группа, похожая физика).
Поэтому это НЕ полностью независимая проверка, а проверка согласованности
между двумя версиями физической модели. Это надо честно отметить.

Запуск:
    python3 compare_map_vs_obu.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

MAP_CSV = Path("reports/russia_map_data.csv")   # наши предсказания (lon,lat,pred_magt,pred_class)
OBU_TIF = Path("data/obu2019/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif")
REPORTS = Path("reports")


def main():
    df = pd.read_csv(MAP_CSV)
    print(f"Точек в нашей карте: {len(df)}")

    # Сэмплируем Obu в узлах сетки
    with rasterio.open(OBU_TIF) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, df["lon"].tolist(), df["lat"].tolist())
        obu = np.array([v[0] for v in src.sample(list(zip(xs, ys)))], dtype=float)
        obu[obu == src.nodata] = np.nan
        obu[obu < -100] = np.nan
    df["obu_magt"] = obu

    v = df.dropna(subset=["pred_magt", "obu_magt"]).copy()
    print(f"Точек с Obu (есть мерзлота): {len(v)}")

    # Метрики согласованности
    res = v["pred_magt"] - v["obu_magt"]
    rmse = np.sqrt((res**2).mean())
    bias = res.mean()
    mae = res.abs().mean()
    r, _ = pearsonr(v["pred_magt"], v["obu_magt"])
    print("\n=== Согласованность нашей карты с Obu 2019 ===")
    print(f"  N точек: {len(v)}")
    print(f"  RMSE: {rmse:.2f} °C")
    print(f"  MAE:  {mae:.2f} °C")
    print(f"  bias: {bias:+.2f} °C  (наша - Obu)")
    print(f"  corr: {r:.3f}")
    print(f"  R²:   {r**2:.3f}")
    print("\n  Напоминание: Obu и ESA CCI родственны, поэтому высокая")
    print("  согласованность ожидаема и подтверждает консистентность, а не")
    print("  независимую точность.")

    # Графики
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # scatter согласованности
    idx = np.random.RandomState(42).choice(len(v), min(8000, len(v)), replace=False)
    axes[0].scatter(v["obu_magt"].values[idx], v["pred_magt"].values[idx], alpha=0.3, s=5)
    lims = [-14, 4]
    axes[0].plot(lims, lims, "r--", lw=1.2)
    axes[0].set_xlim(lims); axes[0].set_ylim(lims)
    axes[0].set_xlabel("Obu 2019 MAGT °C")
    axes[0].set_ylabel("Наша модель MAGT °C")
    axes[0].set_title(f"Согласованность\nR²={r**2:.3f}, RMSE={rmse:.2f}, bias={bias:+.2f}")
    axes[0].grid(alpha=0.3)

    # карта Obu
    s1 = axes[1].scatter(v["lon"], v["lat"], c=v["obu_magt"], cmap="coolwarm_r", s=4, vmin=-12, vmax=2)
    axes[1].set_title("Obu 2019 (2000-2016)")
    axes[1].set_xlabel("lon"); axes[1].set_ylabel("lat")
    plt.colorbar(s1, ax=axes[1], shrink=0.6)

    # карта наша
    s2 = axes[2].scatter(v["lon"], v["lat"], c=v["pred_magt"], cmap="coolwarm_r", s=4, vmin=-12, vmax=2)
    axes[2].set_title("Наша модель (2021)")
    axes[2].set_xlabel("lon"); axes[2].set_ylabel("lat")
    plt.colorbar(s2, ax=axes[2], shrink=0.6)

    plt.tight_layout()
    plt.savefig(REPORTS / "compare_vs_obu.png", dpi=120)
    print(f"\nГрафик: {REPORTS / 'compare_vs_obu.png'}")

    # разница карт
    fig, ax = plt.subplots(figsize=(15, 7))
    sc = ax.scatter(v["lon"], v["lat"], c=res.values, cmap="RdBu_r", s=5, vmin=-4, vmax=4)
    ax.set_title("Разница: наша модель − Obu 2019 (°C)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(sc, ax=ax, label="разница °C", shrink=0.7)
    plt.tight_layout()
    plt.savefig(REPORTS / "compare_vs_obu_diff.png", dpi=120)
    print(f"Карта разницы: {REPORTS / 'compare_vs_obu_diff.png'}")


if __name__ == "__main__":
    main()