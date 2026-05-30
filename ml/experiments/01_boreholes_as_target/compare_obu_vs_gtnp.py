"""
Сравнение MAGT из Obu 2019 (GeoTIFF, 1 км) с измерениями GTN-P.

Работает с GeoTIFF в проекции Polar Stereographic (как скачались с PANGAEA).
Точки GTN-P (в lat/lon) автоматически перепроецируются в CRS файла.

Запуск:
    python3 compare_obu_vs_gtnp_tif.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ============================================================
# КОНФИГ
# ============================================================
DATA_DIR = Path("./data")
OBU_TIF  = DATA_DIR / "obu2019" / "UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif"
GTNP_CSV = DATA_DIR / "labels" / "gtnp_boreholes_v2.csv"
REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


def sample_geotiff_at_latlon(tif_path, lats, lons):
    """
    Сэмплирует значения GeoTIFF в точках (lats, lons), заданных в EPSG:4326.
    Автоматически перепроецирует точки в CRS файла.
    """
    with rasterio.open(tif_path) as src:
        print(f"      CRS файла: {src.crs}")
        print(f"      Размер: {src.width} x {src.height}")
        print(f"      NoData: {src.nodata}")
        print(f"      Bounds: {src.bounds}")

        # Перепроецируем точки lat/lon -> CRS файла
        xs, ys = warp_transform(
            "EPSG:4326", src.crs, list(lons), list(lats)
        )
        coords = list(zip(xs, ys))

        sampled = [v[0] for v in src.sample(coords)]
        sampled = np.array(sampled, dtype=float)

        # Заменяем NoData на NaN
        if src.nodata is not None:
            sampled[sampled == src.nodata] = np.nan
        # Очень отрицательные значения тоже считаем NoData
        sampled[sampled < -100] = np.nan

    return sampled


def main():
    print("=" * 60)
    print("Сравнение Obu 2019 (GeoTIFF 1 км) vs GTN-P")
    print("=" * 60)

    if not OBU_TIF.exists():
        print(f"ОШИБКА: нет файла {OBU_TIF}")
        return

    # ---- Загрузка GTN-P ----
    print(f"\n[1/2] Загружаю GTN-P: {GTNP_CSV}")
    gtnp = pd.read_csv(GTNP_CSV)
    print(f"      Скважин: {len(gtnp)}")

    # ---- Сэмплирование Obu ----
    print(f"\n[2/2] Сэмплирую Obu GeoTIFF в точки GTN-P...")
    gtnp["obu_magt"] = sample_geotiff_at_latlon(
        OBU_TIF, gtnp["lat"].values, gtnp["lon"].values
    )
    n_valid = gtnp["obu_magt"].notna().sum()
    print(f"      Obu вернул значения для {n_valid}/{len(gtnp)} точек")

    valid = gtnp.dropna(subset=["magt_c", "obu_magt"]).copy()
    valid["residual"] = valid["obu_magt"] - valid["magt_c"]

    if len(valid) == 0:
        print("\nОШИБКА: ни одна точка не получила значение Obu.")
        print("Вероятно проблема с координатами/проекцией. Покажи мне вывод -- разберёмся.")
        return

    # ---- Метрики ----
    bias = valid["residual"].mean()
    mae = valid["residual"].abs().mean()
    rmse = np.sqrt((valid["residual"] ** 2).mean())
    r_pearson, _ = pearsonr(valid["magt_c"], valid["obu_magt"])
    r2 = r_pearson ** 2

    print("\n" + "=" * 60)
    print("МЕТРИКИ СОГЛАСИЯ Obu 2019 vs GTN-P")
    print("=" * 60)
    print(f"  N точек:    {len(valid)}")
    print(f"  Bias:       {bias:+.2f} С    (Obu - GTN-P)")
    print(f"  MAE:        {mae:.2f} С")
    print(f"  RMSE:       {rmse:.2f} С  <- главная метрика")
    print(f"  R^2:        {r2:.3f}")
    print(f"  Corr:       {r_pearson:.3f}")
    print()
    if rmse < 2.0:
        print("  ВЕРДИКТ: ХОРОШЕЕ согласие. Obu -> weak labels с весом 0.5.")
    elif rmse < 3.0:
        print("  ВЕРДИКТ: УДОВЛЕТВОРИТЕЛЬНОЕ. Obu -> weak labels с весом 0.3.")
    else:
        print("  ВЕРДИКТ: ПЛОХОЕ. Нужно разбираться (эпоха / регион / координаты).")
    print()

    # ---- По регионам ----
    print("По регионам (где >= 3 скважин):")
    for site in valid["site_name"].value_counts().index:
        sub = valid[valid["site_name"] == site]
        if len(sub) < 3:
            continue
        sub_rmse = np.sqrt((sub["residual"] ** 2).mean())
        sub_bias = sub["residual"].mean()
        print(f"  {site:20s} N={len(sub):3d}  bias={sub_bias:+.2f}  RMSE={sub_rmse:.2f}")

    # ---- Анализ временного разрыва ----
    if "year_max" in valid.columns and valid["year_max"].notna().any():
        print("\nЗависимость ошибки от года измерения:")
        v2 = valid.dropna(subset=["year_max"])
        if len(v2) > 5:
            corr_year, _ = pearsonr(v2["year_max"], v2["residual"])
            print(f"  Корреляция (year_max, residual): {corr_year:+.3f}")
            if abs(corr_year) > 0.4:
                print("  -> Заметная зависимость от года. Возможна нужна коррекция эпохи.")
            else:
                print("  -> Слабая зависимость. Временной разрыв не критичен.")

    # ---- Графики ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Scatter
    ax = axes[0]
    ax.scatter(valid["magt_c"], valid["obu_magt"], alpha=0.7, s=60, edgecolor="black")
    lims = [
        min(valid["magt_c"].min(), valid["obu_magt"].min()) - 1,
        max(valid["magt_c"].max(), valid["obu_magt"].max()) + 1,
    ]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 line")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("GTN-P measured MAGT [°C]")
    ax.set_ylabel("Obu 2019 modelled MAGT [°C]")
    ax.set_title(f"Obu vs GTN-P\nN={len(valid)}, RMSE={rmse:.2f}, bias={bias:+.2f}, R²={r2:.3f}")
    ax.axhline(0, color="gray", alpha=0.3, lw=0.5)
    ax.axvline(0, color="gray", alpha=0.3, lw=0.5)
    ax.legend(); ax.grid(alpha=0.3)

    # Гистограмма ошибок
    ax = axes[1]
    ax.hist(valid["residual"], bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="zero")
    ax.axvline(bias, color="orange", lw=2, label=f"bias={bias:+.2f}")
    ax.set_xlabel("Residual (Obu - GTN-P) [°C]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of residuals")
    ax.legend(); ax.grid(alpha=0.3)

    # Residual vs year
    ax = axes[2]
    if "year_max" in valid.columns and valid["year_max"].notna().any():
        v2 = valid.dropna(subset=["year_max"])
        ax.scatter(v2["year_max"], v2["residual"], alpha=0.7, s=60, edgecolor="black", color="green")
        ax.axhline(0, color="red", linestyle="--", lw=1.5)
        ax.set_xlabel("Year of GTN-P measurement")
        ax.set_ylabel("Residual (Obu - GTN-P) [°C]")
        ax.set_title("Error vs measurement year")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Нет данных по годам", ha="center", va="center")

    plt.tight_layout()
    plot_path = REPORTS_DIR / "obu_vs_gtnp.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"\nГрафик сохранён: {plot_path}")

    table_path = REPORTS_DIR / "obu_vs_gtnp_table.csv"
    cols = ["borehole_id", "name", "site_name", "lat", "lon",
            "magt_c", "obu_magt", "residual"]
    if "year_max" in valid.columns:
        cols.append("year_max")
    valid[cols].to_csv(table_path, index=False)
    print(f"Таблица сохранена: {table_path}")
    print("\n[OK] Готово!")


if __name__ == "__main__":
    main()