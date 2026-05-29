"""
ШАГ 2: НЕЗАВИСИМАЯ ПРОВЕРКА ТРЕНДА.

Сравнивает НАШ тренд (модель на фичах разных лет) с трендом САМОГО ESA CCI
(из их данных T10m за 2010/2021/2023 напрямую, без нашей модели).

Если тренды совпадают -> наш результат подтверждён эталоном.

Что делает:
  1. Берёт наши точки (из change_clean_data.csv — там lon/lat и наш тренд).
  2. Сэмплирует ESA CCI T10m за 2010, 2021, 2023 в этих точках.
  3. Считает тренд ESA CCI (linfit по 3 годам).
  4. Сравнивает: наш тренд vs тренд ESA CCI (корреляция, средние, карта).

Запуск:
    python3 verify_trend_esacci.py
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

REPORTS = Path("reports")
OUR = REPORTS / "change_clean_data.csv"   # lon,lat,dmagt,dalt,trend,reliable

# Находим GTD файлы по годам
def find_gtd(year):
    cands = [f for f in glob.glob("data/esa_cci/*.nc")
             if "GTD" in f.upper() and str(year) in f]
    return cands[0] if cands else None

YEARS = [2010, 2021, 2023]


def sample_esacci(df, year):
    f = find_gtd(year)
    if f is None:
        print(f"  ESA CCI за {year} не найден!")
        return None
    ds = xr.open_dataset(f)
    t10 = ds["T10m"].isel(time=0)
    la = xr.DataArray(df["lat"].values, dims="p")
    lo = xr.DataArray(df["lon"].values, dims="p")
    vals = t10.sel(lat=la, lon=lo, method="nearest").values
    return vals


def main():
    df = pd.read_csv(OUR)
    print(f"Наших точек: {len(df)}")

    # Сэмплируем ESA CCI за каждый год
    for y in YEARS:
        print(f"Сэмплирую ESA CCI T10m {y}...")
        df[f"esa_{y}"] = sample_esacci(df, y)

    # Оставляем точки, где ESA CCI есть во все годы
    esa_cols = [f"esa_{y}" for y in YEARS]
    v = df.dropna(subset=esa_cols).copy()
    print(f"Точек с ESA CCI во все годы: {len(v)}")

    # Тренд ESA CCI (linfit по 3 годам), °C/год
    yrs = np.array(YEARS, dtype=float)
    v["esa_trend"] = np.polyfit(yrs, v[esa_cols].values.T, 1)[0]
    # Изменение ESA CCI 2010->2023
    v["esa_dmagt"] = v["esa_2023"] - v["esa_2010"]

    # === СРАВНЕНИЕ ===
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ТРЕНДОВ: наш vs ESA CCI")
    print("=" * 60)
    print(f"  Наш тренд (среднее):      {v['trend'].mean()*10:+.3f} °C/декада")
    print(f"  ESA CCI тренд (среднее):  {v['esa_trend'].mean()*10:+.3f} °C/декада")
    print(f"  Наше изменение 13 лет:    {v['dmagt'].mean():+.3f} °C")
    print(f"  ESA CCI изменение 13 лет: {v['esa_dmagt'].mean():+.3f} °C")

    # корреляция трендов
    vv = v.dropna(subset=["trend", "esa_trend"])
    if len(vv) > 10:
        r_tr, _ = pearsonr(vv["trend"], vv["esa_trend"])
        r_dm, _ = pearsonr(vv["dmagt"], vv["esa_dmagt"])
        print(f"\n  Корреляция трендов (наш vs ESA): r={r_tr:.3f}")
        print(f"  Корреляция изменений:           r={r_dm:.3f}")
        # согласованность знака
        same_sign = (np.sign(vv["trend"]) == np.sign(vv["esa_trend"])).mean()
        print(f"  Совпадение знака тренда:        {100*same_sign:.0f}%")

    print("\nИнтерпретация:")
    if vv['esa_trend'].mean() > 0 and v['trend'].mean() > 0:
        print("  Оба показывают ПОТЕПЛЕНИЕ — направление подтверждено эталоном.")
    if len(vv) > 10 and r_tr > 0.3:
        print("  Положительная корреляция — пространственный паттерн согласуется.")
    elif len(vv) > 10:
        print("  Слабая корреляция паттерна — наш тренд шумнее ESA CCI на коротком ряде.")

    # === ГРАФИКИ ===
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # scatter трендов
    ax = axes[0]
    idx = np.random.RandomState(42).choice(len(vv), min(5000, len(vv)), replace=False)
    ax.scatter(vv["esa_trend"].values[idx]*10, vv["trend"].values[idx]*10, alpha=0.3, s=5)
    lim = [-2, 2]
    ax.plot(lim, lim, "r--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Тренд ESA CCI, °C/декада")
    ax.set_ylabel("Наш тренд, °C/декада")
    ax.set_title(f"Согласованность трендов\nr={r_tr:.2f}, знак совпадает {100*same_sign:.0f}%")
    ax.grid(alpha=0.3)

    # карта тренда ESA CCI
    def raster(sub, col, res=0.3, md=0.5):
        lon_g = np.arange(30, 180, res); lat_g = np.arange(55, 78, res)
        LON, LAT = np.meshgrid(lon_g, lat_g)
        ss = sub.dropna(subset=[col])
        Z = griddata((ss["lon"], ss["lat"]), ss[col], (LON, LAT), method="linear")
        tree = cKDTree(np.column_stack([ss["lon"], ss["lat"]]))
        d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
        return LON, LAT, np.where(d.reshape(LON.shape) <= md, Z, np.nan)

    L1, A1, Z1 = raster(vv, "esa_trend")
    im1 = axes[1].pcolormesh(L1, A1, Z1*10, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    axes[1].set_title("Тренд ESA CCI (°C/декада)")
    axes[1].set_aspect(2.0)
    plt.colorbar(im1, ax=axes[1], shrink=0.6)

    L2, A2, Z2 = raster(vv, "trend")
    im2 = axes[2].pcolormesh(L2, A2, Z2*10, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto")
    axes[2].set_title("Наш тренд (°C/декада)")
    axes[2].set_aspect(2.0)
    plt.colorbar(im2, ax=axes[2], shrink=0.6)

    plt.tight_layout()
    plt.savefig(REPORTS / "verify_trend_esacci.png", dpi=130)
    print(f"\nГрафик: {REPORTS / 'verify_trend_esacci.png'}")
    v.to_csv(REPORTS / "verify_trend_data.csv", index=False)


if __name__ == "__main__":
    main()