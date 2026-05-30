"""
ЭТАП 1.А — TTOP-РАЗМЕТКА.

Считаем target по физической формуле, не эмулируем ESA CCI.

Формула Smith & Riseborough 2002 (та же, что у Obu 2019):

    TTOP = (rk · TDD − FDD) / P

где:
  rk  — отношение теплопроводностей талой и мёрзлой почвы (0.7, стандарт)
  TDD — сумма градусо-дней оттепели (есть в нашей сетке)
  FDD — сумма градусо-дней мороза (есть в нашей сетке)
  P   — длина года в днях (365)

Смысл: баланс тепла и холода за год определяет температуру на верхней
границе мерзлоты. TTOP > 0 → нет мерзлоты; TTOP < 0 → мерзлота.

Запуск:
    python3 compute_ttop.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

# Параметры формулы
RK = 0.7        # отношение теплопроводностей (стандарт)
P_YEAR = 365    # длина года в днях

FEAT_DIR = Path("ml/data/features")
REPORTS = Path("ml/reports")
REPORTS.mkdir(exist_ok=True)
YEARS = [2010, 2013, 2015, 2017, 2019, 2021, 2023]


def load_year(year):
    """Загружает сетку признаков за год."""
    fpath = FEAT_DIR / f"RussiaGrid_0.25deg_{year}.geojson"
    with open(fpath) as fh:
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
    df["year"] = year
    return df


def compute_ttop(df):
    """TTOP по формуле Smith & Riseborough."""
    fdd = df["FDD"].values
    tdd = df["TDD"].values
    ttop = (RK * tdd - fdd) / P_YEAR
    return ttop


def load_border():
    """Контур России для маски карты."""
    fpath = Path("data/russia_border.geojson")
    if not fpath.exists():
        return None
    with open(fpath) as f:
        gj = json.load(f)
    polys = []
    for ft in gj["features"]:
        geom = ft["geometry"]
        if geom["type"] == "Polygon":
            polys.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            for p in geom["coordinates"]:
                polys.append(p[0])
    return polys


def make_mask(LON, LAT, polys):
    """Маска: точки внутри полигонов России."""
    pts = np.column_stack([LON.ravel(), LAT.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        path = MplPath(np.array(poly))
        inside |= path.contains_points(pts)
    return inside.reshape(LON.shape)


def main():
    print("=" * 60)
    print("ЭТАП 1.А: TTOP-разметка")
    print(f"Параметры формулы: rk={RK}, P={P_YEAR} дней")
    print("=" * 60)

    all_years = []
    for year in YEARS:
        try:
            df = load_year(year)
        except FileNotFoundError:
            print(f"  ⚠ {year}: файл не найден, пропуск")
            continue

        df["TTOP"] = compute_ttop(df)
        df_clean = df.dropna(subset=["TTOP", "FDD", "TDD"])

        # статистика
        ttop = df_clean["TTOP"].values
        n_permafrost = (ttop < 0).sum()
        pct = 100 * n_permafrost / len(ttop)

        print(f"\n  {year}: {len(df_clean)} точек")
        print(f"    среднее TTOP: {ttop.mean():+.2f}°C")
        print(f"    диапазон:     [{ttop.min():+.1f}, {ttop.max():+.1f}] °C")
        print(f"    мерзлота (TTOP<0): {n_permafrost} точек ({pct:.1f}%)")

        all_years.append(df_clean[["lon", "lat", "year", "FDD", "TDD", "TTOP"]])

    # Сводный CSV
    combined = pd.concat(all_years, ignore_index=True)
    out_csv = REPORTS / "ttop_data.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\n  Сохранено: {out_csv}")
    print(f"  Всего строк (год×точка): {len(combined)}")

    # === Карта TTOP_2021 для проверки концепта ===
    df_2021 = combined[combined["year"] == 2021]
    if len(df_2021) == 0:
        print("  ⚠ нет 2021 для карты")
        return

    polys = load_border()
    res = 0.25
    lon_g = np.arange(28, 182, res)
    lat_g = np.arange(53, 79, res)
    LON, LAT = np.meshgrid(lon_g, lat_g)

    Z = griddata(
        (df_2021["lon"], df_2021["lat"]), df_2021["TTOP"],
        (LON, LAT), method="linear"
    )

    # обрезка по расстоянию (чтобы не было артефактов далеко от точек)
    tree = cKDTree(np.column_stack([df_2021["lon"], df_2021["lat"]]))
    d, _ = tree.query(np.column_stack([LON.ravel(), LAT.ravel()]), k=1)
    Z = np.where(d.reshape(LON.shape) <= 0.5, Z, np.nan)

    # маска по контуру России
    if polys is not None:
        m = make_mask(LON, LAT, polys)
        Z = np.where(m, Z, np.nan)

    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_facecolor("#f7f9fb")
    im = ax.pcolormesh(LON, LAT, Z, cmap="RdBu_r", vmin=-15, vmax=5, shading="auto")

    if polys is not None:
        for poly in polys:
            arr = np.array(poly)
            ax.plot(arr[:, 0], arr[:, 1], color="#333", lw=0.6, alpha=0.7)

    ax.set_xlim(28, 182); ax.set_ylim(53, 79)
    ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
    ax.set_title(
        "TTOP-разметка 2021 (формула Smith & Riseborough)\n"
        f"target вычислен по физической формуле, без эмуляции ESA CCI · rk={RK}"
    )
    ax.set_aspect(1.9)
    ax.grid(alpha=0.2, linestyle=":")
    plt.colorbar(im, ax=ax, label="TTOP, °C", shrink=0.65, pad=0.02)
    plt.tight_layout()

    out_png = REPORTS / "ttop_map_2021.png"
    plt.savefig(out_png, dpi=160, facecolor="white")
    plt.close()
    print(f"  Карта: {out_png}")

    # Сравнение средних TTOP по годам — тренд
    print("\n=== ТРЕНД средних TTOP по годам ===")
    yearly = combined.groupby("year")["TTOP"].mean()
    print(yearly.to_string())
    if len(yearly) >= 3:
        slope = np.polyfit(yearly.index.values, yearly.values, 1)[0]
        print(f"\nЛинейный тренд: {slope*10:+.3f}°C/декада")
        print("(положительный = потепление)")

    print("\n" + "=" * 60)
    print("ЭТАП 1.А ЗАВЕРШЁН")
    print("=" * 60)
    print("\nЧто проверить вручную:")
    print("  1. Карта ttop_map_2021.png — географически осмысленна?")
    print("     - юг тёплый (TTOP > 0), север/Якутия холодные (TTOP < −5)")
    print("     - горы холоднее окружения")
    print("  2. Доля мерзлоты ~50% (как у нас в основной модели)")
    print("  3. Тренд положительный (потепление)")
    print("\nЕсли всё ок — день 2: обучаем XGBoost на этом target.")


if __name__ == "__main__":
    main()