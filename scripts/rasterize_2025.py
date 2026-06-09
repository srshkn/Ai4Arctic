"""
Растеризация 2025 года + конкатенация в 23-year тензор.

Использует те же функции что rasterize_extended.py, чтобы 2025 был
совместим с существующими 22 годами (одна сетка, тот же порядок фичей).

Вход:
    data/tensor_01deg_extended_22y.npz                 (22 года 2003-2024)
    data/gee/RussiaGrid_0.1deg_v2_2025.geojson         (новый годовой файл)

Выход:
    data/tensor_01deg_extended_23y.npz                 (23 года 2003-2025)
"""

from pathlib import Path
import json
import numpy as np
import time

# ===== Константы (точно как в rasterize_extended.py) =====
DATA_DIR = Path("data/gee")
INPUT_TENSOR = Path("data/tensor_01deg_extended_22y.npz")
OUT_FILE = Path("data/tensor_01deg_extended_23y.npz")

STEP = 0.1
LON_MIN, LON_MAX = 30.0, 180.0
LAT_MIN, LAT_MAX = 55.0, 78.0

NEW_YEAR = 2025

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def make_grid():
    lons = np.arange(LON_MIN, LON_MAX + STEP / 2, STEP)
    lats = np.arange(LAT_MIN, LAT_MAX + STEP / 2, STEP)
    return lons, lats


def lonlat_to_indices(lon, lat, lons, lats):
    j = np.round((lon - lons[0]) / STEP).astype(int)
    i = np.round((lat - lats[0]) / STEP).astype(int)
    return i, j


def load_year_to_array(year, lons, lats):
    fpath = DATA_DIR / f"RussiaGrid_0.1deg_v2_{year}.geojson"
    H, W = len(lats), len(lons)
    n_feat = len(FEATURE_COLS)
    X = np.full((H, W, n_feat), np.nan, dtype=np.float32)
    print(f"  {fpath.name} ({fpath.stat().st_size / 1024 / 1024:.0f} МБ)...", end="", flush=True)

    t0 = time.time()
    with open(fpath) as fh:
        gj = json.load(fh)
    t_parse = time.time() - t0

    t0 = time.time()
    n_ok = 0
    for ft in gj["features"]:
        geom = ft.get("geometry")
        if geom is None:
            continue
        lo, la = geom["coordinates"]
        i, j = lonlat_to_indices(lo, la, lons, lats)
        if 0 <= i < H and 0 <= j < W:
            props = ft["properties"]
            for k, col in enumerate(FEATURE_COLS):
                v = props.get(col)
                if v is not None:
                    X[i, j, k] = v
            n_ok += 1
    t_fill = time.time() - t0
    print(f" parse {t_parse:.1f}с, fill {t_fill:.1f}с, X точек {n_ok:,}")
    del gj
    return X


def main():
    print(f"=== Растеризация 2025 + конкатенация → 23 года ===\n")

    # ===== Шаг 1: загрузка существующего 22-year тензора =====
    print(f"[Шаг 1] Загружаем существующий 22-year тензор...")
    if not INPUT_TENSOR.exists():
        print(f"ОШИБКА: {INPUT_TENSOR} не найден")
        return

    data22 = np.load(INPUT_TENSOR)
    X22 = data22['X']
    years22 = list(data22['years'])
    lons22 = data22['lons']
    lats22 = data22['lats']
    fnames22 = list(data22['feature_names'])

    print(f"  X: {X22.shape}")
    print(f"  Years: {years22[0]}..{years22[-1]} ({len(years22)} лет)")
    print(f"  Grid: lats {lats22.shape} × lons {lons22.shape}")

    # Sanity check
    if NEW_YEAR in years22:
        print(f"\nВНИМАНИЕ: {NEW_YEAR} уже в тензоре! Перезаписать?")
        ans = input("Продолжить? (y/N): ").strip().lower()
        if ans != 'y':
            print("Отмена.")
            return

    # ===== Шаг 2: растеризация 2025 =====
    print(f"\n[Шаг 2] Растеризация {NEW_YEAR}...")
    lons, lats = make_grid()

    # Проверим что grid совпадает с существующим
    if lons.shape != lons22.shape or lats.shape != lats22.shape:
        print(f"ОШИБКА: размеры grid не совпадают!")
        print(f"  Existing: lats {lats22.shape} × lons {lons22.shape}")
        print(f"  New:      lats {lats.shape} × lons {lons.shape}")
        return

    if not np.allclose(lons, lons22) or not np.allclose(lats, lats22):
        print(f"ОШИБКА: координаты grid не совпадают!")
        print(f"  lats max diff: {np.abs(lats - lats22).max()}")
        print(f"  lons max diff: {np.abs(lons - lons22).max()}")
        return
    print(f"  Grid совпадает: lats[0..-1] = {lats[0]:.1f}..{lats[-1]:.1f}, "
          f"lons[0..-1] = {lons[0]:.1f}..{lons[-1]:.1f}")

    X_new = load_year_to_array(NEW_YEAR, lons, lats)  # (H, W, 20)

    # Sanity check на наличие данных
    valid_per_feat = (~np.isnan(X_new)).sum(axis=(0, 1))
    print(f"\n  Заполненность по фичам:")
    for k, col in enumerate(FEATURE_COLS):
        pct = 100 * valid_per_feat[k] / (X_new.shape[0] * X_new.shape[1])
        warn = " ⚠️" if pct < 30 else ""
        print(f"    {col:14s}: {valid_per_feat[k]:>7,} ({pct:.1f}%){warn}")

    # ===== Шаг 3: конкатенация =====
    print(f"\n[Шаг 3] Конкатенация...")
    # X_new имеет форму (H, W, F) → добавляем ось time → (1, H, W, F)
    X_new_t = X_new[np.newaxis, :, :, :]  # (1, 231, 1501, 20)

    X23 = np.concatenate([X22, X_new_t], axis=0)  # (23, 231, 1501, 20)
    years23 = years22 + [NEW_YEAR]

    print(f"  X23 shape: {X23.shape}")
    print(f"  Years: {years23[0]}..{years23[-1]} ({len(years23)} лет)")

    # ===== Шаг 4: сохранение =====
    print(f"\n[Шаг 4] Сохранение...")
    np.savez_compressed(
        OUT_FILE,
        X=X23.astype(np.float32),
        years=np.array(years23),
        lats=lats22,
        lons=lons22,
        feature_names=np.array(fnames22),
        description='23-year tensor: 22 historical years (2003-2024) + real 2025 from GEE',
    )

    out_size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"  Сохранено: {OUT_FILE.name} ({out_size_mb:.0f} МБ)")

    # ===== Sanity check =====
    print(f"\n[Sanity check] Сравнение 2024 (real) vs 2025 (new):")
    for k, col in enumerate(FEATURE_COLS[:9]):  # только 9 динамичных
        m24 = float(np.nanmedian(X23[-2, :, :, k]))
        m25 = float(np.nanmedian(X23[-1, :, :, k]))
        delta = m25 - m24
        sign = "+" if delta >= 0 else ""
        print(f"  {col:14s}: 2024 median {m24:+.3f}, 2025 median {m25:+.3f}, Δ {sign}{delta:.3f}")


if __name__ == '__main__':
    main()
