"""
Растеризация 22 годовых geojson → X-тензор формы (22, 231, 1501, 20).

БЕЗ target! Target пересчитаем отдельно через compute_target_23y.py.

Вход:  data/gee/RussiaGrid_0.1deg_v2_<year>.geojson  (22 файла, 2003–2024)
Выход: data/tensor_01deg_extended_22y.npz с ключами:
       - X (22, 231, 1501, 20)
       - lons (1501,)
       - lats (231,)
       - years (22,)
       - feature_names (20,)

Следующий шаг: scripts/rasterize_2025.py → tensor_01deg_extended_23y.npz
"""

from pathlib import Path
import json
import numpy as np
import time

DATA_DIR = Path("data/gee")
OUT_FILE = Path("data/tensor_01deg_extended_22y.npz")

STEP = 0.1
LON_MIN, LON_MAX = 30.0, 180.0
LAT_MIN, LAT_MAX = 55.0, 78.0
YEARS = list(range(2003, 2025))  # 2003..2024, 22 года

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
    print("=" * 60)
    print(f"РАСТЕРИЗАЦИЯ EXTENDED: {len(YEARS)} лет → X-тензор")
    print("=" * 60)

    lons, lats = make_grid()
    H, W = len(lats), len(lons)
    print(f"\nСетка: H={H} × W={W}")
    print(f"  всего ячеек: {H*W:,}")

    T = len(YEARS)
    n_feat = len(FEATURE_COLS)

    print(f"\nВыделяю X-тензор: ({T}, {H}, {W}, {n_feat})")
    mem_X = T * H * W * n_feat * 4 / 1024 / 1024 / 1024
    print(f"  память X: {mem_X:.2f} ГБ")
    X_all = np.full((T, H, W, n_feat), np.nan, dtype=np.float32)

    print(f"\nРастеризация:")
    for t, year in enumerate(YEARS):
        print(f"[{t+1:>2}/{T}] {year}", end=" ")
        X_year = load_year_to_array(year, lons, lats)
        X_all[t] = X_year
        del X_year

    # Маска валидности
    nan_pct = np.isnan(X_all).mean() * 100
    print(f"\n=== СТАТИСТИКА X ===")
    print(f"X: {X_all.shape}, dtype={X_all.dtype}")
    print(f"  память: {X_all.nbytes / 1024 / 1024 / 1024:.2f} ГБ")
    print(f"  NaN: {nan_pct:.1f}%")

    # Сохраняем
    print(f"\nСохраняю в {OUT_FILE}...")
    np.savez_compressed(
        OUT_FILE,
        X=X_all,
        lons=lons,
        lats=lats,
        years=np.array(YEARS),
        feature_names=np.array(FEATURE_COLS),
    )
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f"Готово: {OUT_FILE} ({size_mb:.0f} МБ)")


if __name__ == "__main__":
    main()
