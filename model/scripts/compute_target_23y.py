"""
Этап 4: пересчёт TTOP target для 23 лет (2003-2025).

Использует статичный landcover из baseline v2 и FDD/TDD из extended-тензора.

Вход:
    data/tensor_01deg_extended_23y.npz
    data/y_new_rk_landcover.npz          (landcover + rk_map)

Опционально (sanity check):
    data/y_new_rk_landcover_extended.npz (21-летний target, если есть)

Выход:
    data/y_new_rk_landcover_extended_23y.npz

Запуск:
    python3 scripts/compute_target_23y.py
"""

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.landcover import compute_ttop_target

DATA_DIR = BASE_DIR / 'data'
TENSOR_PATH = DATA_DIR / 'tensor_01deg_extended_23y.npz'
LANDCOVER_PATH = DATA_DIR / 'y_new_rk_landcover.npz'
PREV_TARGET_PATH = DATA_DIR / 'y_new_rk_landcover_extended.npz'
OUTPUT = DATA_DIR / 'y_new_rk_landcover_extended_23y.npz'


def sanity_check_vs_baseline(y23, years23, y_old_arr):
    """Сверка 2010-2023 с baseline target (14 лет)."""
    print("\nSanity check vs y_new_rk_landcover.npz (2010-2023):")
    max_diff = 0.0
    for old_t, year in enumerate(range(2010, 2024)):
        if year not in years23:
            continue
        new_t = years23.index(year)
        diff = y23[new_t] - y_old_arr[old_t]
        diff_valid = diff[~np.isnan(diff)]
        if len(diff_valid) > 0:
            max_diff = max(max_diff, float(np.abs(diff_valid).max()))
    print(f"  Max abs diff: {max_diff:.6f}°C")
    if max_diff < 0.001:
        print("  OK: target совпадает с baseline")
    else:
        print("  WARN: расхождение больше ожидаемого")


def sanity_check_vs_prev(y23, years23):
    """Сверка с предыдущим extended target (21 год), если файл есть."""
    if not PREV_TARGET_PATH.exists():
        print("\nSanity check vs y_new_rk_landcover_extended.npz: пропущен (файл не найден)")
        return

    prev = np.load(PREV_TARGET_PATH)
    y_prev = prev['y_new']
    years_prev = list(prev['years'])
    max_diff = 0.0
    for year in years_prev:
        if year not in years23:
            continue
        i_prev = years_prev.index(year)
        i_new = years23.index(year)
        mask = ~np.isnan(y_prev[i_prev]) & ~np.isnan(y23[i_new])
        if mask.any():
            diff = np.abs(y_prev[i_prev][mask] - y23[i_new][mask]).max()
            max_diff = max(max_diff, diff)

    print("\nSanity check vs y_new_rk_landcover_extended.npz:")
    print(f"  Max abs diff на общих годах: {max_diff:.6f}°C")
    if max_diff > 0.01:
        print("  WARN: расхождение значительное")
    else:
        print("  OK: target идентичен предыдущему")


def main():
    print("Загрузка 23-year тензора...")
    if not TENSOR_PATH.exists():
        raise FileNotFoundError(f"Не найден: {TENSOR_PATH}")
    tensor = np.load(TENSOR_PATH)
    X23 = tensor['X']
    years23 = list(tensor['years'])
    lats = tensor['lats']
    lons = tensor['lons']
    print(f"  X: {X23.shape}")
    print(f"  Years: {years23[0]}..{years23[-1]} ({len(years23)} лет)")

    print("\nЗагрузка landcover...")
    if not LANDCOVER_PATH.exists():
        raise FileNotFoundError(
            f"Не найден: {LANDCOVER_PATH}\n"
            "Создаётся в v2 baseline (notebooks/01 или notebooks/03)."
        )
    y_old_npz = np.load(LANDCOVER_PATH)
    landcover = y_old_npz['landcover']
    rk_map = y_old_npz['rk_map']
    print(f"  landcover: {landcover.shape}")

    print("\nПересчёт TTOP target...")
    y23 = compute_ttop_target(X23, landcover, fdd_idx=7, tdd_idx=8)
    print(f"  y23: {y23.shape}")

    print("\nМедиана TTOP по годам (2020+):")
    for i, year in enumerate(years23):
        if year >= 2020:
            median = float(np.nanmedian(y23[i]))
            marker = " ← NEW" if year == 2025 else ""
            print(f"  {year}: {median:+.2f}°C{marker}")

    if 'y_new' in y_old_npz:
        sanity_check_vs_baseline(y23, years23, y_old_npz['y_new'])
    sanity_check_vs_prev(y23, years23)

    mask = ~np.isnan(y23).all(axis=0)
    print(f"\nМаска валидности: {mask.sum():,} ячеек ({100 * mask.mean():.1f}% сетки)")

    np.savez_compressed(
        OUTPUT,
        y_new=y23.astype(np.float32),
        landcover=landcover,
        rk_map=rk_map,
        years=np.array(years23),
        lats=lats,
        lons=lons,
    )
    print(f"\nСохранено: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} МБ)")


if __name__ == '__main__':
    main()
