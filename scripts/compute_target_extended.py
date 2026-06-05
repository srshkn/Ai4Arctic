"""
P2: пересчёт TTOP target с landcover-varying rk для расширенного
тензора 2003-2023.

Использует src.landcover.compute_ttop_target — ту же формулу что
для основной модели. Применяет существующий landcover (статичный, из
maps_lh_v3.npz) к FDD/TDD каждого года.

Sanity check: для 2010-2023 результат должен МАТЕМАТИЧЕСКИ совпадать
со старым y_new_rk_landcover.npz (max abs diff < 0.001°C).

Вход:
  data/tensor_01deg_extended.npz
  data/y_new_rk_landcover.npz (для landcover и sanity check)

Выход:
  data/y_new_rk_landcover_extended.npz
"""

import numpy as np
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.landcover import compute_ttop_target, RK_TABLE

DATA_DIR = BASE_DIR / 'data'
OUTPUT = DATA_DIR / 'y_new_rk_landcover_extended.npz'


def main():
    # X-тензор
    print("Загружаем X-тензор (21 год)...")
    tensor_ext = np.load(DATA_DIR / 'tensor_01deg_extended.npz')
    X_ext = tensor_ext['X']
    years_ext = list(tensor_ext['years'])
    lats = tensor_ext['lats']
    lons = tensor_ext['lons']
    print(f"  X: {X_ext.shape}, years: {years_ext[0]}..{years_ext[-1]}")

    # Landcover
    print("Загружаем landcover...")
    y_old_npz = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old_npz['landcover']
    print(f"  landcover: {landcover.shape}, classes {np.unique(landcover)}")
    print(f"  RK_TABLE: {RK_TABLE}")

    # Пересчёт target
    print(f"\nПересчёт TTOP target для 21 года...")
    y_ext = compute_ttop_target(X_ext, landcover, fdd_idx=7, tdd_idx=8)
    print(f"  y_ext: {y_ext.shape}")

    # Тренд по годам
    print(f"\nМедиана TTOP по годам (warming signal):")
    for t, year in enumerate(years_ext):
        print(f"  {year}: {np.nanmedian(y_ext[t]):+.2f}°C")

    # Sanity check на общих годах 2010-2023
    y_old_arr = y_old_npz['y_new']
    print(f"\nSanity check: новый target vs старый для 2010-2023:")
    max_global_diff = 0.0
    for old_t, year in enumerate(range(2010, 2024)):
        new_t = years_ext.index(year)
        diff = y_ext[new_t] - y_old_arr[old_t]
        diff_valid = diff[~np.isnan(diff)]
        if len(diff_valid) > 0:
            max_diff = float(np.abs(diff_valid).max())
            max_global_diff = max(max_global_diff, max_diff)
    print(f"  Max abs diff across all 14 common years: {max_global_diff:.6f}°C")
    if max_global_diff < 0.001:
        print(f"  OK: target идентичен старому в пределах numerical precision")
    else:
        print(f"  WARN: расхождение больше ожидаемого — проверь pipeline")

    # Маска валидности
    mask = ~np.isnan(y_ext).all(axis=0)
    print(f"\nМаска валидности: {mask.sum():,} ячеек ({100*mask.mean():.1f}% сетки)")

    # Сохраняем
    np.savez_compressed(
        OUTPUT,
        y_new=y_ext.astype(np.float32),
        landcover=landcover,
        rk_map=y_old_npz['rk_map'],
        years=np.array(years_ext),
        lats=lats,
        lons=lons,
    )
    print(f"\nСохранено: {OUTPUT}")
    print(f"  размер: {OUTPUT.stat().st_size/1e6:.1f} МБ")


if __name__ == '__main__':
    main()
