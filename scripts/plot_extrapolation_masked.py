"""
P2-projection: маскированная карта 2030 для зон с R² > 0.5.

Показывает экстраполяцию только там где исторический тренд статистически
сильный. На зонах со слабым трендом (R² < 0.5) — серая маска "недостаточная
уверенность".

Также пересчитывает статистики только на маскированной зоне для честных
числовых выводов.

Вход:
  results/maps/p2_extrapolation_2025_2030.npz
  results/maps/p2_yearly_maps.npz

Выход:
  results/figures/p2_map_2030_masked.png
  results/metrics/p2_projection_masked_summary.json
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import json
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / 'results' / 'maps'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'

R2_THRESHOLD = 0.5


def main():
    # Загружаем данные
    print("Загружаем данные...")
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    projection = np.load(MAPS_DIR / 'p2_extrapolation_2025_2030.npz')

    magt_hist = history['magt_yearly']  # (18, H, W)
    years_hist = history['years']
    lats = history['lats']
    lons = history['lons']

    magt_proj = projection['magt_extrapolated']  # (6, H, W) для 2025-2030
    slope_map = projection['slope_map']
    r2_map = projection['r2_map']
    years_proj = projection['years_extrap']

    magt_2024 = magt_hist[-1]  # последний реальный
    magt_2030 = magt_proj[-1]  # экстраполяция

    # Маска уверенности
    print(f"\nПрименяем маску R² > {R2_THRESHOLD}...")
    confident_mask = (r2_map > R2_THRESHOLD) & ~np.isnan(r2_map)
    n_total_valid = (~np.isnan(r2_map)).sum()
    n_confident = confident_mask.sum()
    pct_confident = 100 * n_confident / n_total_valid

    print(f"  Всего валидных пикселей: {n_total_valid:,}")
    print(f"  Пикселей с R² > {R2_THRESHOLD}: {n_confident:,} ({pct_confident:.1f}%)")

    # Применяем маску к проекции 2030
    magt_2030_masked = np.where(confident_mask, magt_2030, np.nan)
    change_2024_2030 = magt_2030 - magt_2024
    change_masked = np.where(confident_mask, change_2024_2030, np.nan)

    # Статистики на маскированной зоне
    valid_proj = magt_2030_masked[~np.isnan(magt_2030_masked)]
    valid_change = change_masked[~np.isnan(change_masked)]
    valid_slope_masked = slope_map[confident_mask]

    print(f"\n=== Статистика на confident зоне (R² > {R2_THRESHOLD}) ===")
    print(f"  Median MAGT 2030:        {np.median(valid_proj):+.2f}°C")
    print(f"  Mean MAGT 2030:          {np.mean(valid_proj):+.2f}°C")
    print(f"  Median ΔT (2030-2024):   {np.median(valid_change):+.2f}°C")
    print(f"  Mean ΔT (2030-2024):     {np.mean(valid_change):+.2f}°C")
    print(f"  Median slope:            {np.median(valid_slope_masked):+.4f}°C/год")

    # Сравним с unmasked
    valid_unmasked = magt_2030[~np.isnan(magt_2030)]
    valid_change_unmasked = change_2024_2030[~np.isnan(change_2024_2030)]
    print(f"\n=== Сравнение с unmasked (вся валидная зона) ===")
    print(f"  Median MAGT 2030:  unmasked {np.median(valid_unmasked):+.2f}°C  vs  confident {np.median(valid_proj):+.2f}°C")
    print(f"  Median ΔT:         unmasked {np.median(valid_change_unmasked):+.2f}°C  vs  confident {np.median(valid_change):+.2f}°C")

    # ===== Карта =====
    print("\nРисуем карту...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. MAGT 2030 unmasked (для сравнения)
    im0 = axes[0, 0].pcolormesh(lons, lats, magt_2030,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 0].set_title(f'MAGT 2030 — вся валидная зона\n'
                          f'median {float(np.nanmedian(magt_2030)):+.2f}°C',
                          fontsize=12)
    axes[0, 0].set_xlabel('Долгота, °E')
    axes[0, 0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0, 0], label='MAGT, °C', fraction=0.04)

    # 2. MAGT 2030 masked (R² > 0.5)
    cmap_with_gray = plt.cm.RdBu_r.copy()
    cmap_with_gray.set_bad('lightgray', alpha=0.7)
    im1 = axes[0, 1].pcolormesh(lons, lats, magt_2030_masked,
                                  cmap=cmap_with_gray, vmin=-12, vmax=2, shading='auto')
    axes[0, 1].set_title(f'MAGT 2030 — только зоны с R² > {R2_THRESHOLD}\n'
                          f'median {float(np.median(valid_proj)):+.2f}°C ({pct_confident:.1f}% территории)',
                          fontsize=12)
    axes[0, 1].set_xlabel('Долгота, °E')
    axes[0, 1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[0, 1], label='MAGT, °C', fraction=0.04)

    # 3. Изменение 2030 - 2024 masked
    cmap_reds_gray = plt.cm.Reds.copy()
    cmap_reds_gray.set_bad('lightgray', alpha=0.7)
    im2 = axes[1, 0].pcolormesh(lons, lats, change_masked,
                                  cmap=cmap_reds_gray, vmin=0, vmax=1.0, shading='auto')
    axes[1, 0].set_title(f'Потепление 2024→2030 (только R² > {R2_THRESHOLD})\n'
                          f'mean ΔT = {float(np.mean(valid_change)):+.2f}°C',
                          fontsize=12)
    axes[1, 0].set_xlabel('Долгота, °E')
    axes[1, 0].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[1, 0], label='ΔT, °C', fraction=0.04)

    # 4. R² map с пометкой порога
    cmap_viridis_gray = plt.cm.viridis.copy()
    im3 = axes[1, 1].pcolormesh(lons, lats, r2_map,
                                  cmap=cmap_viridis_gray, vmin=0, vmax=1, shading='auto')
    axes[1, 1].contour(lons, lats, r2_map, levels=[R2_THRESHOLD],
                        colors='white', linewidths=2, alpha=0.8)
    axes[1, 1].set_title(f'R² линейной регрессии (белый контур = R² = {R2_THRESHOLD})\n'
                          f'median {float(np.nanmedian(r2_map)):.3f}, '
                          f'{pct_confident:.1f}% территории удовлетворяет порогу',
                          fontsize=12)
    axes[1, 1].set_xlabel('Долгота, °E')
    axes[1, 1].set_ylabel('Широта, °N')
    plt.colorbar(im3, ax=axes[1, 1], label='R²', fraction=0.04)

    plt.suptitle(f'Маскированная проекция MAGT 2030 — только зоны со статистически сильным трендом (R² > {R2_THRESHOLD})',
                  fontsize=15, y=1.00)
    plt.tight_layout()
    out_fig = FIGURES_DIR / 'p2_map_2030_masked.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out_fig.name}")

    # ===== JSON сводка =====
    summary = {
        'r2_threshold': R2_THRESHOLD,
        'territory': {
            'total_valid_pixels': int(n_total_valid),
            'confident_pixels': int(n_confident),
            'pct_confident': float(pct_confident),
        },
        'magt_2030': {
            'unmasked_median': float(np.nanmedian(magt_2030)),
            'confident_median': float(np.median(valid_proj)),
            'confident_mean': float(np.mean(valid_proj)),
        },
        'change_2024_2030': {
            'unmasked_median': float(np.nanmedian(change_2024_2030)),
            'confident_median': float(np.median(valid_change)),
            'confident_mean': float(np.mean(valid_change)),
            'unit': '°C',
        },
        'slope_per_pixel': {
            'confident_median': float(np.median(valid_slope_masked)),
            'unit': '°C/year',
        },
        'methodology': (
            'Pixel-wise linear regression of MAGT 2007-2024 (18 years). '
            f'Extrapolation to 2030. Mask R² > {R2_THRESHOLD} excludes pixels '
            'where the linear trend is statistically weak (insufficient signal '
            'over noise on the per-pixel level). NOT a ConvLSTM forecast.'
        ),
    }

    out_json = METRICS_DIR / 'p2_projection_masked_summary.json'
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Сохранено: {out_json.name}")

    # Открыть карту
    import subprocess
    subprocess.run(['open', str(out_fig)])
    print(f"\nОткрыто в Preview")


if __name__ == '__main__':
    main()
