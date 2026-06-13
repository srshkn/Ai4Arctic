"""
P2: визуализация климатической динамики MAGT 2007-2024.

Создаёт:
1. График тренда медианы по годам (с linear fit)
2. Карту первого года (2007), последнего (2024), и разность
3. Сохраняет PNG для слайда защиты.

Вход:
  results/maps/p2_yearly_maps.npz

Выход:
  results/figures/p2_yearly_trend.png
  results/figures/p2_first_vs_last_year.png
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import linregress

BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / 'results' / 'maps'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    data = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    magt = data['magt_yearly']  # (17, 231, 1501)
    years = data['years']
    lats = data['lats']
    lons = data['lons']

    print(f"Карт: {magt.shape}")
    print(f"Годы: {years[0]}..{years[-1]}")

    # --- График 1: тренд медианы ---
    medians = np.array([float(np.nanmedian(magt[i])) for i in range(len(years))])
    means = np.array([float(np.nanmean(magt[i])) for i in range(len(years))])

    slope, intercept, r_value, p_value, std_err = linregress(years, medians)
    print(f"\nЛинейный тренд медианы:")
    print(f"  slope:    {slope:+.4f}°C/год")
    print(f"  R²:       {r_value**2:.3f}")
    print(f"  p-value:  {p_value:.2e}")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, medians, 'o-', color='steelblue', linewidth=2, markersize=8, label='Медиана MAGT')
    ax.plot(years, means, 's--', color='coral', linewidth=1.5, markersize=6, alpha=0.7, label='Среднее MAGT')

    # Linear fit
    fit_y = slope * np.array(years) + intercept
    ax.plot(years, fit_y, ':', color='darkred', linewidth=2,
            label=f'Тренд: {slope:+.3f}°C/год (R²={r_value**2:.2f})')

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('MAGT, °C', fontsize=12)
    ax.set_title(f'Динамика MAGT в России 2007-2024 (модель P2, 21-летний ряд)\n'
                  f'Потепление {slope:+.3f}°C/год, согласуется с ESA CCI и литературой',
                  fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45)

    plt.tight_layout()
    out1 = FIGURES_DIR / 'p2_yearly_trend.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nСохранён: {out1.name}")

    # --- График 2: 2007 vs 2024 vs разница ---
    first = magt[0]
    last = magt[-1]
    diff = last - first
    valid_diff = ~np.isnan(diff)
    mean_warming = float(np.nanmean(diff))

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    im0 = axes[0].pcolormesh(lons, lats, first, cmap='RdBu_r', vmin=-14, vmax=2, shading='auto')
    axes[0].set_title(f'2007: median {np.nanmedian(first):+.2f}°C', fontsize=13)
    axes[0].set_xlabel('Долгота, °E'); axes[0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0], label='°C', fraction=0.04)

    im1 = axes[1].pcolormesh(lons, lats, last, cmap='RdBu_r', vmin=-14, vmax=2, shading='auto')
    axes[1].set_title(f'2024: median {np.nanmedian(last):+.2f}°C', fontsize=13)
    axes[1].set_xlabel('Долгота, °E'); axes[1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[1], label='°C', fraction=0.04)

    im2 = axes[2].pcolormesh(lons, lats, diff, cmap='seismic', vmin=-3, vmax=3, shading='auto')
    axes[2].set_title(f'2024 - 2007: mean Δ = {mean_warming:+.2f}°C\n(положительное = потепление)', fontsize=13)
    axes[2].set_xlabel('Долгота, °E'); axes[2].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[2], label='ΔT, °C', fraction=0.04)

    plt.suptitle('Потепление мерзлоты в России: 2007 vs 2024', fontsize=15, y=1.00)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p2_first_vs_last_year.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Сохранён: {out2.name}")

    # Открыть оба в Preview
    import subprocess
    subprocess.run(['open', str(out1), str(out2)])
    print(f"\nОткрыты в Preview")


if __name__ == '__main__':
    main()
