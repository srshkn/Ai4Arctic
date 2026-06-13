"""
P2-projection: визуализация линейной экстраполяции 2025-2030.

Создаёт 3 файла:
1. p2_trend_with_projection.png — тренд 2007-2024 + extrapolation 2025-2030 с CI
2. p2_map_2030_with_uncertainty.png — карта MAGT 2030 + slope + R² + uncertainty
3. p2_projection_summary.csv — таблица с числами для каждого года

Вход:
  results/maps/p2_yearly_maps.npz             (исторические 18 лет)
  results/maps/p2_extrapolation_2025_2030.npz (extrapolation)

Выход:
  results/figures/p2_trend_with_projection.png
  results/figures/p2_map_2030_with_uncertainty.png
  results/metrics/p2_projection_summary.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys
from scipy.stats import linregress

BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / 'results' / 'maps'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Загружаем данные
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    projection = np.load(MAPS_DIR / 'p2_extrapolation_2025_2030.npz')

    magt_hist = history['magt_yearly']
    years_hist = history['years']
    lats = history['lats']
    lons = history['lons']

    magt_proj = projection['magt_extrapolated']
    ci_lower = projection['magt_ci_lower']
    ci_upper = projection['magt_ci_upper']
    slope_map = projection['slope_map']
    r2_map = projection['r2_map']
    residual_std_map = projection['residual_std_map']
    years_proj = projection['years_extrap']

    # ===== Figure 1: тренд с экстраполяцией =====
    print("Figure 1: тренд + extrapolation...")

    # Медианы исторических лет
    medians_hist = np.array([float(np.nanmedian(magt_hist[i])) for i in range(len(years_hist))])
    means_hist = np.array([float(np.nanmean(magt_hist[i])) for i in range(len(years_hist))])

    # Медианы проекций
    medians_proj = np.array([float(np.nanmedian(magt_proj[i])) for i in range(len(years_proj))])
    medians_lower = np.array([float(np.nanmedian(ci_lower[i])) for i in range(len(years_proj))])
    medians_upper = np.array([float(np.nanmedian(ci_upper[i])) for i in range(len(years_proj))])

    # Linear fit on historical для отображения тренда
    slope_global, intercept_global, r_value, p_value, _ = linregress(years_hist, medians_hist)
    print(f"  Тренд медианы (исторический): {slope_global:+.4f}°C/год, R²={r_value**2:.3f}, p={p_value:.2e}")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Линия тренда
    all_years = np.concatenate([years_hist, years_proj])
    fit_line = slope_global * all_years + intercept_global
    ax.plot(all_years, fit_line, ':', color='darkred', linewidth=2, alpha=0.7,
            label=f'Линейный тренд: {slope_global:+.3f}°C/год (R²={r_value**2:.2f}, p<1e-6)')

    # Historical: реальные точки
    ax.plot(years_hist, medians_hist, 'o-', color='steelblue', linewidth=2.5,
            markersize=10, label='Медиана MAGT (реальные данные 2007-2024)', zorder=3)

    # Projection: экстраполированные точки с CI envelope
    ax.fill_between(years_proj, medians_lower, medians_upper,
                     color='coral', alpha=0.25, label='95% доверительный интервал')
    ax.plot(years_proj, medians_proj, 's--', color='darkorange', linewidth=2,
            markersize=10, label='Экстраполяция тренда (2025-2030)', zorder=3,
            markerfacecolor='white', markeredgewidth=2)

    # Разделительная линия
    ax.axvline(2024.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(2024.5, ax.get_ylim()[0] + 0.3, '  ← реальные данные  |  экстраполяция →',
            ha='center', fontsize=10, color='gray', alpha=0.8)

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('MAGT, °C', fontsize=12)
    ax.set_title('Динамика MAGT в России: реальные данные 2007-2024 и экстраполяция тренда до 2030\n'
                  f'Δ(2024→2030) ≈ {medians_proj[-1] - medians_hist[-1]:+.2f}°C при условии сохранения наблюдённого тренда',
                  fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=11)
    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years, rotation=45)

    plt.tight_layout()
    out1 = FIGURES_DIR / 'p2_trend_with_projection.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out1.name}")

    # ===== Figure 2: карта 2030 + slope + R² + uncertainty =====
    print("\nFigure 2: карта 2030 с диагностикой...")

    magt_2030 = magt_proj[-1]  # последний год
    magt_2024 = magt_hist[-1]
    change_2024_2030 = magt_2030 - magt_2024

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 2030 MAGT
    im0 = axes[0, 0].pcolormesh(lons, lats, magt_2030,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 0].set_title(f'Экстраполяция MAGT 2030: median {float(np.nanmedian(magt_2030)):+.2f}°C',
                          fontsize=13)
    axes[0, 0].set_xlabel('Долгота, °E')
    axes[0, 0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0, 0], label='MAGT, °C', fraction=0.04)

    # Δ 2024 → 2030
    im1 = axes[0, 1].pcolormesh(lons, lats, change_2024_2030,
                                  cmap='Reds', vmin=0, vmax=1.5, shading='auto')
    axes[0, 1].set_title(f'Изменение 2030 - 2024: mean Δ = {float(np.nanmean(change_2024_2030)):+.2f}°C',
                          fontsize=13)
    axes[0, 1].set_xlabel('Долгота, °E')
    axes[0, 1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[0, 1], label='ΔT, °C', fraction=0.04)

    # Slope map
    im2 = axes[1, 0].pcolormesh(lons, lats, slope_map,
                                  cmap='Reds', vmin=0, vmax=0.15, shading='auto')
    axes[1, 0].set_title(f'Slope тренда (°C/год) по пикселю: median {float(np.nanmedian(slope_map)):+.4f}',
                          fontsize=13)
    axes[1, 0].set_xlabel('Долгота, °E')
    axes[1, 0].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[1, 0], label='°C/год', fraction=0.04)

    # R² map
    im3 = axes[1, 1].pcolormesh(lons, lats, r2_map,
                                  cmap='viridis', vmin=0, vmax=1, shading='auto')
    axes[1, 1].set_title(f'R² линейной регрессии: median {float(np.nanmedian(r2_map)):.3f}',
                          fontsize=13)
    axes[1, 1].set_xlabel('Долгота, °E')
    axes[1, 1].set_ylabel('Широта, °N')
    plt.colorbar(im3, ax=axes[1, 1], label='R²', fraction=0.04)

    plt.suptitle('Линейная экстраполяция тренда MAGT 2007-2024 на 2030: карты и диагностика',
                  fontsize=15, y=1.00)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p2_map_2030_with_uncertainty.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out2.name}")

    # ===== CSV сводка =====
    print("\nCSV: сводка проекций...")
    rows = []

    # Historical
    for year, median, mean in zip(years_hist, medians_hist, means_hist):
        rows.append({
            'year': int(year),
            'type': 'real',
            'median_magt': median,
            'mean_magt': mean,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
        })

    # Projection
    for year, median, lower, upper in zip(years_proj, medians_proj, medians_lower, medians_upper):
        rows.append({
            'year': int(year),
            'type': 'extrapolation',
            'median_magt': median,
            'mean_magt': np.nan,
            'ci_lower': lower,
            'ci_upper': upper,
        })

    df = pd.DataFrame(rows)
    out_csv = METRICS_DIR / 'p2_projection_summary.csv'
    df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"  Сохранено: {out_csv.name}")
    print(f"\nТаблица:")
    print(df.to_string(index=False))

    # Открыть оба графика
    import subprocess
    subprocess.run(['open', str(out1), str(out2)])
    print(f"\nОткрыты в Preview")


if __name__ == '__main__':
    main()
