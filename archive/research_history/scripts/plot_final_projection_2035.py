"""
Финальные визуализации прогноза MAGT до 2035 через Model A (научная модель).

Артефакты:
1. p3_trend_2007_2035.png — главный график (real 2007-2025 + forecast 2026-2035)
2. p3_map_2035.png — 4-панельная карта 2025 vs 2035 + детали
3. p3_interactive_2007_2035.html — slider на 29 годов

Вход:
    results/maps/p2_yearly_maps.npz             (real 2007-2024)
    results/maps/p2_real_2025.npz               (real 2025)
    results/maps/p3_projection_2026_2035_modelA.npz (прогноз 2026-2035)

Опционально: тоже показывает Model B как sanity check
    results/maps/p3_projection_2026_2035_modelB.npz
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from scipy.stats import linregress

BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / 'results' / 'maps'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ============ Загрузка ============
    print("Загружаем данные...")

    # Real 2007-2024 (P2 yearly maps — 18 годов)
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    magt_2007_2024 = history['magt_yearly']
    years_2007_2024 = list(history['years'])
    lats = history['lats']
    lons = history['lons']

    # Real 2025
    real_2025 = np.load(MAPS_DIR / 'p2_real_2025.npz')
    magt_2025 = real_2025['magt']

    # Forecast 2026-2035 Model A
    forecast_A = np.load(MAPS_DIR / 'p3_projection_2026_2035_modelA.npz')
    magt_A = forecast_A['magt']
    years_A = list(forecast_A['years'])

    # Forecast 2026-2035 Model B (для sanity check)
    forecast_B_path = MAPS_DIR / 'p3_projection_2026_2035_modelB.npz'
    forecast_B = np.load(forecast_B_path) if forecast_B_path.exists() else None
    magt_B = forecast_B['magt'] if forecast_B is not None else None

    print(f"  Real 2007-2024: {magt_2007_2024.shape}")
    print(f"  Real 2025: {magt_2025.shape}")
    print(f"  Forecast Model A 2026-2035: {magt_A.shape}")
    if magt_B is not None:
        print(f"  Forecast Model B 2026-2035: {magt_B.shape}")

    # Объединяем real (2007-2025) + forecast Model A (2026-2035)
    all_magt = np.concatenate([
        magt_2007_2024,
        magt_2025[np.newaxis, :, :],
        magt_A,
    ], axis=0)
    all_years = years_2007_2024 + [2025] + years_A
    print(f"\n  Объединено: {all_magt.shape}, годы {all_years[0]}..{all_years[-1]}")

    # Медианы
    medians_real = [float(np.nanmedian(magt_2007_2024[i])) for i in range(len(years_2007_2024))]
    median_2025 = float(np.nanmedian(magt_2025))
    medians_A = [float(np.nanmedian(magt_A[i])) for i in range(len(years_A))]
    medians_B = [float(np.nanmedian(magt_B[i])) for i in range(len(years_A))] if magt_B is not None else None

    # Все реальные годы 2007-2025
    real_years = years_2007_2024 + [2025]
    real_medians = medians_real + [median_2025]

    # ============ ГРАФИК 1: главный тренд ============
    print("\n[Figure 1] Главный график тренда 2007-2035...")

    # Linear fit на исторических данных 2007-2025 для контекста
    slope_hist, intercept_hist, r_value, p_value, _ = linregress(real_years, real_medians)
    print(f"  Исторический тренд (2007-2025): {slope_hist:+.4f}°C/год, R²={r_value**2:.3f}, p={p_value:.2e}")

    # Линейная регрессия Model A на 2026-2035 для тренда
    slope_A, intercept_A, r_A, _, _ = linregress(years_A, medians_A)
    print(f"  Model A прогноз тренд: {slope_A:+.4f}°C/год")

    fig, ax = plt.subplots(figsize=(15, 7))

    # Линия тренда исторических данных продлённая в будущее
    all_years_arr = np.array(all_years)
    fit_line_hist = slope_hist * all_years_arr + intercept_hist
    ax.plot(all_years_arr, fit_line_hist, ':', color='gray', linewidth=1.5, alpha=0.5,
            label=f'Исторический тренд: {slope_hist:+.3f}°C/год (R²={r_value**2:.2f})')

    # Реальные данные 2007-2024
    ax.plot(years_2007_2024, medians_real, 'o-', color='steelblue', linewidth=2.5,
            markersize=8, label='Реальные данные 2007-2024', zorder=4)

    # 2025 — выделяем (новый real)
    ax.plot([2025], [median_2025], 'o', color='darkred', markersize=14,
            label=f'2025 NEW (real): {median_2025:+.2f}°C', zorder=5,
            markeredgecolor='black', markeredgewidth=1.5)

    # Прогноз Model A
    ax.plot(years_A, medians_A, 's--', color='darkgreen', linewidth=2,
            markersize=9, label='Прогноз Model A (val on 2023-2025, val_rmse=0.785°C)',
            zorder=3, markerfacecolor='lightgreen', markeredgewidth=2)

    # Model B как опциональная sanity check (бледнее)
    if medians_B is not None:
        ax.plot(years_A, medians_B, '^:', color='orange', linewidth=1.2,
                markersize=7, label='Model B (sanity check, без val)', zorder=2, alpha=0.7)

    # Разделительные линии
    ax.axvline(2024.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(2025.5, color='red', linestyle=':', linewidth=1, alpha=0.5)
    y_min, y_max = ax.get_ylim()
    y_text = y_min + 0.3

    delta_A = medians_A[-1] - median_2025

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('MAGT, °C (медиана по России)', fontsize=12)

    title = (
        f'Динамика MAGT в России: 19 лет реальных данных + прогноз до 2035\n'
        f'2025 (real): {median_2025:+.2f}°C  •  '
        f'2035 (Model A): {medians_A[-1]:+.2f}°C  •  '
        f'Δ 2025→2035: {delta_A:+.2f}°C за 10 лет ({slope_A:+.3f}°C/год)'
    )
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)

    # Подписи режимов внизу
    ax.text(2015, ax.get_ylim()[0] + 0.15, 'РЕАЛЬНЫЕ ДАННЫЕ', ha='center',
            fontsize=9, color='steelblue', alpha=0.7,
            transform=ax.transData)
    ax.text(2030.5, ax.get_ylim()[0] + 0.15, 'ПРОГНОЗ (synthetic features → ConvLSTM)',
            ha='center', fontsize=9, color='darkgreen', alpha=0.7,
            transform=ax.transData)

    plt.tight_layout()
    out1 = FIGURES_DIR / 'p3_trend_2007_2035.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out1.name}")

    # ============ ГРАФИК 2: 4-панельная карта 2035 ============
    print("\n[Figure 2] Карта 2035 (4 панели)...")

    magt_2035 = magt_A[-1]
    change_2025_2035 = magt_2035 - magt_2025

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. 2025 (real)
    im0 = axes[0, 0].pcolormesh(lons, lats, magt_2025,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 0].set_title(f'2025 (REAL): median {median_2025:+.2f}°C', fontsize=12)
    axes[0, 0].set_xlabel('Долгота, °E'); axes[0, 0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0, 0], label='MAGT, °C', fraction=0.04)

    # 2. 2035 (Model A forecast)
    im1 = axes[0, 1].pcolormesh(lons, lats, magt_2035,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 1].set_title(f'2035 (Model A FORECAST): median {medians_A[-1]:+.2f}°C', fontsize=12)
    axes[0, 1].set_xlabel('Долгота, °E'); axes[0, 1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[0, 1], label='MAGT, °C', fraction=0.04)

    # 3. Δ 2035-2025
    im2 = axes[1, 0].pcolormesh(lons, lats, change_2025_2035,
                                  cmap='RdBu_r', vmin=-2, vmax=2, shading='auto')
    axes[1, 0].set_title(f'Изменение 2035 - 2025: mean ΔT = {float(np.nanmean(change_2025_2035)):+.2f}°C',
                          fontsize=12)
    axes[1, 0].set_xlabel('Долгота, °E'); axes[1, 0].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[1, 0], label='ΔT, °C', fraction=0.04)

    # 4. Δ Model A - Model B (где модели расходятся)
    if magt_B is not None:
        diff_A_B = magt_A[-1] - magt_B[-1]
        im3 = axes[1, 1].pcolormesh(lons, lats, diff_A_B,
                                      cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
        axes[1, 1].set_title(
            f'Разница моделей 2035: Model A - Model B\n'
            f'median {float(np.nanmedian(diff_A_B)):+.2f}°C (Model A холоднее на ~0.4°C систематически)',
            fontsize=11)
        axes[1, 1].set_xlabel('Долгота, °E'); axes[1, 1].set_ylabel('Широта, °N')
        plt.colorbar(im3, ax=axes[1, 1], label='ΔT, °C', fraction=0.04)
    else:
        axes[1, 1].axis('off')

    plt.suptitle('Прогноз MAGT 2035: real reference 2025 + Model A forecast',
                  fontsize=15, y=1.00)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p3_map_2035.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out2.name}")

    # ============ ГРАФИК 3: интерактивный slider 2007-2035 ============
    print("\n[Figure 3] Plotly slider 2007-2035 (29 кадров)...")

    # Downsample для веба
    downsample = 2
    all_magt_lite = all_magt[:, ::downsample, ::downsample]
    lats_lite = lats[::downsample]
    lons_lite = lons[::downsample]

    first_year = all_years[0]
    first_median = float(np.nanmedian(all_magt_lite[0]))

    fig = go.Figure(
        data=[go.Heatmap(
            z=all_magt_lite[0], x=lons_lite, y=lats_lite,
            colorscale='RdBu_r', zmin=-14, zmax=4, zmid=0,
            colorbar=dict(title='MAGT, °C'),
            hovertemplate='Долгота: %{x:.2f}°E<br>Широта: %{y:.2f}°N<br>MAGT: %{z:.2f}°C<extra></extra>',
        )],
        layout=go.Layout(
            title=dict(
                text=f'MAGT в России: <b>{first_year}</b> (медиана {first_median:+.2f}°C) — реальные данные',
                x=0.5, xanchor='center',
            ),
            xaxis=dict(title='Долгота, °E', range=[30, 180]),
            yaxis=dict(title='Широта, °N', range=[55, 78], scaleanchor='x', scaleratio=2),
            width=1100, height=600,
            margin=dict(l=50, r=50, t=80, b=80),
        ),
    )

    # Frames
    frames = []
    for i, year in enumerate(all_years):
        median = float(np.nanmedian(all_magt_lite[i]))
        if year <= 2025:
            suffix = ' — реальные данные'
        else:
            suffix = ' — прогноз (Model A на synthetic features)'
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=all_magt_lite[i], x=lons_lite, y=lats_lite,
                colorscale='RdBu_r', zmin=-14, zmax=4, zmid=0,
                hovertemplate='Долгота: %{x:.2f}°E<br>Широта: %{y:.2f}°N<br>MAGT: %{z:.2f}°C<extra></extra>',
            )],
            name=str(year),
            layout=go.Layout(title=dict(text=f'MAGT в России: <b>{year}</b> (медиана {median:+.2f}°C){suffix}')),
        ))
    fig.frames = frames

    slider_steps = [
        dict(method='animate', label=str(year),
              args=[[str(year)],
                    dict(mode='immediate', frame=dict(duration=300, redraw=True),
                          transition=dict(duration=200))])
        for year in all_years
    ]

    fig.update_layout(
        updatemenus=[dict(
            type='buttons', direction='left', x=0.1, y=-0.05,
            xanchor='right', yanchor='top',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(mode='immediate', frame=dict(duration=600, redraw=True),
                                       transition=dict(duration=300), fromcurrent=True)]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(mode='immediate', frame=dict(duration=0, redraw=False),
                                         transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            steps=slider_steps, active=0, x=0.15, y=-0.05, len=0.8,
            currentvalue=dict(prefix='Год: ', font=dict(size=14)),
            transition=dict(duration=200),
        )],
    )

    out3 = FIGURES_DIR / 'p3_interactive_2007_2035.html'
    fig.write_html(str(out3), include_plotlyjs='cdn')
    print(f"  Сохранено: {out3.name} ({out3.stat().st_size/1024/1024:.1f} МБ)")

    # ============ Открыть результаты ============
    import subprocess
    subprocess.run(['open', str(out1), str(out2), str(out3)])
    print(f"\nВсе 3 файла открыты")


if __name__ == '__main__':
    main()
