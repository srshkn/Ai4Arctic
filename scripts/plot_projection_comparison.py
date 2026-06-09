"""
P2-projection: визуализации прогноза 2025-2030.

Создаёт 4 артефакта:
1. p2_projection_comparison.png — два метода на одном графике
2. p2_map_2030_convlstm.png — карта MAGT 2030 через ConvLSTM + разность с linear
3. p2_interactive_with_projection.html — slider 2007-2030 (real + synthetic)
4. p2_projection_summary_extended.csv — таблица с двумя методами

Вход:
  results/maps/p2_yearly_maps.npz                (real 2007-2024)
  results/maps/p2_extrapolation_2025_2030.npz   (linear extrapolation)
  results/maps/p2_projection_synthetic.npz       (ConvLSTM на synthetic features)
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
    print("Загружаем данные...")
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    linear = np.load(MAPS_DIR / 'p2_extrapolation_2025_2030.npz')
    convlstm = np.load(MAPS_DIR / 'p2_projection_synthetic.npz')

    magt_hist = history['magt_yearly']
    years_hist = history['years']
    lats = history['lats']
    lons = history['lons']

    magt_linear = linear['magt_extrapolated']
    ci_lower = linear['magt_ci_lower']
    ci_upper = linear['magt_ci_upper']
    years_proj = linear['years_extrap']

    magt_convlstm = convlstm['magt_yearly']

    print(f"  Real: {magt_hist.shape}, годы {years_hist[0]}..{years_hist[-1]}")
    print(f"  Linear projection: {magt_linear.shape}")
    print(f"  ConvLSTM projection: {magt_convlstm.shape}")

    # ===== Figure 1: сравнительный график =====
    print("\nFigure 1: сравнение двух методов...")

    medians_hist = np.array([float(np.nanmedian(magt_hist[i])) for i in range(len(years_hist))])
    medians_linear = np.array([float(np.nanmedian(magt_linear[i])) for i in range(len(years_proj))])
    medians_convlstm = np.array([float(np.nanmedian(magt_convlstm[i])) for i in range(len(years_proj))])
    medians_lower = np.array([float(np.nanmedian(ci_lower[i])) for i in range(len(years_proj))])
    medians_upper = np.array([float(np.nanmedian(ci_upper[i])) for i in range(len(years_proj))])

    # Linear fit на исторических для базы
    slope_hist, intercept_hist, r_value, p_value, _ = linregress(years_hist, medians_hist)
    print(f"  Тренд исторический: {slope_hist:+.4f}°C/год, R²={r_value**2:.3f}, p={p_value:.2e}")

    fig, ax = plt.subplots(figsize=(15, 7))

    # Историческая линия тренда
    all_years_extended = np.concatenate([years_hist, years_proj])
    fit_line = slope_hist * all_years_extended + intercept_hist
    ax.plot(all_years_extended, fit_line, ':', color='gray', linewidth=1.5, alpha=0.6,
            label=f'Исторический тренд: {slope_hist:+.3f}°C/год (R²={r_value**2:.2f})')

    # Реальные данные
    ax.plot(years_hist, medians_hist, 'o-', color='steelblue', linewidth=2.5,
            markersize=9, label='Медиана MAGT (реальные данные 2007-2024)', zorder=4)

    # 95% CI envelope для linear
    ax.fill_between(years_proj, medians_lower, medians_upper,
                     color='lightcoral', alpha=0.18, label='95% CI (линейная экстраполяция)')

    # Linear projection
    ax.plot(years_proj, medians_linear, 's--', color='darkorange', linewidth=2,
            markersize=9, label='Метод 1: Линейная экстраполяция MAGT', zorder=3,
            markerfacecolor='white', markeredgewidth=2)

    # ConvLSTM projection
    ax.plot(years_proj, medians_convlstm, '^--', color='darkgreen', linewidth=2,
            markersize=10, label='Метод 2: Synthetic features → ConvLSTM',
            zorder=3, markerfacecolor='lightgreen', markeredgewidth=2)

    # Разделительная линия
    ax.axvline(2024.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    y_min, y_max = ax.get_ylim()
    ax.text(2024.5, y_min + 0.3, '← реальные данные | прогноз →',
            ha='center', fontsize=10, color='gray', alpha=0.8)

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('MAGT, °C', fontsize=12)

    delta_linear = medians_linear[-1] - medians_hist[-1]
    delta_convlstm = medians_convlstm[-1] - medians_hist[-1]
    delta_diff = medians_linear[-1] - medians_convlstm[-1]

    ax.set_title(
        f'Прогноз MAGT 2025-2030 двумя методами\n'
        f'Δ(2024→2030): Linear = {delta_linear:+.2f}°C, ConvLSTM = {delta_convlstm:+.2f}°C, '
        f'разница = {delta_diff:.2f}°C (физическая инерция мерзлоты)',
        fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    ax.set_xticks(all_years_extended)
    ax.set_xticklabels(all_years_extended, rotation=45)

    plt.tight_layout()
    out1 = FIGURES_DIR / 'p2_projection_comparison.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out1.name}")

    # ===== Figure 2: карта 2030 ConvLSTM + разность с linear =====
    print("\nFigure 2: карта 2030 ConvLSTM + разность...")

    magt_2030_convlstm = magt_convlstm[-1]  # 2030
    magt_2030_linear = magt_linear[-1]      # 2030
    magt_2024 = magt_hist[-1]

    diff_methods = magt_2030_convlstm - magt_2030_linear  # convlstm - linear

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. MAGT 2024 (real)
    im0 = axes[0, 0].pcolormesh(lons, lats, magt_2024,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 0].set_title(f'2024 (реальные данные): median {float(np.nanmedian(magt_2024)):+.2f}°C',
                          fontsize=12)
    axes[0, 0].set_xlabel('Долгота, °E'); axes[0, 0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0, 0], label='MAGT, °C', fraction=0.04)

    # 2. ConvLSTM 2030
    im1 = axes[0, 1].pcolormesh(lons, lats, magt_2030_convlstm,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 1].set_title(f'2030 (ConvLSTM на synthetic features): median {float(np.nanmedian(magt_2030_convlstm)):+.2f}°C',
                          fontsize=12)
    axes[0, 1].set_xlabel('Долгота, °E'); axes[0, 1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[0, 1], label='MAGT, °C', fraction=0.04)

    # 3. Linear 2030
    im2 = axes[1, 0].pcolormesh(lons, lats, magt_2030_linear,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[1, 0].set_title(f'2030 (линейная экстраполяция): median {float(np.nanmedian(magt_2030_linear)):+.2f}°C',
                          fontsize=12)
    axes[1, 0].set_xlabel('Долгота, °E'); axes[1, 0].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[1, 0], label='MAGT, °C', fraction=0.04)

    # 4. Difference: ConvLSTM - Linear
    im3 = axes[1, 1].pcolormesh(lons, lats, diff_methods,
                                  cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    axes[1, 1].set_title(
        f'Разница методов: ConvLSTM - Linear\n'
        f'median {float(np.nanmedian(diff_methods)):+.2f}°C (отрицательное = ConvLSTM холоднее = инерция мерзлоты)',
        fontsize=11)
    axes[1, 1].set_xlabel('Долгота, °E'); axes[1, 1].set_ylabel('Широта, °N')
    plt.colorbar(im3, ax=axes[1, 1], label='ΔT, °C', fraction=0.04)

    plt.suptitle('Прогноз MAGT 2030: сравнение методов прямой экстраполяции и model-based projection',
                  fontsize=15, y=1.00)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p2_map_2030_convlstm.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out2.name}")

    # ===== Figure 3: расширенный Plotly slider 2007-2030 =====
    print("\nFigure 3: расширенный интерактивный slider 2007-2030...")

    # Объединяем real (2007-2024) + ConvLSTM projection (2025-2030)
    all_magt = np.concatenate([magt_hist, magt_convlstm], axis=0)  # (24, H, W)
    all_years_int = list(years_hist) + [int(y) for y in years_proj]

    print(f"  Объединено: {all_magt.shape}, годы {all_years_int[0]}..{all_years_int[-1]}")

    # Downsample для веба (slider будет быстрее)
    downsample = 2
    all_magt_lite = all_magt[:, ::downsample, ::downsample]
    lats_lite = lats[::downsample]
    lons_lite = lons[::downsample]

    # Базовый кадр — первый год
    first_year = all_years_int[0]
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
                text=f'MAGT в России: <b>{first_year}</b> (медиана {first_median:+.2f}°C) — модель ConvLSTM P2',
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
    for i, year in enumerate(all_years_int):
        median = float(np.nanmedian(all_magt_lite[i]))
        suffix = ' — реальные данные' if year <= 2024 else ' — прогноз (synthetic features)'
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

    # Slider
    slider_steps = [
        dict(method='animate', label=str(year),
              args=[[str(year)],
                    dict(mode='immediate', frame=dict(duration=300, redraw=True),
                          transition=dict(duration=200))])
        for year in all_years_int
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

    out3 = FIGURES_DIR / 'p2_interactive_with_projection.html'
    fig.write_html(str(out3), include_plotlyjs='cdn')
    print(f"  Сохранено: {out3.name} ({out3.stat().st_size/1024/1024:.1f} МБ)")

    # ===== CSV сводка =====
    print("\nCSV сводка...")
    rows = []
    for year, median in zip(years_hist, medians_hist):
        rows.append({'year': int(year), 'type': 'real', 'method': 'observation',
                      'median_magt': median, 'ci_lower': np.nan, 'ci_upper': np.nan})
    for year, m_lin, lower, upper, m_conv in zip(
        years_proj, medians_linear, medians_lower, medians_upper, medians_convlstm
    ):
        rows.append({'year': int(year), 'type': 'projection', 'method': 'linear_extrapolation',
                      'median_magt': m_lin, 'ci_lower': lower, 'ci_upper': upper})
        rows.append({'year': int(year), 'type': 'projection', 'method': 'convlstm_synthetic',
                      'median_magt': m_conv, 'ci_lower': np.nan, 'ci_upper': np.nan})
    df = pd.DataFrame(rows)
    out_csv = METRICS_DIR / 'p2_projection_summary_extended.csv'
    df.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"  Сохранено: {out_csv.name}")

    # Финальная сводка
    print(f"\n=== ФИНАЛЬНАЯ СВОДКА ===")
    print(f"  2024 (real):           median {medians_hist[-1]:+.2f}°C")
    print(f"  2030 Linear:           median {medians_linear[-1]:+.2f}°C  (Δ {delta_linear:+.2f}°C)")
    print(f"  2030 ConvLSTM:         median {medians_convlstm[-1]:+.2f}°C  (Δ {delta_convlstm:+.2f}°C)")
    print(f"  Разница методов:       {delta_diff:+.2f}°C")
    print(f"  Интерпретация:         ConvLSTM учитывает физическую инерцию мерзлоты")

    # Открыть в Preview
    import subprocess
    subprocess.run(['open', str(out1), str(out2), str(out3)])
    print(f"\nОткрыты в браузере/Preview")


if __name__ == '__main__':
    main()
