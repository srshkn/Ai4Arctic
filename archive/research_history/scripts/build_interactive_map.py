"""
Построение интерактивной карты MAGT 2007-2024 (18 годов) с slider.

Создаёт два HTML файла:
1. p2_interactive_full.html — полное разрешение 0.1° (для отчёта)
2. p2_interactive_lite.html — downsampled 0.2° (для устной защиты)

Использует Plotly heatmap с frames для slider анимации.

Вход:
  results/maps/p2_yearly_maps.npz

Выход:
  results/figures/p2_interactive_full.html
  results/figures/p2_interactive_lite.html
"""

import numpy as np
import plotly.graph_objects as go
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_FILE = BASE_DIR / 'results' / 'maps' / 'p2_yearly_maps.npz'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_map(magt, years, lats, lons, downsample=1, title_suffix=''):
    """
    Построение Plotly heatmap с slider по годам.

    Parameters
    ----------
    magt : (T, H, W) array
        Карты MAGT по годам.
    years : list of int
        Годы для отображения.
    downsample : int
        Если >1, берёт каждый downsample-й пиксель по lat/lon.
    """
    if downsample > 1:
        magt = magt[:, ::downsample, ::downsample]
        lats = lats[::downsample]
        lons = lons[::downsample]

    T, H, W = magt.shape
    print(f"  Карты: {T} годов, разрешение {H}×{W} = {H*W:,} точек/год")

    # Базовый кадр — первый год
    first_year = int(years[0])
    first_median = float(np.nanmedian(magt[0]))
    fig = go.Figure(
        data=[go.Heatmap(
            z=magt[0],
            x=lons,
            y=lats,
            colorscale='RdBu_r',
            zmin=-14, zmax=4,
            zmid=0,
            colorbar=dict(title='MAGT, °C'),
            hovertemplate='Долгота: %{x:.2f}°E<br>Широта: %{y:.2f}°N<br>MAGT: %{z:.2f}°C<extra></extra>',
        )],
        layout=go.Layout(
            title=dict(
                text=f'MAGT в России: <b>{first_year}</b> '
                     f'(медиана {first_median:+.2f}°C){title_suffix}',
                x=0.5, xanchor='center',
            ),
            xaxis=dict(title='Долгота, °E', range=[30, 180]),
            yaxis=dict(title='Широта, °N', range=[55, 78], scaleanchor='x', scaleratio=2),
            width=1100, height=600,
            margin=dict(l=50, r=50, t=80, b=80),
        ),
    )

    # Кадры для slider
    frames = []
    for i, year in enumerate(years):
        median = float(np.nanmedian(magt[i]))
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=magt[i],
                x=lons, y=lats,
                colorscale='RdBu_r', zmin=-14, zmax=4, zmid=0,
                hovertemplate='Долгота: %{x:.2f}°E<br>Широта: %{y:.2f}°N<br>MAGT: %{z:.2f}°C<extra></extra>',
            )],
            name=str(int(year)),
            layout=go.Layout(
                title=dict(text=f'MAGT в России: <b>{int(year)}</b> '
                                f'(медиана {median:+.2f}°C){title_suffix}'),
            ),
        ))
    fig.frames = frames

    # Slider
    slider_steps = []
    for i, year in enumerate(years):
        slider_steps.append(dict(
            method='animate',
            label=str(int(year)),
            args=[[str(int(year))],
                  dict(mode='immediate',
                       frame=dict(duration=300, redraw=True),
                       transition=dict(duration=200))],
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='left',
            x=0.1, y=-0.05,
            xanchor='right', yanchor='top',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(mode='immediate',
                                       frame=dict(duration=600, redraw=True),
                                       transition=dict(duration=300),
                                       fromcurrent=True)]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(mode='immediate',
                                         frame=dict(duration=0, redraw=False),
                                         transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            steps=slider_steps,
            active=0,
            x=0.15, y=-0.05,
            len=0.8,
            currentvalue=dict(prefix='Год: ', font=dict(size=14)),
            transition=dict(duration=200),
        )],
    )
    return fig


def main():
    print("Загружаем p2_yearly_maps.npz...")
    data = np.load(MAPS_FILE)
    magt = data['magt_yearly']
    years = [int(y) for y in data['years']]
    lats = data['lats']
    lons = data['lons']
    print(f"  {len(years)} годов: {years[0]}..{years[-1]}")
    print(f"  Сетка: {len(lats)}×{len(lons)}")

    # --- Версия 1: полное разрешение 0.1° ---
    print("\n=== Версия FULL (0.1°) ===")
    fig_full = build_map(magt, years, lats, lons, downsample=1,
                          title_suffix=' — модель ConvLSTM P2')
    out_full = FIGURES_DIR / 'p2_interactive_full.html'
    fig_full.write_html(str(out_full), include_plotlyjs='cdn')
    size_mb = out_full.stat().st_size / 1024 / 1024
    print(f"  Сохранено: {out_full.name} ({size_mb:.1f} МБ)")

    # --- Версия 2: downsampled 0.2° ---
    print("\n=== Версия LITE (0.2°, downsample 2x) ===")
    fig_lite = build_map(magt, years, lats, lons, downsample=2,
                          title_suffix=' — модель ConvLSTM P2 (downsampled)')
    out_lite = FIGURES_DIR / 'p2_interactive_lite.html'
    fig_lite.write_html(str(out_lite), include_plotlyjs='cdn')
    size_mb_lite = out_lite.stat().st_size / 1024 / 1024
    print(f"  Сохранено: {out_lite.name} ({size_mb_lite:.1f} МБ)")

    # Открыть LITE версию в браузере (она быстрее для просмотра)
    import subprocess
    subprocess.run(['open', str(out_lite)])
    print(f"\nОткрыт в браузере: {out_lite.name}")
    print(f"Для FULL: open {out_full}")


if __name__ == '__main__':
    main()
