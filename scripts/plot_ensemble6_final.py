"""
Финальные визуализации ensemble прогноза 2026-2035 — 6 моделей.

Исключаем seed=456 как outlier по IQR-тесту (1.5*IQR rule).
Используем 6 моделей: seeds [42, 123, 7, 99, 777, 2024].

Артефакты:
1. p3_ensemble6_trend_2007_2035.png — главный график (улучшенный)
2. p3_ensemble6_uncertainty_map_2035.png — карта uncertainty 2035
3. p3_ensemble6_full_period_maps.png — серия карт на ключевые годы
4. p3_ensemble6_interactive_2007_2035.html — интерактивный slider до 2035

Вход:
    results/maps/p2_yearly_maps.npz (real 2007-2024)  ← inference_p2_yearly_maps.py
    results/maps/p2_real_2025.npz (real 2025)        ← inference_p2_2025.py
    (или оба сразу: inference_p2_history.py)
    models/convlstm_ttop_p3_model_A_seed_{42,123,7,99,777,2024}.pt (6 моделей)
    data/tensor_01deg_synthetic_2026_2035.npz
"""

import numpy as np
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.model import ConvLSTMNet
from src.data import normalize_features
from src.inference import predict_full_map
from src.ablation import features_by_names
from src.landcover import compute_ttop_target

DATA_DIR = BASE_DIR / 'data'
MAPS_DIR = BASE_DIR / 'results' / 'maps'
FIGURES_DIR = BASE_DIR / 'results' / 'figures'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
MODELS_DIR = BASE_DIR / 'models'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

P1_SHIFT = 1.985
PROJ_YEARS = list(range(2026, 2036))
KEPT_SEEDS = [42, 123, 7, 99, 777, 2024]  # без seed=456 (outlier)
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_inference(model_path, X_n, y_nm, t_targets, device, delta_t):
    """Inference одной модели на synthetic 2026-2035."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    H, W = X_n.shape[1], X_n.shape[2]
    n_proj = len(t_targets)
    preds = np.full((n_proj, H, W), np.nan, dtype=np.float32)

    for i, t_target in enumerate(t_targets):
        pred_raw = predict_full_map(
            model=model, X_norm=X_n, y_nan_mask=y_nm,
            t_target=t_target,
            y_mean=ckpt['y_mean'], y_std=ckpt['y_std'],
            device=device,
        )
        pred_lh = pred_raw + delta_t
        pred_final = pred_lh + P1_SHIFT
        preds[i] = pred_final.astype(np.float32)

    return preds, ckpt['val_rmse'], ckpt['best_epoch']


def main():
    device = get_device()
    print(f"Device: {device}\n")
    print(f"Используем {len(KEPT_SEEDS)} моделей (исключая seed=456 как outlier по IQR-тесту)")
    print(f"Seeds: {KEPT_SEEDS}\n")

    # ===== Загрузка =====
    print("Загружаем данные...")
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    magt_hist = history['magt_yearly']
    years_hist = list(history['years'])
    lats = history['lats']
    lons = history['lons']

    real_2025 = np.load(MAPS_DIR / 'p2_real_2025.npz')
    magt_2025 = real_2025['magt']

    synth = np.load(DATA_DIR / 'tensor_01deg_synthetic_2026_2035.npz')
    X_full = synth['X']
    years_full = list(synth['years'])

    y_old = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old['landcover']
    y_full = compute_ttop_target(X_full, landcover, fdd_idx=7, tdd_idx=8)

    delta_t = np.load(DATA_DIR / 'maps_lh_v3.npz')['delta_t']

    core_idx = features_by_names(CORE_6)
    X_core = X_full[..., core_idx]
    train_idx = list(range(4, 20))
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y_full, train_idx)

    t_targets = [years_full.index(y) for y in PROJ_YEARS]

    # ===== Inference 6 моделей =====
    print(f"\n[Inference] 6 моделей × 10 годов...")
    all_preds = []
    val_rmses = []
    for seed in KEPT_SEEDS:
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        t0 = time.time()
        preds, val_rmse, _ = run_inference(model_path, X_n, y_nm, t_targets, device, delta_t)
        all_preds.append(preds)
        val_rmses.append(val_rmse)
        medians = [float(np.nanmedian(preds[i])) for i in range(len(PROJ_YEARS))]
        print(f"  seed={seed:4d} (val_rmse={val_rmse:.3f}): "
              f"medians 2026={medians[0]:+.2f}, 2030={medians[4]:+.2f}, 2035={medians[-1]:+.2f}, "
              f"время {time.time()-t0:.1f}c")

    all_preds = np.stack(all_preds, axis=0)  # (6, 10, H, W)

    # ===== Ensemble статистика =====
    ens_mean = all_preds.mean(axis=0)
    ens_std = all_preds.std(axis=0)

    # Медианы по России
    medians_per_model = np.array([
        [float(np.nanmedian(all_preds[k, i])) for i in range(len(PROJ_YEARS))]
        for k in range(len(KEPT_SEEDS))
    ])
    median_mean = medians_per_model.mean(axis=0)
    median_std = medians_per_model.std(axis=0)

    print(f"\nEnsemble сводка (6 моделей):")
    for i, year in enumerate(PROJ_YEARS):
        print(f"  {year}: {median_mean[i]:+.2f} ± {median_std[i]:.2f}°C")

    # ===== Real data =====
    real_years = years_hist + [2025]
    real_medians = [float(np.nanmedian(magt_hist[i])) for i in range(len(years_hist))] + [float(np.nanmedian(magt_2025))]

    # =============== FIGURE 1: главный график ===============
    print(f"\n[Figure 1] Главный график тренда 2007-2035 (6 моделей)...")

    fig, ax = plt.subplots(figsize=(15, 7))

    # Historical
    ax.plot(years_hist, real_medians[:-1], 'o-', color='steelblue', linewidth=2.5,
            markersize=8, label='Реальные данные 2007-2024', zorder=4)

    # 2025 real
    ax.plot([2025], [real_medians[-1]], 'o', color='darkred', markersize=14,
            label=f'2025 NEW (real): {real_medians[-1]:+.2f}°C', zorder=5,
            markeredgecolor='black', markeredgewidth=1.5)

    # 95% CI envelope (mean ± 2*std)
    ax.fill_between(PROJ_YEARS,
                     median_mean - 2 * median_std,
                     median_mean + 2 * median_std,
                     color='lightgreen', alpha=0.35,
                     label='95% CI ensemble (mean ± 2·std)')

    # 6 индивидуальных моделей (тонкими)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for k, seed in enumerate(KEPT_SEEDS):
        ax.plot(PROJ_YEARS, medians_per_model[k], '--', color=colors[k],
                linewidth=0.9, alpha=0.4, label=f'seed={seed}')

    # Ensemble mean — главная линия
    ax.plot(PROJ_YEARS, median_mean, 's-', color='darkgreen', linewidth=2.8,
            markersize=11, label='Ensemble mean (6 моделей)',
            zorder=6, markerfacecolor='lightgreen', markeredgewidth=2)

    # Разделители
    ax.axvline(2024.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(2025.5, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('MAGT, °C (медиана по России)', fontsize=12)

    delta_mean = median_mean[-1] - real_medians[-1]
    val_rmse_mean = np.mean(val_rmses)
    val_rmse_std = np.std(val_rmses)

    title = (
        f'Финальный прогноз MAGT 2026-2035 — ENSEMBLE из 6 моделей (после IQR filter)\n'
        f'val RMSE = {val_rmse_mean:.3f} ± {val_rmse_std:.3f}°C  •  '
        f'2025 (real): {real_medians[-1]:+.2f}°C  •  '
        f'2035 (mean): {median_mean[-1]:+.2f} ± {median_std[-1]:.2f}°C  •  '
        f'Δ 2025→2035: {delta_mean:+.2f}°C'
    )
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    out1 = FIGURES_DIR / 'p3_ensemble6_trend_2007_2035.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out1.name}")

    # =============== FIGURE 2: карта uncertainty 2035 ===============
    print(f"\n[Figure 2] Карта uncertainty 2035 (6 моделей)...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. Mean MAGT 2035
    im0 = axes[0, 0].pcolormesh(lons, lats, ens_mean[-1],
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 0].set_title(f'Mean MAGT 2035 (6 моделей): median {float(np.nanmedian(ens_mean[-1])):+.2f}°C',
                          fontsize=12)
    axes[0, 0].set_xlabel('Долгота, °E'); axes[0, 0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0, 0], label='MAGT, °C', fraction=0.04)

    # 2. Std per pixel
    im1 = axes[0, 1].pcolormesh(lons, lats, ens_std[-1],
                                  cmap='YlOrRd', vmin=0, vmax=0.5, shading='auto')
    axes[0, 1].set_title(f'Std MAGT 2035 (uncertainty): median {float(np.nanmedian(ens_std[-1])):.3f}°C',
                          fontsize=12)
    axes[0, 1].set_xlabel('Долгота, °E'); axes[0, 1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[0, 1], label='σ, °C', fraction=0.04)

    # 3. CI lower
    ci_low = ens_mean[-1] - 2 * ens_std[-1]
    im2 = axes[1, 0].pcolormesh(lons, lats, ci_low,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[1, 0].set_title(f'95% CI lower bound 2035: median {float(np.nanmedian(ci_low)):+.2f}°C',
                          fontsize=12)
    axes[1, 0].set_xlabel('Долгота, °E'); axes[1, 0].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[1, 0], label='MAGT, °C', fraction=0.04)

    # 4. CI upper
    ci_high = ens_mean[-1] + 2 * ens_std[-1]
    im3 = axes[1, 1].pcolormesh(lons, lats, ci_high,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[1, 1].set_title(f'95% CI upper bound 2035: median {float(np.nanmedian(ci_high)):+.2f}°C',
                          fontsize=12)
    axes[1, 1].set_xlabel('Долгота, °E'); axes[1, 1].set_ylabel('Широта, °N')
    plt.colorbar(im3, ax=axes[1, 1], label='MAGT, °C', fraction=0.04)

    plt.suptitle('Ensemble uncertainty 2035 (6 моделей, после IQR filter)',
                  fontsize=15, y=1.00)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p3_ensemble6_uncertainty_map_2035.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out2.name}")

    # =============== FIGURE 3: серия карт ===============
    print(f"\n[Figure 3] Серия карт: 2025 real + 2028, 2030, 2032, 2035 ensemble...")

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    # 2025 real
    im0 = axes[0].pcolormesh(lons, lats, magt_2025,
                              cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0].set_title(f'2025 REAL: {float(np.nanmedian(magt_2025)):+.2f}°C',
                       fontsize=13, color='darkred', fontweight='bold')
    axes[0].set_xlabel('°E'); axes[0].set_ylabel('°N')

    # 2028, 2030, 2032, 2035
    for ax_idx, year in enumerate([2028, 2030, 2032, 2035]):
        proj_idx = PROJ_YEARS.index(year)
        im = axes[ax_idx + 1].pcolormesh(lons, lats, ens_mean[proj_idx],
                                          cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
        median = float(np.nanmedian(ens_mean[proj_idx]))
        std = float(np.nanmedian(ens_std[proj_idx]))
        axes[ax_idx + 1].set_title(f'{year} forecast: {median:+.2f}°C\n(±{std:.2f}°C σ)',
                                    fontsize=12, color='darkgreen')
        axes[ax_idx + 1].set_xlabel('°E')

    # Один colorbar
    cbar = fig.colorbar(im0, ax=axes, label='MAGT, °C', fraction=0.015, pad=0.02)

    plt.suptitle('Прогресс мерзлоты 2025 (real) → 2035 (ensemble forecast)', fontsize=15)
    out3 = FIGURES_DIR / 'p3_ensemble6_full_period_maps.png'
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out3.name}")

    # =============== FIGURE 4: Plotly slider ===============
    print(f"\n[Figure 4] Plotly slider 2007-2035 (29 кадров)...")

    all_magt = np.concatenate([
        magt_hist,
        magt_2025[np.newaxis, :, :],
        ens_mean,
    ], axis=0)
    all_years = years_hist + [2025] + PROJ_YEARS

    downsample = 2
    all_magt_lite = all_magt[:, ::downsample, ::downsample]
    lats_lite = lats[::downsample]
    lons_lite = lons[::downsample]

    fig = go.Figure(
        data=[go.Heatmap(
            z=all_magt_lite[0], x=lons_lite, y=lats_lite,
            colorscale='RdBu_r', zmin=-14, zmax=4, zmid=0,
            colorbar=dict(title='MAGT, °C'),
            hovertemplate='Долгота: %{x:.2f}°E<br>Широта: %{y:.2f}°N<br>MAGT: %{z:.2f}°C<extra></extra>',
        )],
        layout=go.Layout(
            title=dict(text=f'MAGT в России: <b>{all_years[0]}</b>',
                        x=0.5, xanchor='center'),
            xaxis=dict(title='Долгота, °E', range=[30, 180]),
            yaxis=dict(title='Широта, °N', range=[55, 78], scaleanchor='x', scaleratio=2),
            width=1100, height=600,
        ),
    )

    frames = []
    for i, year in enumerate(all_years):
        median = float(np.nanmedian(all_magt_lite[i]))
        if year <= 2025:
            suffix = ' — реальные данные'
        else:
            suffix = ' — ensemble прогноз (6 моделей)'
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

    out4 = FIGURES_DIR / 'p3_ensemble6_interactive_2007_2035.html'
    fig.write_html(str(out4), include_plotlyjs='cdn')
    print(f"  Сохранено: {out4.name} ({out4.stat().st_size/1024/1024:.1f} МБ)")

    # ===== Финальная сводка =====
    print(f"\n{'='*60}")
    print(f"=== ФИНАЛЬНАЯ СВОДКА (6 моделей) ===")
    print(f"{'='*60}")
    print(f"\n  Validation:        val_rmse = {val_rmse_mean:.3f} ± {val_rmse_std:.3f}°C")
    print(f"\n  Projection 2035:")
    print(f"    Mean:            {median_mean[-1]:+.2f}°C")
    print(f"    Std:             {median_std[-1]:.2f}°C")
    print(f"    95% CI:          [{median_mean[-1] - 2*median_std[-1]:+.2f}, "
          f"{median_mean[-1] + 2*median_std[-1]:+.2f}]°C")
    print(f"    Δ 2025→2035:     {median_mean[-1] - real_medians[-1]:+.2f}°C ({(median_mean[-1] - real_medians[-1])/10:+.3f}°C/год)")

    print(f"\n  Артефакты:")
    print(f"    {out1.name}")
    print(f"    {out2.name}")
    print(f"    {out3.name}")
    print(f"    {out4.name}")

    # Открыть
    import subprocess
    subprocess.run(['open', str(out1), str(out2), str(out3)])


if __name__ == '__main__':
    main()
