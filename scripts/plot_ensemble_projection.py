"""
Визуализация ensemble прогноза 2026-2035: 3 модели Model A с разными seeds.

Цель: показать robustness прогноза через ensemble vatiance.

Артефакты:
1. p3_ensemble_trend_2007_2035.png — главный график с CI envelope
2. p3_ensemble_uncertainty_map_2035.png — карта std по пикселям для 2035
3. p3_ensemble_summary.csv — таблица

Вход:
    results/maps/p2_yearly_maps.npz        (real 2007-2024)
    results/maps/p2_real_2025.npz          (real 2025)
    models/convlstm_ttop_p3_model_A_seed_{42,123,456}.pt  (3 модели)

Pipeline:
1. Запускаем inference каждой из 3 моделей на synthetic 2026-2035
2. Считаем mean, std, min, max по ensemble
3. Визуализируем
"""

import numpy as np
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
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
PROJ_YEARS = list(range(2026, 2036))   # 2026-2035
SEEDS = [42, 123, 456]
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_synthetic_tensor():
    """Создаёт synthetic тензор 33 года (2003-2035)."""
    print("Создаём synthetic тензор (если ещё нет)...")

    cache_path = DATA_DIR / 'tensor_01deg_synthetic_2026_2035.npz'
    if cache_path.exists():
        print(f"  Используем кэш: {cache_path.name}")
        tensor = np.load(cache_path)
        return (tensor['X'].astype(np.float32),
                list(tensor['years']),
                tensor['lats'], tensor['lons'])

    # Если нет — создаём заново (но обычно есть из предыдущего projection_2026_2035.py)
    print("  ОШИБКА: synthetic тензор не найден. Запусти projection_2026_2035.py сначала.")
    sys.exit(1)


def run_inference(model_path, X_full, y_full, train_idx, t_targets, device, delta_t):
    """Inference одной модели на список target years."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    core_idx = features_by_names(CORE_6)
    X_core = X_full[..., core_idx]
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y_full, train_idx)

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

    return preds


def main():
    device = get_device()
    print(f"Device: {device}\n")

    # ===== Загрузка =====
    X_full, years_full, lats, lons = build_synthetic_tensor()
    print(f"  X_full: {X_full.shape}, годы {years_full[0]}..{years_full[-1]}")

    # Пересчёт target
    y_old = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old['landcover']
    y_full = compute_ttop_target(X_full, landcover, fdd_idx=7, tdd_idx=8)

    # lh delta_t
    delta_t = np.load(DATA_DIR / 'maps_lh_v3.npz')['delta_t']

    # Train idx для Model A (2007-2022, индексы 4..19)
    train_idx = list(range(4, 20))
    t_targets = [years_full.index(y) for y in PROJ_YEARS]

    # ===== Inference всех 3 моделей Model A =====
    print(f"\n[Inference] Запускаем 3 модели A (seeds: {SEEDS}):")

    all_preds = []  # (3, 10, H, W)

    for seed in SEEDS:
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        if not model_path.exists():
            print(f"  ОШИБКА: {model_path.name} не найден")
            continue
        print(f"\n  seed={seed}...")
        t0 = time.time()
        preds = run_inference(model_path, X_full, y_full, train_idx, t_targets, device, delta_t)
        elapsed = time.time() - t0
        medians = [float(np.nanmedian(preds[i])) for i in range(len(PROJ_YEARS))]
        print(f"    время {elapsed:.1f}с, medians: {[f'{m:+.2f}' for m in medians]}")
        all_preds.append(preds)

    all_preds = np.stack(all_preds, axis=0)  # (3, 10, H, W)
    print(f"\n  All predictions: {all_preds.shape}")

    # ===== Статистика по ensemble =====
    print(f"\n[Ensemble статистика]")
    ens_mean = all_preds.mean(axis=0)    # (10, H, W) — mean per pixel per year
    ens_std = all_preds.std(axis=0)      # (10, H, W) — std per pixel per year
    ens_min = all_preds.min(axis=0)
    ens_max = all_preds.max(axis=0)

    # Медианы по России для каждой модели
    medians_per_model = [[float(np.nanmedian(all_preds[k, i])) for i in range(len(PROJ_YEARS))]
                         for k in range(len(SEEDS))]
    medians_per_model = np.array(medians_per_model)  # (3, 10)

    # Mean и std медиан
    median_mean = medians_per_model.mean(axis=0)
    median_std = medians_per_model.std(axis=0)
    median_min = medians_per_model.min(axis=0)
    median_max = medians_per_model.max(axis=0)

    print(f"\n  Сводка по годам (median ± std):")
    print(f"  ┌──────┬─────────┬─────────┬─────────┬─────────┐")
    print(f"  │ Year │  seed42 │ seed123 │ seed456 │  mean ± std │")
    print(f"  ├──────┼─────────┼─────────┼─────────┼─────────┤")
    for i, year in enumerate(PROJ_YEARS):
        m42 = medians_per_model[0, i]
        m123 = medians_per_model[1, i]
        m456 = medians_per_model[2, i]
        mean = median_mean[i]
        std = median_std[i]
        print(f"  │ {year} │  {m42:+.3f} │ {m123:+.3f} │ {m456:+.3f} │ {mean:+.3f} ± {std:.3f} │")
    print(f"  └──────┴─────────┴─────────┴─────────┴─────────┘")

    # ===== Сохранение CSV =====
    print(f"\n[CSV] Сохраняем сводку...")
    rows = []
    for i, year in enumerate(PROJ_YEARS):
        rows.append({
            'year': year,
            'seed_42': medians_per_model[0, i],
            'seed_123': medians_per_model[1, i],
            'seed_456': medians_per_model[2, i],
            'mean': median_mean[i],
            'std': median_std[i],
            'min': median_min[i],
            'max': median_max[i],
        })
    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / 'p3_ensemble_projection_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  Сохранено: {csv_path.relative_to(BASE_DIR)}")

    # ===== График 1: главный с CI envelope =====
    print(f"\n[Figure 1] Главный график с ensemble envelope...")

    # Historical
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    hist_magt = history['magt_yearly']
    hist_years = list(history['years'])
    hist_medians = [float(np.nanmedian(hist_magt[i])) for i in range(len(hist_years))]

    real_2025 = float(np.nanmedian(np.load(MAPS_DIR / 'p2_real_2025.npz')['magt']))

    real_years = hist_years + [2025]
    real_medians = hist_medians + [real_2025]

    fig, ax = plt.subplots(figsize=(15, 7))

    # Реальные данные
    ax.plot(hist_years, hist_medians, 'o-', color='steelblue', linewidth=2.5,
            markersize=8, label='Реальные данные 2007-2024', zorder=4)

    # 2025 real (выделен)
    ax.plot([2025], [real_2025], 'o', color='darkred', markersize=14,
            label=f'2025 NEW (real): {real_2025:+.2f}°C', zorder=5,
            markeredgecolor='black', markeredgewidth=1.5)

    # Ensemble CI envelope (mean ± 2*std для 95%)
    ax.fill_between(PROJ_YEARS,
                     median_mean - 2 * median_std,
                     median_mean + 2 * median_std,
                     color='lightgreen', alpha=0.3,
                     label='95% CI ensemble (mean ± 2·std)')

    # Все 3 модели — тонкие линии
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for k, seed in enumerate(SEEDS):
        ax.plot(PROJ_YEARS, medians_per_model[k], '--', color=colors[k],
                linewidth=1.0, alpha=0.4, label=f'seed={seed}')

    # Mean ensemble — основная линия
    ax.plot(PROJ_YEARS, median_mean, 's-', color='darkgreen', linewidth=2.5,
            markersize=10, label='Ensemble mean (главный прогноз)',
            zorder=3, markerfacecolor='lightgreen', markeredgewidth=2)

    # Разделители
    ax.axvline(2024.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(2025.5, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Год', fontsize=12)
    ax.set_ylabel('MAGT, °C (медиана по России)', fontsize=12)

    delta_mean = median_mean[-1] - real_2025
    title = (
        f'Прогноз MAGT 2026-2035 через ENSEMBLE из 3 моделей (Model A)\n'
        f'2025 (real): {real_2025:+.2f}°C  •  '
        f'2035 (ensemble mean): {median_mean[-1]:+.2f} ± {median_std[-1]:.2f}°C  •  '
        f'Δ 2025→2035: {delta_mean:+.2f}°C'
    )
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    plt.tight_layout()
    out1 = FIGURES_DIR / 'p3_ensemble_trend_2007_2035.png'
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out1.name}")

    # ===== График 2: карта uncertainty (std) для 2035 =====
    print(f"\n[Figure 2] Карта uncertainty (std) ensemble для 2035...")

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # 1. Mean MAGT 2035
    im0 = axes[0, 0].pcolormesh(lons, lats, ens_mean[-1],
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[0, 0].set_title(f'Mean MAGT 2035 (ensemble): median {float(np.nanmedian(ens_mean[-1])):+.2f}°C',
                          fontsize=12)
    axes[0, 0].set_xlabel('Долгота, °E'); axes[0, 0].set_ylabel('Широта, °N')
    plt.colorbar(im0, ax=axes[0, 0], label='MAGT, °C', fraction=0.04)

    # 2. Std per pixel
    im1 = axes[0, 1].pcolormesh(lons, lats, ens_std[-1],
                                  cmap='YlOrRd', vmin=0, vmax=1.0, shading='auto')
    axes[0, 1].set_title(f'Std MAGT 2035 (uncertainty): median {float(np.nanmedian(ens_std[-1])):.3f}°C',
                          fontsize=12)
    axes[0, 1].set_xlabel('Долгота, °E'); axes[0, 1].set_ylabel('Широта, °N')
    plt.colorbar(im1, ax=axes[0, 1], label='σ, °C', fraction=0.04)

    # 3. CI lower (mean - 2*std)
    ci_low = ens_mean[-1] - 2 * ens_std[-1]
    im2 = axes[1, 0].pcolormesh(lons, lats, ci_low,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[1, 0].set_title(f'95% CI lower bound 2035: median {float(np.nanmedian(ci_low)):+.2f}°C',
                          fontsize=12)
    axes[1, 0].set_xlabel('Долгота, °E'); axes[1, 0].set_ylabel('Широта, °N')
    plt.colorbar(im2, ax=axes[1, 0], label='MAGT, °C', fraction=0.04)

    # 4. CI upper (mean + 2*std)
    ci_high = ens_mean[-1] + 2 * ens_std[-1]
    im3 = axes[1, 1].pcolormesh(lons, lats, ci_high,
                                  cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    axes[1, 1].set_title(f'95% CI upper bound 2035: median {float(np.nanmedian(ci_high)):+.2f}°C',
                          fontsize=12)
    axes[1, 1].set_xlabel('Долгота, °E'); axes[1, 1].set_ylabel('Широта, °N')
    plt.colorbar(im3, ax=axes[1, 1], label='MAGT, °C', fraction=0.04)

    plt.suptitle('Ensemble uncertainty (3 модели × разные seeds) для прогноза 2035',
                  fontsize=15, y=1.00)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p3_ensemble_uncertainty_map_2035.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out2.name}")

    # ===== Финальная сводка =====
    print(f"\n{'='*60}")
    print(f"=== ФИНАЛЬНАЯ ENSEMBLE СВОДКА ===")
    print(f"{'='*60}")
    print(f"\n  2025 (real):           {real_2025:+.2f}°C")
    print(f"\n  2035 (ensemble mean):  {median_mean[-1]:+.2f} ± {median_std[-1]:.2f}°C")
    print(f"  2035 (CI 95%):         [{median_mean[-1] - 2*median_std[-1]:+.2f}, "
          f"{median_mean[-1] + 2*median_std[-1]:+.2f}]°C")
    print(f"\n  Δ 2025→2035:           {median_mean[-1] - real_2025:+.2f}°C за 10 лет")
    print(f"\n  Uncertainty по пикселям (median std):")
    for i, year in enumerate(PROJ_YEARS):
        pixel_std = float(np.nanmedian(ens_std[i]))
        print(f"    {year}: {pixel_std:.3f}°C")

    print(f"\nАртефакты:")
    print(f"  {out1.name}")
    print(f"  {out2.name}")
    print(f"  {csv_path.name}")

    # Открыть
    import subprocess
    subprocess.run(['open', str(out1), str(out2)])


if __name__ == '__main__':
    main()
