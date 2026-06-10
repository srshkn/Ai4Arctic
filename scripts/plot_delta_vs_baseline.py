"""
Серия карт ΔMAGT относительно baseline mean(2018-2024).

Исправление от предыдущей версии (vs real 2025):
- Baseline = усреднение 7 годов 2018-2024 (без La Niña outlier 2025)
- Это стандартная практика в climate science: multi-year reference period
- Результат: чистый climate change signal без inter-annual noise

Вход:
    results/maps/p2_yearly_maps.npz                  (real 2007-2024)
    data/tensor_01deg_synthetic_2026_2035.npz       (synthetic features)
    models/convlstm_ttop_p3_model_A_seed_{...}.pt    (6 моделей)

Выход:
    results/figures/p3_ensemble6_delta_vs_baseline.png
    results/figures/p3_ensemble6_baseline_map.png    (само reference)
"""

import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from pathlib import Path

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
MODELS_DIR = BASE_DIR / 'models'

P1_SHIFT = 1.985
PROJ_YEARS = list(range(2026, 2036))
FILTERED_SEEDS = [42, 123, 7, 99, 777, 2024]
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']

# Baseline: 7-year reference period
BASELINE_YEARS = list(range(2018, 2025))   # 2018, 2019, ..., 2024


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_inference(model_path, X_n, y_nm, t_targets, device, delta_t):
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

    return preds


def main():
    device = get_device()
    print(f"Device: {device}\n")

    # ===== Baseline: mean(2018-2024) =====
    print(f"[1] Строим baseline = mean({BASELINE_YEARS[0]}-{BASELINE_YEARS[-1]})...")
    history = np.load(MAPS_DIR / 'p2_yearly_maps.npz')
    hist_magt = history['magt_yearly']
    hist_years = list(history['years'])
    lats = history['lats']
    lons = history['lons']

    baseline_idx = [hist_years.index(y) for y in BASELINE_YEARS]
    print(f"  Индексы в p2_yearly_maps: {baseline_idx}")
    print(f"  Соответствующие годы: {[hist_years[i] for i in baseline_idx]}")

    baseline_stack = hist_magt[baseline_idx]  # (7, H, W)
    baseline = np.nanmean(baseline_stack, axis=0)  # (H, W)
    baseline_median = float(np.nanmedian(baseline))
    print(f"  Baseline MAGT (mean 2018-2024): median {baseline_median:+.2f}°C")
    print(f"  Range: [{float(np.nanmin(baseline)):.1f}, {float(np.nanmax(baseline)):.1f}]°C")

    # Сохраним baseline для последующего использования
    np.savez_compressed(
        MAPS_DIR / 'baseline_2018_2024.npz',
        magt=baseline.astype(np.float32),
        years_used=np.array(BASELINE_YEARS),
        lats=lats, lons=lons,
        description='Reference baseline: mean MAGT over 2018-2024 (7 years)',
    )
    print(f"  Сохранено: baseline_2018_2024.npz")

    # ===== Inference 6 моделей для ensemble mean =====
    print(f"\n[2] Inference 6 моделей на 2026-2035...")
    synth = np.load(DATA_DIR / 'tensor_01deg_synthetic_2026_2035.npz')
    X_full = synth['X']
    years_full = list(synth['years'])

    y_old = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old['landcover']
    y_full = compute_ttop_target(X_full, landcover, fdd_idx=7, tdd_idx=8)

    core_idx = features_by_names(CORE_6)
    X_core = X_full[..., core_idx]
    train_idx = list(range(4, 20))
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y_full, train_idx)

    delta_t = np.load(DATA_DIR / 'maps_lh_v3.npz')['delta_t']
    t_targets = [years_full.index(yr) for yr in PROJ_YEARS]

    all_preds = []
    for seed in FILTERED_SEEDS:
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        if not model_path.exists():
            continue
        preds = run_inference(model_path, X_n, y_nm, t_targets, device, delta_t)
        all_preds.append(preds)
        print(f"  seed={seed}: ok")

    all_preds = np.stack(all_preds, axis=0)
    ens_mean = all_preds.mean(axis=0)  # (10, H, W)

    # ===== Дельта-карты vs baseline =====
    print(f"\n[3] Строим серию ΔMAGT vs baseline...")

    years_to_show = [2028, 2030, 2032, 2034, 2035]
    indices = [PROJ_YEARS.index(y) for y in years_to_show]

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    for ax, year, idx in zip(axes, years_to_show, indices):
        delta = ens_mean[idx] - baseline
        mean_delta = float(np.nanmean(delta))
        median_delta = float(np.nanmedian(delta))
        max_delta = float(np.nanmax(delta))
        pos_pct = float(np.nanmean(delta > 0) * 100)

        im = ax.pcolormesh(lons, lats, delta,
                            cmap='Reds', vmin=0, vmax=2.0, shading='auto')
        ax.set_title(
            f'Δ MAGT ({year} − baseline 2018-2024)\n'
            f'mean = {mean_delta:+.2f}°C, median = {median_delta:+.2f}°C',
            fontsize=11)
        ax.set_xlabel('Долгота, °E', fontsize=9)
        ax.set_ylabel('Широта, °N', fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, label='ΔT, °C', fraction=0.04)

    plt.suptitle(
        f'Потепление мерзлоты: ΔMAGT относительно baseline mean(2018-2024)',
        fontsize=14, y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / 'p3_ensemble6_delta_vs_baseline.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nСохранено: {out.name}")

    # ===== Baseline reference map =====
    print(f"\n[4] Карта baseline reference...")
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.pcolormesh(lons, lats, baseline,
                        cmap='RdBu_r', vmin=-12, vmax=2, shading='auto')
    ax.set_title(
        f'Reference baseline: mean MAGT 2018-2024 (7 лет)\n'
        f'Median {baseline_median:+.2f}°C — используется как точка отсчёта прогноза',
        fontsize=12)
    ax.set_xlabel('Долгота, °E')
    ax.set_ylabel('Широта, °N')
    plt.colorbar(im, ax=ax, label='MAGT, °C', fraction=0.04)
    plt.tight_layout()
    out2 = FIGURES_DIR / 'p3_ensemble6_baseline_map.png'
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out2.name}")

    # ===== Sanity check =====
    print(f"\n=== ΔMAGT vs baseline mean(2018-2024) ===")
    for year, idx in zip(years_to_show, indices):
        delta = ens_mean[idx] - baseline
        mean_d = float(np.nanmean(delta))
        median_d = float(np.nanmedian(delta))
        pos_pct = float(np.nanmean(delta > 0) * 100)
        print(f"  {year}: mean Δ = {mean_d:+.2f}°C, median = {median_d:+.2f}°C, "
              f"потепление в {pos_pct:.1f}% пикселей")

    # Сравнение с предыдущей версией (vs real 2025)
    print(f"\n=== Сравнение методов baseline ===")
    print(f"  Baseline 1 (real 2025):       MAGT = -3.03°C (отдельный outlier год)")
    print(f"  Baseline 2 (mean 2018-2024):  MAGT = {baseline_median:+.2f}°C (7-year reference)")
    print(f"  Разница: {baseline_median - (-3.03):+.2f}°C")

    print(f"\nАртефакты:")
    print(f"  {out.relative_to(BASE_DIR)}")
    print(f"  {out2.relative_to(BASE_DIR)}")

    import subprocess
    subprocess.run(['open', str(out), str(out2)])


if __name__ == '__main__':
    main()
