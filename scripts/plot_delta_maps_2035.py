"""
Дополнительная визуализация: серия карт ΔMAGT относительно 2025.

Зачем: основные карты show абсолютные значения MAGT в диапазоне -12..+2°C,
где сдвиг 0.5-1°C визуально незаметен. Эта картинка показывает
ИЗМЕНЕНИЕ относительно 2025 с узкой шкалой 0..+2°C — потепление
будет хорошо видно красным.

Вход:
    results/maps/p2_real_2025.npz
    results/maps/p3_projection_2026_2035_modelA.npz  (или ensemble mean)

Выход:
    results/figures/p3_ensemble6_delta_maps.png
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

    # Загрузка
    real_2025_map = np.load(MAPS_DIR / 'p2_real_2025.npz')['magt']
    print(f"Real 2025: {real_2025_map.shape}, median {float(np.nanmedian(real_2025_map)):+.2f}°C")

    # Inference 6 моделей для ensemble mean
    print(f"\nInference 6 моделей...")
    synth = np.load(DATA_DIR / 'tensor_01deg_synthetic_2026_2035.npz')
    X_full = synth['X']
    years_full = list(synth['years'])
    lats = synth['lats']
    lons = synth['lons']

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
    print(f"\nEnsemble mean: {ens_mean.shape}")

    # ===== Дельта-карты =====
    print(f"\nСтроим серию ΔMAGT относительно 2025...")

    years_to_show = [2028, 2030, 2032, 2034, 2035]
    indices = [PROJ_YEARS.index(y) for y in years_to_show]

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    for ax, year, idx in zip(axes, years_to_show, indices):
        delta = ens_mean[idx] - real_2025_map
        mean_delta = float(np.nanmean(delta))
        median_delta = float(np.nanmedian(delta))

        im = ax.pcolormesh(lons, lats, delta,
                            cmap='Reds', vmin=0, vmax=2.0, shading='auto')
        ax.set_title(
            f'Δ MAGT ({year} − 2025)\n'
            f'mean Δ = {mean_delta:+.2f}°C, median = {median_delta:+.2f}°C',
            fontsize=11)
        ax.set_xlabel('Долгота, °E', fontsize=9)
        ax.set_ylabel('Широта, °N', fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, label='ΔT, °C', fraction=0.04)

    plt.suptitle(
        'Потепление мерзлоты: ΔMAGT относительно 2025 (узкая шкала для наглядности)',
        fontsize=14, y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / 'p3_ensemble6_delta_maps.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nСохранено: {out.name}")

    # ===== Sanity check =====
    print(f"\n=== ΔMAGT статистика (mean per pixel) ===")
    for year, idx in zip(years_to_show, indices):
        delta = ens_mean[idx] - real_2025_map
        mean_d = float(np.nanmean(delta))
        median_d = float(np.nanmedian(delta))
        max_d = float(np.nanmax(delta))
        min_d = float(np.nanmin(delta))
        pos_pct = float(np.nanmean(delta > 0) * 100)
        print(f"  {year}: mean Δ = {mean_d:+.2f}°C, median = {median_d:+.2f}°C, "
              f"range [{min_d:+.1f}, {max_d:+.1f}], потепление в {pos_pct:.1f}% пикселей")

    import subprocess
    subprocess.run(['open', str(out)])


if __name__ == '__main__':
    main()
