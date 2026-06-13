"""
Анализ ΔMAGT только в зоне реальной мерзлоты (где baseline MAGT < 0°C).

Зачем: на юге России (~50% территории) MAGT уже > 0 — там нет мерзлоты.
Включение этих пикселей в mean Δ разбавляет картину. Корректно фокусироваться
на пикселях с реальной мерзлотой.

Вход:
    results/maps/baseline_2018_2024.npz   (создан plot_delta_vs_baseline.py)
    data/tensor_01deg_synthetic_2026_2035.npz
    models/convlstm_ttop_p3_model_A_seed_{...}.pt

Выход:
    results/figures/p3_ensemble6_delta_permafrost_only.png
    results/figures/p3_ensemble6_permafrost_mask.png
    results/metrics/p3_warming_in_permafrost_zone.csv
"""

import numpy as np
import pandas as pd
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
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
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

    # ===== 1. Загрузка baseline =====
    print("[1] Загружаем baseline mean(2018-2024)...")
    baseline_data = np.load(MAPS_DIR / 'baseline_2018_2024.npz')
    baseline = baseline_data['magt']
    lats = baseline_data['lats']
    lons = baseline_data['lons']
    print(f"  baseline: {baseline.shape}, median {float(np.nanmedian(baseline)):+.2f}°C")

    # ===== 2. Permafrost mask =====
    print("\n[2] Создаём маску мерзлоты (baseline MAGT < 0°C)...")
    permafrost_mask = (baseline < 0) & ~np.isnan(baseline)

    n_total = (~np.isnan(baseline)).sum()
    n_permafrost = permafrost_mask.sum()
    pct = 100 * n_permafrost / n_total
    print(f"  Всего валидных пикселей: {n_total:,}")
    print(f"  Пикселей с мерзлотой:    {n_permafrost:,} ({pct:.1f}%)")
    print(f"  Median MAGT в зоне мерзлоты: {float(np.nanmedian(baseline[permafrost_mask])):+.2f}°C")

    # ===== 3. Карта permafrost mask =====
    print("\n[3] Сохраняем карту маски...")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Показываем маску поверх baseline
    masked_baseline = np.where(permafrost_mask, baseline, np.nan)
    cmap = plt.cm.Blues_r.copy()
    cmap.set_bad('lightgray', alpha=0.5)
    im = ax.pcolormesh(lons, lats, masked_baseline,
                        cmap=cmap, vmin=-12, vmax=0, shading='auto')

    ax.set_title(
        f'Зона анализа: пиксели с baseline MAGT < 0°C (есть мерзлота)\n'
        f'{n_permafrost:,} пикселей ({pct:.1f}% территории), '
        f'median {float(np.nanmedian(baseline[permafrost_mask])):+.2f}°C',
        fontsize=12)
    ax.set_xlabel('Долгота, °E')
    ax.set_ylabel('Широта, °N')
    plt.colorbar(im, ax=ax, label='MAGT baseline, °C', fraction=0.04)

    plt.tight_layout()
    out_mask = FIGURES_DIR / 'p3_ensemble6_permafrost_mask.png'
    plt.savefig(out_mask, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out_mask.name}")

    # ===== 4. Inference 6 моделей =====
    print("\n[4] Inference 6 моделей на 2026-2035...")
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
    ens_mean = all_preds.mean(axis=0)

    # ===== 5. Серия ΔMAGT в зоне мерзлоты =====
    print("\n[5] Строим серию ΔMAGT в зоне мерзлоты...")

    years_to_show = [2028, 2030, 2032, 2034, 2035]
    indices = [PROJ_YEARS.index(y) for y in years_to_show]

    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    cmap_reds_gray = plt.cm.Reds.copy()
    cmap_reds_gray.set_bad('lightgray', alpha=0.4)

    for ax, year, idx in zip(axes, years_to_show, indices):
        delta = ens_mean[idx] - baseline
        # Применяем permafrost mask
        delta_masked = np.where(permafrost_mask, delta, np.nan)

        # Статистика только в зоне мерзлоты
        valid_delta = delta_masked[~np.isnan(delta_masked)]
        mean_d = float(valid_delta.mean())
        median_d = float(np.median(valid_delta))
        pos_pct = float((valid_delta > 0).mean() * 100)

        im = ax.pcolormesh(lons, lats, delta_masked,
                            cmap=cmap_reds_gray, vmin=0, vmax=2.0, shading='auto')
        ax.set_title(
            f'Δ MAGT ({year} − baseline) в зоне мерзлоты\n'
            f'mean = {mean_d:+.2f}°C, median = {median_d:+.2f}°C\n'
            f'потепление в {pos_pct:.1f}% пикселей',
            fontsize=11)
        ax.set_xlabel('Долгота, °E', fontsize=9)
        ax.set_ylabel('Широта, °N', fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, label='ΔT, °C', fraction=0.04)

    plt.suptitle(
        f'Потепление в зоне реальной мерзлоты (baseline MAGT < 0°C) — {pct:.1f}% территории',
        fontsize=14, y=1.02)
    plt.tight_layout()

    out_delta = FIGURES_DIR / 'p3_ensemble6_delta_permafrost_only.png'
    plt.savefig(out_delta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Сохранено: {out_delta.name}")

    # ===== 6. Подробная статистика =====
    print(f"\n=== ΔMAGT в зоне мерзлоты (vs baseline 2018-2024) ===")
    rows = []
    for year, idx in zip(PROJ_YEARS, range(len(PROJ_YEARS))):
        delta = ens_mean[idx] - baseline

        # В зоне мерзлоты
        delta_p = delta[permafrost_mask]
        mean_p = float(delta_p.mean())
        median_p = float(np.median(delta_p))
        pos_pct_p = float((delta_p > 0).mean() * 100)

        # Везде
        delta_all = delta[~np.isnan(delta)]
        mean_all = float(delta_all.mean())
        median_all = float(np.median(delta_all))
        pos_pct_all = float((delta_all > 0).mean() * 100)

        marker = " ⭐" if year in [2030, 2035] else ""
        print(f"  {year}{marker}: ")
        print(f"    Зона мерзлоты: mean Δ = {mean_p:+.2f}°C, median = {median_p:+.2f}°C, "
              f"потепление в {pos_pct_p:.1f}%")
        print(f"    Вся территория: mean Δ = {mean_all:+.2f}°C, median = {median_all:+.2f}°C, "
              f"потепление в {pos_pct_all:.1f}%")

        rows.append({
            'year': year,
            'permafrost_mean_delta': mean_p,
            'permafrost_median_delta': median_p,
            'permafrost_warming_pct': pos_pct_p,
            'all_mean_delta': mean_all,
            'all_median_delta': median_all,
            'all_warming_pct': pos_pct_all,
        })

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / 'p3_warming_in_permafrost_zone.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nCSV: {csv_path.relative_to(BASE_DIR)}")

    # ===== 7. Финальная сводка =====
    print(f"\n{'='*60}")
    print(f"=== СРАВНЕНИЕ: вся территория vs зона мерзлоты ===")
    print(f"{'='*60}")

    final_mean_p = df['permafrost_mean_delta'].iloc[-1]
    final_mean_all = df['all_mean_delta'].iloc[-1]
    print(f"\n  Прогноз 2035 относительно baseline 2018-2024:")
    print(f"\n  Вся территория (включая юг где нет мерзлоты):")
    print(f"    mean Δ = {final_mean_all:+.2f}°C")
    print(f"    median Δ = {df['all_median_delta'].iloc[-1]:+.2f}°C")
    print(f"    потепление в {df['all_warming_pct'].iloc[-1]:.1f}% пикселей")

    print(f"\n  Только зона мерзлоты ({pct:.1f}% территории) ⭐:")
    print(f"    mean Δ = {final_mean_p:+.2f}°C")
    print(f"    median Δ = {df['permafrost_median_delta'].iloc[-1]:+.2f}°C")
    print(f"    потепление в {df['permafrost_warming_pct'].iloc[-1]:.1f}% пикселей")

    print(f"\n  Разница: {final_mean_p - final_mean_all:+.2f}°C")
    print(f"  (зона мерзлоты теплеет в {final_mean_p / max(final_mean_all, 0.01):.1f}x сильнее)")

    print(f"\nАртефакты:")
    print(f"  {out_mask.relative_to(BASE_DIR)}")
    print(f"  {out_delta.relative_to(BASE_DIR)}")
    print(f"  {csv_path.relative_to(BASE_DIR)}")

    import subprocess
    subprocess.run(['open', str(out_mask), str(out_delta)])


if __name__ == '__main__':
    main()
