"""
Расширение ensemble Model A до 7 моделей.

ЭТАП 1: дообучение 4 новых моделей (seeds: 7, 99, 777, 2024)
ЭТАП 2: загрузка всех 7 моделей и финальный ensemble inference на 2026-2035

Существующие модели:
  models/convlstm_ttop_p3_model_A_seed_{42,123,456}.pt

Новые модели (создаст этот скрипт):
  models/convlstm_ttop_p3_model_A_seed_{7,99,777,2024}.pt

Выход:
  results/metrics/p3_model_A_ensemble7.csv         (val_rmse 7 моделей)
  results/metrics/p3_projection_ensemble7.csv     (projection 2026-2035, 7 моделей)
"""

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ablation import train_one_config, features_by_names, load_ablation_winner
from src.model import ConvLSTMNet
from src.data import normalize_features
from src.inference import predict_full_map
from src.landcover import compute_ttop_target

DATA_DIR = BASE_DIR / 'data'
MAPS_DIR = BASE_DIR / 'results' / 'maps'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
MODELS_DIR = BASE_DIR / 'models'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_TABLE = METRICS_DIR / 'ablation_table.csv'

# Все 7 seeds
ALL_SEEDS = [42, 123, 456, 7, 99, 777, 2024]
NEW_SEEDS = [7, 99, 777, 2024]   # эти будем обучать

# Train/Val (как в Model A)
TRAIN_TARGETS = list(range(4, 20))   # 2007..2022
VAL_TARGETS = [20, 21, 22]            # 2023, 2024, 2025

EPOCHS = 80
LR = 1e-4
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 20

P1_SHIFT = 1.985
PROJ_YEARS = list(range(2026, 2036))
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def train_one_seed(seed, X, y, core_idx, winner_feats, device, years):
    """Обучает одну модель Model A с указанным seed."""
    result = train_one_config(
        model_class=ConvLSTMNet,
        X_raw=X,
        y_raw=y,
        feature_indices=core_idx,
        train_targets=TRAIN_TARGETS,
        val_targets=VAL_TARGETS,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=device,
        early_stop_patience=EARLY_STOP_PATIENCE,
        seed=seed,
    )

    model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
    torch.save({
        'state_dict': result['model_state'],
        'feature_names': winner_feats,
        'feature_indices': core_idx,
        'val_rmse': result['val_rmse'],
        'y_mean': result['y_mean'],
        'y_std': result['y_std'],
        'best_epoch': result['best_epoch'],
        'seed': seed,
        'train_years': [int(years[i]) for i in TRAIN_TARGETS],
        'val_years': [int(years[i]) for i in VAL_TARGETS],
    }, model_path)

    return result


def run_inference_for_model(model_path, X_n, y_nm, t_targets, device, delta_t):
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

    # Загружаем общие данные
    print("Загружаем 23-year тензор...")
    tensor = np.load(DATA_DIR / 'tensor_01deg_extended_23y.npz')
    X = tensor['X']
    years = list(tensor['years'])
    print(f"  X: {X.shape}, годы {years[0]}..{years[-1]}")

    y_npz = np.load(DATA_DIR / 'y_new_rk_landcover_extended_23y.npz')
    y = y_npz['y_new']

    winner_name, winner_feats = load_ablation_winner(ABLATION_TABLE)
    core_idx = features_by_names(winner_feats)

    # ============ ЭТАП 1: обучение 4 новых моделей ============
    print(f"\n{'='*60}")
    print(f"=== ЭТАП 1: Обучение 4 новых моделей ===")
    print(f"{'='*60}")
    print(f"Новые seeds: {NEW_SEEDS}")
    print(f"Ожидаемое время: ~{len(NEW_SEEDS) * 8} минут\n")

    train_results = []

    for i, seed in enumerate(NEW_SEEDS):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(NEW_SEEDS)}] Обучение seed={seed}")
        print(f"{'─'*60}")

        # Проверяем не обучена ли уже
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        if model_path.exists():
            print(f"  Модель уже существует, пропускаем")
            continue

        t0 = time.time()
        result = train_one_seed(seed, X, y, core_idx, winner_feats, device, years)
        elapsed = (time.time() - t0) / 60

        print(f"\n  seed={seed}: val_rmse={result['val_rmse']:.3f}°C, "
              f"best_epoch={result['best_epoch']}, time={elapsed:.1f} мин")

        train_results.append({
            'seed': seed,
            'val_rmse': result['val_rmse'],
            'best_epoch': result['best_epoch'],
            'time_min': elapsed,
        })

    # ============ ЭТАП 2: загрузка существующих val_rmse для всех 7 ============
    print(f"\n{'='*60}")
    print(f"=== ЭТАП 2: Сводка val_rmse 7 моделей ===")
    print(f"{'='*60}\n")

    all_val_rmses = []
    for seed in ALL_SEEDS:
        path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        if not path.exists():
            print(f"  seed={seed}: ОШИБКА — файл не найден!")
            continue
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        all_val_rmses.append({
            'seed': seed,
            'val_rmse': ckpt['val_rmse'],
            'best_epoch': ckpt['best_epoch'],
        })

    df_val = pd.DataFrame(all_val_rmses)
    print(df_val.to_string(index=False, float_format='%.4f'))

    val_arr = df_val['val_rmse'].values
    print(f"\n  Mean val_rmse: {val_arr.mean():.3f}°C")
    print(f"  Std val_rmse:  {val_arr.std():.3f}°C")
    print(f"  Min/Max:       {val_arr.min():.3f} / {val_arr.max():.3f}°C")

    df_val.to_csv(METRICS_DIR / 'p3_model_A_ensemble7.csv', index=False, float_format='%.4f')

    # ============ ЭТАП 3: ensemble inference 2026-2035 ============
    print(f"\n{'='*60}")
    print(f"=== ЭТАП 3: Ensemble projection 2026-2035 (7 моделей) ===")
    print(f"{'='*60}\n")

    # Synthetic тензор
    print("Загружаем synthetic тензор...")
    synth = np.load(DATA_DIR / 'tensor_01deg_synthetic_2026_2035.npz')
    X_full = synth['X']
    years_full = list(synth['years'])
    lats = synth['lats']
    lons = synth['lons']

    # Target
    y_old = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old['landcover']
    y_full = compute_ttop_target(X_full, landcover, fdd_idx=7, tdd_idx=8)

    # Normalize (как в Model A — train 2007-2022)
    X_core = X_full[..., core_idx]
    train_idx = list(range(4, 20))
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y_full, train_idx)

    # lh
    delta_t = np.load(DATA_DIR / 'maps_lh_v3.npz')['delta_t']

    t_targets = [years_full.index(yr) for yr in PROJ_YEARS]

    # Inference 7 моделей
    print(f"\nInference 7 моделей на 10 годов:\n")

    all_medians = []  # [(seed, [10 medians])]

    for seed in ALL_SEEDS:
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        if not model_path.exists():
            print(f"  seed={seed}: пропуск (модель не найдена)")
            continue

        t0 = time.time()
        preds, val_rmse, best_epoch = run_inference_for_model(
            model_path, X_n, y_nm, t_targets, device, delta_t
        )
        elapsed = time.time() - t0

        medians = [float(np.nanmedian(preds[i])) for i in range(len(PROJ_YEARS))]
        print(f"  seed={seed:4d} (val_rmse={val_rmse:.3f}, "
              f"best_ep={best_epoch:2d}): "
              f"medians {[f'{m:+.2f}' for m in medians]}, время {elapsed:.1f}c")

        all_medians.append({
            'seed': seed,
            'val_rmse': val_rmse,
            'best_epoch': best_epoch,
            'medians': medians,
        })

    # ============ ФИНАЛЬНАЯ СВОДКА ============
    print(f"\n{'='*60}")
    print(f"=== ФИНАЛЬНАЯ СВОДКА ENSEMBLE 7 МОДЕЛЕЙ ===")
    print(f"{'='*60}\n")

    medians_arr = np.array([d['medians'] for d in all_medians])  # (N_models, 10)
    n_models = len(all_medians)
    print(f"  Моделей в ensemble: {n_models}")

    # По годам
    rows = []
    print(f"\n  ┌──────┬──────────────────────────────────────────┬───────────┐")
    print(f"  │ Year │ Все модели                                │ mean ± std │")
    print(f"  ├──────┼──────────────────────────────────────────┼───────────┤")
    for i, year in enumerate(PROJ_YEARS):
        vals = medians_arr[:, i]
        mean = vals.mean()
        std = vals.std()
        vals_str = ' '.join([f'{v:+.2f}' for v in vals])
        print(f"  │ {year} │ {vals_str}  │ {mean:+.2f}±{std:.2f} │")
        rows.append({
            'year': year,
            'mean': mean,
            'std': std,
            'min': vals.min(),
            'max': vals.max(),
            'range': vals.max() - vals.min(),
            **{f'seed_{ALL_SEEDS[k]}': medians_arr[k, i] for k in range(n_models)},
        })
    print(f"  └──────┴──────────────────────────────────────────┴───────────┘")

    df_proj = pd.DataFrame(rows)
    df_proj.to_csv(METRICS_DIR / 'p3_projection_ensemble7.csv',
                    index=False, float_format='%.4f')

    # Финальные числа
    print(f"\n  2035 mean ± std: {df_proj['mean'].iloc[-1]:+.2f} ± {df_proj['std'].iloc[-1]:.2f}°C")
    print(f"  2035 range:      [{df_proj['min'].iloc[-1]:+.2f}, {df_proj['max'].iloc[-1]:+.2f}]°C")
    print(f"  Spread (max-min): {df_proj['range'].iloc[-1]:.2f}°C")

    # Анализ outliers
    print(f"\n  Анализ распределения medians 2035:")
    medians_2035 = medians_arr[:, -1]
    sorted_idx = np.argsort(medians_2035)
    for idx in sorted_idx:
        seed_v = ALL_SEEDS[idx]
        m = medians_2035[idx]
        be = all_medians[idx]['best_epoch']
        print(f"    seed={seed_v:4d}: 2035 = {m:+.3f}°C (best_epoch={be})")

    # IQR детектор outliers
    q1, q3 = np.percentile(medians_2035, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    print(f"\n  IQR analysis для 2035:")
    print(f"    Q1={q1:+.3f}, Q3={q3:+.3f}, IQR={iqr:.3f}")
    print(f"    Outlier bounds: [{lower:+.3f}, {upper:+.3f}]")
    outliers = [i for i, m in enumerate(medians_2035) if m < lower or m > upper]
    if outliers:
        print(f"    Outliers: {[ALL_SEEDS[i] for i in outliers]}")
    else:
        print(f"    Outliers: НЕТ — все 7 моделей внутри IQR bounds")


if __name__ == '__main__':
    main()
