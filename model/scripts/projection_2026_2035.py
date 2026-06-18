"""
Финальный прогноз MAGT 2026-2035 через synthetic features.

Использует ОБЕ модели для сравнения:
- Model A BEST (val on 2023-2025, val_rmse=0.781°C) — научная модель
- Model B BEST (trained on ALL 2007-2024) — production модель для прогноза

Pipeline:
1. Загрузить 23-year тензор (2003-2025 real data)
2. Pixel-wise линейная экстраполяция 6 climate_core фичей на 2026-2035 (10 лет)
3. Построить synthetic 33-year тензор (real 2003-2025 + synthetic 2026-2035)
4. Inference обеих моделей на synthetic 2026-2035
5. Сравнение результатов
6. Сохранение карт

Вход:
    data/tensor_01deg_extended_23y.npz
    data/y_new_rk_landcover_extended_23y.npz
    models/convlstm_ttop_p3_model_A_BEST.pt
    models/convlstm_ttop_p3_model_B_BEST.pt
    data/maps_lh_v3.npz

Выход:
    data/tensor_01deg_synthetic_2026_2035.npz       (synthetic features)
    results/maps/p3_projection_2026_2035_modelA.npz (прогноз Model A)
    results/maps/p3_projection_2026_2035_modelB.npz (прогноз Model B)
    results/metrics/p3_projection_comparison.csv     (сравнение)
"""

import numpy as np
import pandas as pd
import torch
import sys
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
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
MODELS_DIR = BASE_DIR / 'models'

P1_SHIFT = 1.985
PROJ_YEARS = list(range(2026, 2036))   # 2026-2035 (10 лет)
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_inference_for_model(model_path, X_n_full, y_nm_full, t_targets, ckpt, device, lh_delta_t):
    """Прогон одной модели по списку target years."""
    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    n_proj = len(t_targets)
    H, W = X_n_full.shape[1], X_n_full.shape[2]
    predictions = np.full((n_proj, H, W), np.nan, dtype=np.float32)

    for i, t_target in enumerate(t_targets):
        pred_raw = predict_full_map(
            model=model,
            X_norm=X_n_full,
            y_nan_mask=y_nm_full,
            t_target=t_target,
            y_mean=ckpt['y_mean'],
            y_std=ckpt['y_std'],
            device=device,
        )
        pred_lh = pred_raw + lh_delta_t
        pred_final = pred_lh + P1_SHIFT
        predictions[i] = pred_final.astype(np.float32)

    return predictions


def main():
    device = get_device()
    print(f"Device: {device}\n")

    # ===== Шаг 1: загружаем 23-year тензор =====
    print("[Шаг 1] Загружаем 23-year тензор...")
    tensor = np.load(DATA_DIR / 'tensor_01deg_extended_23y.npz')
    X23 = tensor['X']
    years23 = list(tensor['years'])
    lats = tensor['lats']
    lons = tensor['lons']
    feature_names = list(tensor['feature_names'])
    print(f"  X: {X23.shape}, годы {years23[0]}..{years23[-1]}")

    # ===== Шаг 2: экстраполяция 6 climate_core фичей =====
    print(f"\n[Шаг 2] Pixel-wise экстраполяция 6 climate_core фичей на {PROJ_YEARS[0]}..{PROJ_YEARS[-1]}...")

    # Fit на 2007-2025 (19 лет, max доступные)
    fit_start_idx = years23.index(2007)
    fit_years = np.array(years23[fit_start_idx:])  # 2007..2025
    X_fit = X23[fit_start_idx:]
    T_fit, H, W, F = X_fit.shape
    print(f"  Fit: {fit_years[0]}..{fit_years[-1]} ({T_fit} лет)")

    core_idx = features_by_names(CORE_6)
    static_idx = [i for i in range(F) if i not in core_idx]
    print(f"  Climate (экстраполируем): {CORE_6} → indices {core_idx}")
    print(f"  Static (берём 2025=last): {len(static_idx)} фичей")

    n_proj = len(PROJ_YEARS)
    X_proj = np.zeros((n_proj, H, W, F), dtype=np.float32)

    # Static features: копия с последнего года (2025)
    for i in range(n_proj):
        X_proj[i, :, :, static_idx] = X23[-1, :, :, static_idx]

    # Linear extrapolation 6 climate features
    x = fit_years.astype(np.float64)
    x_mean = x.mean()
    dx = x - x_mean
    var_x = (dx ** 2).sum()

    print(f"\n  Slopes climate-фичей (median per pixel):")
    t0 = time.time()
    for feat_i, feat_name in zip(core_idx, CORE_6):
        y_fit = X_fit[..., feat_i].astype(np.float64)

        valid_mask = ~np.isnan(y_fit).any(axis=0)
        y_flat = y_fit.reshape(T_fit, -1)
        valid_flat = valid_mask.flatten()
        y_valid = y_flat[:, valid_flat]

        y_mean_per_pixel = y_valid.mean(axis=0)
        dy = y_valid - y_mean_per_pixel[np.newaxis, :]
        slope_valid = (dx[:, np.newaxis] * dy).sum(axis=0) / var_x
        intercept_valid = y_mean_per_pixel - slope_valid * x_mean

        for i, year in enumerate(PROJ_YEARS):
            y_year_valid = slope_valid * year + intercept_valid
            X_proj_year = np.full((H, W), np.nan, dtype=np.float32)
            X_proj_year.flat[valid_flat] = y_year_valid.astype(np.float32)
            X_proj[i, :, :, feat_i] = X_proj_year

        median_slope = np.nanmedian(slope_valid)
        median_2025 = np.nanmedian(X23[-1, :, :, feat_i])
        median_2035 = np.nanmedian(X_proj[-1, :, :, feat_i])
        print(f"    {feat_name:14s}: slope {median_slope:+.4f}/год, "
              f"2025 → 2035: {median_2025:+.2f} → {median_2035:+.2f}")
    print(f"  Время: {time.time()-t0:.1f} сек")

    # ===== Шаг 3: собираем 33-year synthetic тензор =====
    print(f"\n[Шаг 3] Собираем 33-year тензор...")
    X_full = np.concatenate([X23, X_proj], axis=0)
    years_full = years23 + [int(y) for y in PROJ_YEARS]
    print(f"  X_full: {X_full.shape}, годы {years_full[0]}..{years_full[-1]}")

    out_tensor = DATA_DIR / 'tensor_01deg_synthetic_2026_2035.npz'
    np.savez_compressed(
        out_tensor,
        X=X_full.astype(np.float32),
        years=np.array(years_full),
        lats=lats, lons=lons,
        feature_names=np.array(feature_names),
        description='33-year tensor: real 2003-2025 + synthetic 2026-2035',
    )
    print(f"  Сохранён: {out_tensor.name} ({out_tensor.stat().st_size/1e6:.0f} МБ)")

    # ===== Шаг 4: пересчёт TTOP target =====
    print(f"\n[Шаг 4] Пересчёт TTOP target...")
    y_old = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old['landcover']
    y_full = compute_ttop_target(X_full, landcover, fdd_idx=7, tdd_idx=8)
    print(f"  y_full: {y_full.shape}")

    # ===== Шаг 5: подготовка к inference =====
    print(f"\n[Шаг 5] Подготовка к inference...")
    X_core_full = X_full[..., core_idx]  # (33, H, W, 6)

    # Train idx такой же как при обучении (P3 Model A): 2007..2022 (индексы 4..19)
    # ВАЖНО: используем СТАТИСТИКИ ОДНОЙ ИЗ МОДЕЛЕЙ для нормализации
    # ckpt содержит y_mean, y_std из train
    train_idx_A = list(range(4, 20))   # 2007..2022 для Model A

    X_n_A, _, _, _, _, _, y_nm_A = normalize_features(
        X_core_full, y_full, train_idx_A
    )

    # lh delta_t
    lh = np.load(DATA_DIR / 'maps_lh_v3.npz')
    delta_t = lh['delta_t']

    # Target индексы для прогноза 2026-2035
    t_targets = [years_full.index(y) for y in PROJ_YEARS]
    print(f"  Target indices: {t_targets} (годы {PROJ_YEARS[0]}..{PROJ_YEARS[-1]})")

    # ===== Шаг 6: inference обеих моделей =====
    print(f"\n[Шаг 6] Inference обеих моделей...")

    # Model A
    print(f"\n  --- Model A (val on 2023-2025, val_rmse=0.781°C) ---")
    ckpt_A = torch.load(MODELS_DIR / 'convlstm_ttop_p3_model_A_BEST.pt',
                         map_location=device, weights_only=False)
    pred_A = run_inference_for_model(
        MODELS_DIR / 'convlstm_ttop_p3_model_A_BEST.pt',
        X_n_A, y_nm_A, t_targets, ckpt_A, device, delta_t
    )

    print(f"  Результаты Model A:")
    for i, year in enumerate(PROJ_YEARS):
        m = float(np.nanmedian(pred_A[i]))
        print(f"    {year}: median {m:+.2f}°C")

    # Model B (использует тот же train_idx для нормализации, т.к. в train_one_config
    # нормализация на train tagets — для Model B это были индексы 4..21)
    print(f"\n  --- Model B (trained on ALL 2007-2024) ---")
    ckpt_B = torch.load(MODELS_DIR / 'convlstm_ttop_p3_model_B_BEST.pt',
                         map_location=device, weights_only=False)

    # Перенормализация для Model B (она училась на других train indices)
    train_idx_B = list(range(4, 22))   # 2007..2024 для Model B
    X_n_B, _, _, _, _, _, y_nm_B = normalize_features(
        X_core_full, y_full, train_idx_B
    )

    pred_B = run_inference_for_model(
        MODELS_DIR / 'convlstm_ttop_p3_model_B_BEST.pt',
        X_n_B, y_nm_B, t_targets, ckpt_B, device, delta_t
    )

    print(f"  Результаты Model B:")
    for i, year in enumerate(PROJ_YEARS):
        m = float(np.nanmedian(pred_B[i]))
        print(f"    {year}: median {m:+.2f}°C")

    # ===== Шаг 7: сохранение карт =====
    print(f"\n[Шаг 7] Сохранение карт...")
    out_A = MAPS_DIR / 'p3_projection_2026_2035_modelA.npz'
    np.savez_compressed(
        out_A,
        magt=pred_A.astype(np.float32),
        years=np.array(PROJ_YEARS),
        lats=lats, lons=lons,
        description='P3 Model A projection 2026-2035 (synthetic climate features)',
    )
    print(f"  Сохранено: {out_A.name}")

    out_B = MAPS_DIR / 'p3_projection_2026_2035_modelB.npz'
    np.savez_compressed(
        out_B,
        magt=pred_B.astype(np.float32),
        years=np.array(PROJ_YEARS),
        lats=lats, lons=lons,
        description='P3 Model B projection 2026-2035 (synthetic climate features)',
    )
    print(f"  Сохранено: {out_B.name}")

    # ===== Шаг 8: сравнительная таблица =====
    print(f"\n[Шаг 8] Сравнение моделей:")
    print(f"\n  ┌──────┬──────────┬──────────┬──────────┐")
    print(f"  │ Year │ Model A  │ Model B  │ Diff B-A │")
    print(f"  ├──────┼──────────┼──────────┼──────────┤")

    rows = []
    for i, year in enumerate(PROJ_YEARS):
        m_A = float(np.nanmedian(pred_A[i]))
        m_B = float(np.nanmedian(pred_B[i]))
        diff = m_B - m_A
        sign = "+" if diff >= 0 else ""
        print(f"  │ {year} │  {m_A:+.2f}   │  {m_B:+.2f}   │  {sign}{diff:.2f}   │")
        rows.append({
            'year': year,
            'magt_modelA': m_A,
            'magt_modelB': m_B,
            'diff_B_minus_A': diff,
        })
    print(f"  └──────┴──────────┴──────────┴──────────┘")

    df = pd.DataFrame(rows)
    csv_path = METRICS_DIR / 'p3_projection_2026_2035_comparison.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  CSV: {csv_path.relative_to(BASE_DIR)}")

    # Финальная статистика
    print(f"\n=== ФИНАЛЬНАЯ СВОДКА ===")
    print(f"  2025 (real reference): -3.03°C")
    print(f"\n  2035 прогноз:")
    print(f"    Model A: {df['magt_modelA'].iloc[-1]:+.2f}°C")
    print(f"    Model B: {df['magt_modelB'].iloc[-1]:+.2f}°C")
    print(f"\n  Δ 2025 → 2035 за 10 лет:")
    print(f"    Model A: {df['magt_modelA'].iloc[-1] - (-3.03):+.2f}°C")
    print(f"    Model B: {df['magt_modelB'].iloc[-1] - (-3.03):+.2f}°C")
    print(f"\n  Mean abs diff между моделями: {np.abs(df['diff_B_minus_A']).mean():.3f}°C")
    print(f"  Max diff: {np.abs(df['diff_B_minus_A']).max():.3f}°C")


if __name__ == '__main__':
    main()
