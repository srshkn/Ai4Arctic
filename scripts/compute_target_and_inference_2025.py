"""
Пересчёт TTOP target для 23 лет (2003-2025) + inference P2 модели на 2025.

Вход:
    data/tensor_01deg_extended_23y.npz
    data/y_new_rk_landcover.npz
    data/y_new_rk_landcover_extended.npz (для sanity check)
    models/convlstm_ttop_extended_21years.pt
    data/maps_lh_v3.npz

Выход:
    data/y_new_rk_landcover_extended_23y.npz
    results/maps/p2_real_2025.npz
"""

import numpy as np
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
MODELS_DIR = BASE_DIR / 'models'

P1_SHIFT = 1.985
TARGET_YEAR = 2025
OUTPUT = MAPS_DIR / 'p2_real_2025.npz'


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # ===== Шаг 1: загрузка 23-year тензора =====
    print("[Шаг 1] Загрузка 23-year тензора...")
    tensor_path = DATA_DIR / 'tensor_01deg_extended_23y.npz'
    tensor = np.load(tensor_path)
    X23 = tensor['X']
    years23 = list(tensor['years'])
    lats = tensor['lats']
    lons = tensor['lons']
    print(f"  X: {X23.shape}")
    print(f"  Years: {years23[0]}..{years23[-1]} ({len(years23)} лет)")

    # ===== Шаг 2: TTOP target =====
    print(f"\n[Шаг 2] Пересчёт TTOP target...")
    y_old_npz = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old_npz['landcover']
    rk_map = y_old_npz['rk_map']

    y23 = compute_ttop_target(X23, landcover, fdd_idx=7, tdd_idx=8)
    print(f"  y23 shape: {y23.shape}")

    print(f"\n  TTOP target по годам (последние):")
    for i, year in enumerate(years23):
        if year >= 2020:
            median = float(np.nanmedian(y23[i]))
            marker = " ← NEW (real 2025)" if year == TARGET_YEAR else ""
            print(f"    {year}: median {median:+.2f}°C{marker}")

    # ===== Шаг 3: sanity check =====
    print(f"\n[Шаг 3] Sanity check vs предыдущий target...")
    prev_target_path = DATA_DIR / 'y_new_rk_landcover_extended.npz'
    if prev_target_path.exists():
        prev = np.load(prev_target_path)
        y_prev = prev['y_new']
        years_prev = list(prev['years'])
        max_diff = 0.0
        for y in years_prev:
            if y in years23:
                i_prev = years_prev.index(y)
                i_new = years23.index(y)
                mask = ~np.isnan(y_prev[i_prev]) & ~np.isnan(y23[i_new])
                if mask.any():
                    diff = np.abs(y_prev[i_prev][mask] - y23[i_new][mask]).max()
                    max_diff = max(max_diff, diff)
        print(f"  Max abs diff на общих годах: {max_diff:.6f}°C")
        if max_diff > 0.01:
            print(f"  ⚠️ ВНИМАНИЕ: расхождение значительное!")
        else:
            print(f"  ✓ Target идентичен предыдущему")
    else:
        print(f"  (пропускаю, предыдущий файл не найден)")

    # ===== Шаг 4: сохранение target =====
    print(f"\n[Шаг 4] Сохранение target...")
    out_target = DATA_DIR / 'y_new_rk_landcover_extended_23y.npz'
    np.savez_compressed(
        out_target,
        y_new=y23.astype(np.float32),
        landcover=landcover,
        rk_map=rk_map,
        years=np.array(years23),
        lats=lats,
        lons=lons,
    )
    print(f"  Сохранён: {out_target.name}")

    # ===== Шаг 5: inference P2 модели на 2025 =====
    print(f"\n[Шаг 5] Inference P2 модели на {TARGET_YEAR}...")
    ckpt_path = MODELS_DIR / 'convlstm_ttop_extended_21years.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    val_rmse = ckpt.get('val_rmse', None)
    val_rmse_str = f'{val_rmse:.3f}°C' if val_rmse is not None else '?'
    print(f"  Checkpoint загружен (val RMSE = {val_rmse_str})")

    # Загружаем модель
    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"  ConvLSTMNet(in_ch=6) в eval режиме")

    # 6 climate_core фичей
    CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']
    core_idx = features_by_names(CORE_6)
    X_core_23 = X23[..., core_idx]
    print(f"  X_core: {X_core_23.shape}")

    # Train idx — те же годы 2007-2021
    train_idx = list(range(4, 19))  # годы 2007..2021
    print(f"  Train idx: {train_idx} ({years23[train_idx[0]]}..{years23[train_idx[-1]]})")

    X_n, _, _, _, _, _, y_nm = normalize_features(X_core_23, y23, train_idx)

    # lh delta_t
    lh = np.load(DATA_DIR / 'maps_lh_v3.npz')
    delta_t = lh['delta_t']

    # Inference на 2025
    t_target = years23.index(TARGET_YEAR)
    print(f"  t_target index: {t_target} (год {years23[t_target]})")
    print(f"  Input window: {years23[t_target-3]}, {years23[t_target-2]}, "
          f"{years23[t_target-1]}, {years23[t_target]}")

    print(f"\n  Запуск inference...")
    t0 = time.time()
    pred_raw = predict_full_map(
        model=model,
        X_norm=X_n,
        y_nan_mask=y_nm,
        t_target=t_target,
        y_mean=ckpt['y_mean'],
        y_std=ckpt['y_std'],
        device=device,
    )
    print(f"  Время: {time.time()-t0:.1f} сек")

    # Применяем lh correction + P1 shift
    pred_lh = pred_raw + delta_t
    pred_final = pred_lh + P1_SHIFT

    print(f"\n  Результат:")
    print(f"    Raw inference:    median {np.nanmedian(pred_raw):+.2f}°C")
    print(f"    + lh correction:  median {np.nanmedian(pred_lh):+.2f}°C")
    print(f"    + P1 shift:       median {np.nanmedian(pred_final):+.2f}°C, "
          f"range [{np.nanmin(pred_final):.2f}, {np.nanmax(pred_final):.2f}]")

    # ===== Шаг 6: сохранение =====
    print(f"\n[Шаг 6] Сохранение карты MAGT {TARGET_YEAR}...")
    np.savez_compressed(
        OUTPUT,
        magt=pred_final.astype(np.float32),
        magt_raw=pred_raw.astype(np.float32),
        year=TARGET_YEAR,
        lats=lats,
        lons=lons,
        p1_shift=P1_SHIFT,
        description=(
            f'P2 model inference on REAL 2025 data from GEE. '
            f'Includes lh correction and P1 pure shift (+{P1_SHIFT}°C).'
        ),
    )
    print(f"  Сохранено: {OUTPUT.name}")

    # ===== Шаг 7: сравнение с прогнозами =====
    print(f"\n[Шаг 7] КЛЮЧЕВОЕ СРАВНЕНИЕ для {TARGET_YEAR}:")
    print(f"  ════════════════════════════════════════════════════")
    real_2025_median = float(np.nanmedian(pred_final))

    # MAGT 2024 для контекста
    magt_history_path = MAPS_DIR / 'p2_yearly_maps.npz'
    if magt_history_path.exists():
        m_hist = np.load(magt_history_path)
        y24_idx = list(m_hist['years']).index(2024)
        m24_median = float(np.nanmedian(m_hist['magt_yearly'][y24_idx]))
        print(f"  MAGT 2024 (real):              {m24_median:+.2f}°C")

    print(f"  MAGT 2025 (REAL — наш новый):  {real_2025_median:+.2f}°C  ⭐")

    if magt_history_path.exists():
        delta_real = real_2025_median - m24_median
        print(f"  Δ 2024 → 2025 (real):          {delta_real:+.2f}°C")

    print(f"  ────────────────────────────────────────────────────")

    # Linear extrapolation
    linear_path = MAPS_DIR / 'p2_extrapolation_2025_2030.npz'
    if linear_path.exists():
        lin = np.load(linear_path)
        idx = list(lin['years_extrap']).index(2025)
        lin_2025_median = float(np.nanmedian(lin['magt_extrapolated'][idx]))
        print(f"  Linear extrapolation 2025:     {lin_2025_median:+.2f}°C")
        print(f"  Diff Real - Linear:            {real_2025_median - lin_2025_median:+.2f}°C")

    # ConvLSTM на synthetic
    synth_path = MAPS_DIR / 'p2_projection_synthetic.npz'
    if synth_path.exists():
        synth = np.load(synth_path)
        idx = list(synth['years']).index(2025)
        synth_2025_median = float(np.nanmedian(synth['magt_yearly'][idx]))
        print(f"  ConvLSTM на synthetic 2025:    {synth_2025_median:+.2f}°C")
        print(f"  Diff Real - Synthetic:         {real_2025_median - synth_2025_median:+.2f}°C")

    print(f"  ════════════════════════════════════════════════════")


if __name__ == '__main__':
    main()
