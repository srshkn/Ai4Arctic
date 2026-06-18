"""
Этап 5b: inference P2 модели на реальных данных 2025 года.

Вход:
    data/tensor_01deg_extended_23y.npz
    data/y_new_rk_landcover_extended_23y.npz
    models/convlstm_ttop_extended_21years.pt
    data/maps_lh_v3.npz

Выход:
    results/maps/p2_real_2025.npz

Запуск:
    python3 scripts/inference_p2_2025.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ablation import features_by_names
from src.data import normalize_features
from src.inference import predict_full_map
from src.model import ConvLSTMNet

DATA_DIR = BASE_DIR / 'data'
MAPS_DIR = BASE_DIR / 'results' / 'maps'
MODELS_DIR = BASE_DIR / 'models'

P1_SHIFT = 1.985
TARGET_YEAR = 2025
OUTPUT = MAPS_DIR / 'p2_real_2025.npz'

CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']
TRAIN_IDX = list(range(4, 19))  # 2007..2021 в 23-year тензоре


def print_comparison(pred_final):
    """Сравнение real 2025 с историей и альтернативными прогнозами."""
    print(f"\nКлючевое сравнение для {TARGET_YEAR}:")
    print("  " + "═" * 52)
    real_2025_median = float(np.nanmedian(pred_final))

    magt_history_path = MAPS_DIR / 'p2_yearly_maps.npz'
    if magt_history_path.exists():
        m_hist = np.load(magt_history_path)
        y24_idx = list(m_hist['years']).index(2024)
        m24_median = float(np.nanmedian(m_hist['magt_yearly'][y24_idx]))
        print(f"  MAGT 2024 (real):              {m24_median:+.2f}°C")
        print(f"  MAGT 2025 (REAL):              {real_2025_median:+.2f}°C")
        print(f"  Δ 2024 → 2025 (real):          {real_2025_median - m24_median:+.2f}°C")
    else:
        print(f"  MAGT 2025 (REAL):              {real_2025_median:+.2f}°C")

    print("  " + "─" * 52)

    linear_path = MAPS_DIR / 'p2_extrapolation_2025_2030.npz'
    if linear_path.exists():
        lin = np.load(linear_path)
        idx = list(lin['years_extrap']).index(2025)
        lin_2025_median = float(np.nanmedian(lin['magt_extrapolated'][idx]))
        print(f"  Linear extrapolation 2025:     {lin_2025_median:+.2f}°C")
        print(f"  Diff Real - Linear:            {real_2025_median - lin_2025_median:+.2f}°C")

    synth_path = MAPS_DIR / 'p2_projection_synthetic.npz'
    if synth_path.exists():
        synth = np.load(synth_path)
        idx = list(synth['years']).index(2025)
        synth_2025_median = float(np.nanmedian(synth['magt_yearly'][idx]))
        print(f"  ConvLSTM на synthetic 2025:    {synth_2025_median:+.2f}°C")
        print(f"  Diff Real - Synthetic:         {real_2025_median - synth_2025_median:+.2f}°C")

    print("  " + "═" * 52)


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    tensor_path = DATA_DIR / 'tensor_01deg_extended_23y.npz'
    target_path = DATA_DIR / 'y_new_rk_landcover_extended_23y.npz'
    ckpt_path = MODELS_DIR / 'convlstm_ttop_extended_21years.pt'
    lh_path = DATA_DIR / 'maps_lh_v3.npz'

    for path in (tensor_path, target_path, ckpt_path, lh_path):
        if not path.exists():
            raise FileNotFoundError(f"Не найден: {path}")

    print("Загрузка данных...")
    tensor = np.load(tensor_path)
    X23 = tensor['X']
    years23 = list(tensor['years'])
    lats = tensor['lats']
    lons = tensor['lons']

    y_npz = np.load(target_path)
    y23 = y_npz['y_new']
    print(f"  X: {X23.shape}, y: {y23.shape}")
    print(f"  Years: {years23[0]}..{years23[-1]}")

    if TARGET_YEAR not in years23:
        raise ValueError(f"Год {TARGET_YEAR} отсутствует в тензоре: {years23}")

    print(f"\nInference P2 модели на {TARGET_YEAR}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    val_rmse = ckpt.get('val_rmse')
    val_rmse_str = f'{val_rmse:.3f}°C' if val_rmse is not None else '?'
    print(f"  Checkpoint: {ckpt_path.name} (val RMSE = {val_rmse_str})")

    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    core_idx = features_by_names(CORE_6)
    X_core_23 = X23[..., core_idx]
    print(f"  Train idx: {TRAIN_IDX} ({years23[TRAIN_IDX[0]]}..{years23[TRAIN_IDX[-1]]})")

    X_n, _, _, _, _, _, y_nm = normalize_features(X_core_23, y23, TRAIN_IDX)
    delta_t = np.load(lh_path)['delta_t']

    t_target = years23.index(TARGET_YEAR)
    print(f"  Input window: {years23[t_target - 3]}, {years23[t_target - 2]}, "
          f"{years23[t_target - 1]}, {years23[t_target]}")

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
    print(f"  Время inference: {time.time() - t0:.1f} сек")

    pred_lh = pred_raw + delta_t
    pred_final = pred_lh + P1_SHIFT

    print("\nРезультат:")
    print(f"  Raw inference:    median {np.nanmedian(pred_raw):+.2f}°C")
    print(f"  + lh correction:  median {np.nanmedian(pred_lh):+.2f}°C")
    print(f"  + P1 shift:       median {np.nanmedian(pred_final):+.2f}°C, "
          f"range [{np.nanmin(pred_final):.2f}, {np.nanmax(pred_final):.2f}]")

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        magt=pred_final.astype(np.float32),
        magt_raw=pred_raw.astype(np.float32),
        year=TARGET_YEAR,
        lats=lats,
        lons=lons,
        p1_shift=P1_SHIFT,
        description=(
            f'P2 model inference on REAL {TARGET_YEAR} data from GEE. '
            f'Includes lh correction and P1 pure shift (+{P1_SHIFT}°C).'
        ),
    )
    print(f"\nСохранено: {OUTPUT}")

    print_comparison(pred_final)


if __name__ == '__main__':
    main()
