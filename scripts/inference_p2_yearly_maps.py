"""
Этап 5a: P2 inference на реальных данных 2007–2024.

Строит исторические карты MAGT для склейки с прогнозом 2026–2035
в plot_ensemble6_final.py.

Вход:
    data/tensor_01deg_extended_23y.npz
    data/y_new_rk_landcover_extended_23y.npz
    models/convlstm_ttop_extended_21years.pt
    data/maps_lh_v3.npz

Выход:
    results/maps/p2_yearly_maps.npz
        magt_yearly: (18, 231, 1501) — 2007..2024, lh + P1 shift
        years, lats, lons

Следующий шаг: scripts/inference_p2_2025.py → p2_real_2025.npz
Или сразу: scripts/inference_p2_history.py (оба шага)
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
INFERENCE_YEARS = list(range(2007, 2025))  # 2007..2024 (18 лет)
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']
TRAIN_IDX = list(range(4, 19))  # 2007..2021
OUTPUT = MAPS_DIR / 'p2_yearly_maps.npz'


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

    tensor = np.load(tensor_path)
    X = tensor['X']
    years = list(tensor['years'])
    lats = tensor['lats']
    lons = tensor['lons']

    y = np.load(target_path)['y_new']
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Years in tensor: {years[0]}..{years[-1]}")

    missing = [y for y in INFERENCE_YEARS if y not in years]
    if missing:
        raise ValueError(f"В тензоре нет годов для inference: {missing}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    val_rmse = ckpt.get('val_rmse')
    print(f"Checkpoint: {ckpt_path.name} (val RMSE = {val_rmse:.3f}°C)" if val_rmse else f"Checkpoint: {ckpt_path.name}")

    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    core_idx = features_by_names(CORE_6)
    X_core = X[..., core_idx]
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y, TRAIN_IDX)
    delta_t = np.load(lh_path)['delta_t']

    n_years = len(INFERENCE_YEARS)
    H, W = len(lats), len(lons)
    magt_yearly = np.full((n_years, H, W), np.nan, dtype=np.float32)
    magt_raw = np.full((n_years, H, W), np.nan, dtype=np.float32)

    print(f"\nInference {n_years} годов ({INFERENCE_YEARS[0]}..{INFERENCE_YEARS[-1]})...")
    t0 = time.time()
    for i, year in enumerate(INFERENCE_YEARS):
        t_target = years.index(year)
        pred_raw = predict_full_map(
            model=model,
            X_norm=X_n,
            y_nan_mask=y_nm,
            t_target=t_target,
            y_mean=ckpt['y_mean'],
            y_std=ckpt['y_std'],
            device=device,
        )
        pred_lh = pred_raw + delta_t
        pred_final = pred_lh + P1_SHIFT

        magt_raw[i] = pred_raw.astype(np.float32)
        magt_yearly[i] = pred_final.astype(np.float32)
        print(f"  [{i + 1:>2}/{n_years}] {year}: median {np.nanmedian(pred_final):+.2f}°C")

    print(f"\nВремя: {(time.time() - t0) / 60:.1f} мин")

    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        magt_yearly=magt_yearly,
        magt_raw=magt_raw,
        years=np.array(INFERENCE_YEARS),
        lats=lats,
        lons=lons,
        intercept=P1_SHIFT,
        description='P2 model inference 2007-2024, lh correction + P1 pure shift',
    )
    print(f"\nСохранено: {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} МБ)")

    print("\nТренд медианы MAGT:")
    for i, year in enumerate(INFERENCE_YEARS):
        print(f"  {year}: {np.nanmedian(magt_yearly[i]):+.2f}°C")


if __name__ == '__main__':
    main()
