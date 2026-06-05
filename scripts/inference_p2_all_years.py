"""
P2: Inference модели на 21 годе на всех годах 2007-2023.

Применяет lh correction и pure shift калибровку.
Сохраняет карты для использования в интерактивном слайдере.

Вход:
  models/convlstm_ttop_extended_21years.pt
  data/tensor_01deg_extended.npz
  data/y_new_rk_landcover_extended.npz
  data/maps_lh_v3.npz (delta_t)

Выход:
  results/maps/p2_yearly_maps.npz с ключами:
    - magt_yearly: (17, 231, 1501) — карты с lh + shift для 2007-2023
    - years: (17,) — годы 2007..2023
    - lats, lons
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

P1_SHIFT = 1.985
INFERENCE_YEARS = list(range(2007, 2024))  # 17 годов (input_window=4 → first usable = 2007)

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MAPS_DIR = BASE_DIR / 'results' / 'maps'
MAPS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = MAPS_DIR / 'p2_yearly_maps.npz'


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # Данные
    tensor_ext = np.load(DATA_DIR / 'tensor_01deg_extended.npz')
    X_ext = tensor_ext['X']
    years_ext = list(tensor_ext['years'])
    lats = tensor_ext['lats']
    lons = tensor_ext['lons']

    y_ext = np.load(DATA_DIR / 'y_new_rk_landcover_extended.npz')['y_new']

    CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']
    core_idx = features_by_names(CORE_6)
    X_core = X_ext[..., core_idx]

    # Модель P2
    ckpt = torch.load(MODELS_DIR / 'convlstm_ttop_extended_21years.pt',
                       map_location=device, weights_only=False)
    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Загружена модель P2 (val RMSE = {ckpt.get('val_rmse', '?'):.3f}°C)")

    # Нормализация по train years (используем тот же split что был при обучении)
    train_idx = list(range(19))  # max(train_targets [4..18]) + 1
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y_ext, train_idx)

    # lh delta_t
    lh = np.load(DATA_DIR / 'maps_lh_v3.npz')
    delta_t = lh['delta_t']

    # Inference по каждому году
    n_years = len(INFERENCE_YEARS)
    magt_yearly = np.full((n_years, 231, 1501), np.nan, dtype=np.float32)
    raw_yearly = np.full((n_years, 231, 1501), np.nan, dtype=np.float32)

    print(f"\nInference {n_years} годов (2007..2023)...")
    t0 = time.time()
    for i, year in enumerate(INFERENCE_YEARS):
        t_target = years_ext.index(year)
        pred_raw = predict_full_map(
            model, X_n, y_nm, t_target=t_target,
            y_mean=ckpt['y_mean'], y_std=ckpt['y_std'],
            device=device,
        )
        pred_lh = pred_raw + delta_t
        pred_final = pred_lh + P1_SHIFT

        raw_yearly[i] = pred_raw.astype(np.float32)
        magt_yearly[i] = pred_final.astype(np.float32)

        print(f"  [{i+1:>2}/{n_years}] {year}: median {np.nanmedian(pred_final):+.2f}°C, "
              f"min {np.nanmin(pred_final):.2f}, max {np.nanmax(pred_final):.2f}")

    print(f"\nВремя инференса: {(time.time()-t0)/60:.1f} мин")

    # Сохраняем
    np.savez_compressed(
        OUTPUT,
        magt_yearly=magt_yearly,
        magt_raw=raw_yearly,
        years=np.array(INFERENCE_YEARS),
        lats=lats,
        lons=lons,
        intercept=P1_SHIFT,
        description='P2 model inference for 2007-2023, with lh correction and pure shift',
    )
    print(f"\nСохранено: {OUTPUT}")
    print(f"  размер: {OUTPUT.stat().st_size/1e6:.1f} МБ")

    # Тренд по медианам
    print(f"\n=== Тренд медианы MAGT (warming signal) ===")
    medians = [float(np.nanmedian(magt_yearly[i])) for i in range(n_years)]
    for year, med in zip(INFERENCE_YEARS, medians):
        print(f"  {year}: {med:+.2f}°C")


if __name__ == '__main__':
    main()
