"""
Сравнение P2 (21 год) vs P4 (14 лет) моделей на 33 бурениях.

Делает:
1. Inference обеих моделей на год 2023
2. Применяет pure shift калибровку (intercept=+1.985)
3. Сравнивает с бурениями (magt_adjusted)

Вход:
  models/convlstm_ttop_rk_v2_climate_core_6.pt        (P4)
  models/convlstm_ttop_extended_21years.pt            (P2)
  data/tensor_01deg_extended.npz                       (X для inference)
  data/y_new_rk_landcover_extended.npz                 (target для нормализации)
  data/boreholes/clean_boreholes_33.csv                (наблюдения)
  data/maps_lh_v3.npz                                  (delta_t для lh correction)

Выход:
  results/metrics/p2_vs_p4_boreholes_comparison.json
"""

import json
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.model import ConvLSTMNet
from src.data import normalize_features
from src.inference import predict_full_map
from src.ablation import features_by_names

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)

P1_SHIFT = 1.985


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # Данные
    tensor_ext = np.load(DATA_DIR / 'tensor_01deg_extended.npz')
    X_ext = tensor_ext['X']
    years_ext = list(tensor_ext['years'])
    lats = tensor_ext['lats']
    lons = tensor_ext['lons']

    y_ext = np.load(DATA_DIR / 'y_new_rk_landcover_extended.npz')['y_new']

    # Делаем X-тензор для P4 (только 14 лет: 2010-2023)
    p4_years_idx = [years_ext.index(y) for y in range(2010, 2024)]
    X_p4 = X_ext[p4_years_idx]
    y_p4 = y_ext[p4_years_idx]

    # Берём 6 фичей climate_core_6
    CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']
    core_idx = features_by_names(CORE_6)

    X_p4_core = X_p4[..., core_idx]
    X_ext_core = X_ext[..., core_idx]

    # Бурения
    boreholes = pd.read_csv(DATA_DIR / 'boreholes' / 'clean_boreholes_33.csv')
    obs = boreholes['magt_adjusted'].values

    def find_pixel(lat, lon):
        return np.argmin(np.abs(lats - lat)), np.argmin(np.abs(lons - lon))

    # ===== P4 (14 лет, как было раньше) =====
    print("\n=== P4 (14 лет, 2010-2023) ===")
    ckpt_p4 = torch.load(MODELS_DIR / 'convlstm_ttop_rk_v2_climate_core_6.pt',
                          map_location=device, weights_only=False)
    model_p4 = ConvLSTMNet(in_ch=6).to(device)
    model_p4.load_state_dict(ckpt_p4['state_dict'])
    model_p4.eval()

    # Нормализация по train year indices в P4 = range(13) = 2010..2022
    X_n_p4, _, _, _, _, _, y_nm_p4 = normalize_features(
        X_p4_core, y_p4, list(range(13))
    )
    pred_p4_raw = predict_full_map(
        model_p4, X_n_p4, y_nm_p4, t_target=13,  # 2023
        y_mean=ckpt_p4['y_mean'], y_std=ckpt_p4['y_std'],
        device=device,
    )

    # Применяем lh + P1 shift
    lh = np.load(DATA_DIR / 'maps_lh_v3.npz')
    delta_t = lh['delta_t']
    pred_p4_lh = pred_p4_raw + delta_t
    pred_p4_final = pred_p4_lh + P1_SHIFT
    print(f"  pred_p4 (raw): median {np.nanmedian(pred_p4_raw):+.2f}, min {np.nanmin(pred_p4_raw):.2f}")
    print(f"  pred_p4 (+lh): median {np.nanmedian(pred_p4_lh):+.2f}")
    print(f"  pred_p4 (+P1): median {np.nanmedian(pred_p4_final):+.2f}, min {np.nanmin(pred_p4_final):.2f}")

    p4_bh = np.array([pred_p4_final[find_pixel(r['lat'], r['lon'])] for _, r in boreholes.iterrows()])

    # ===== P2 (21 год, 2003-2023) =====
    print("\n=== P2 (21 год, 2003-2023) ===")
    ckpt_p2 = torch.load(MODELS_DIR / 'convlstm_ttop_extended_21years.pt',
                          map_location=device, weights_only=False)
    model_p2 = ConvLSTMNet(in_ch=6).to(device)
    model_p2.load_state_dict(ckpt_p2['state_dict'])
    model_p2.eval()

    # train_years у P2 = range(19) (2003..2021 → indices 0..18, max=18, +1=19)
    train_idx_p2 = list(range(19))
    X_n_p2, _, _, _, _, _, y_nm_p2 = normalize_features(
        X_ext_core, y_ext, train_idx_p2
    )
    # 2023 = index 20 в extended
    pred_p2_raw = predict_full_map(
        model_p2, X_n_p2, y_nm_p2, t_target=20,  # 2023
        y_mean=ckpt_p2['y_mean'], y_std=ckpt_p2['y_std'],
        device=device,
    )
    pred_p2_lh = pred_p2_raw + delta_t
    pred_p2_final = pred_p2_lh + P1_SHIFT
    print(f"  pred_p2 (raw): median {np.nanmedian(pred_p2_raw):+.2f}, min {np.nanmin(pred_p2_raw):.2f}")
    print(f"  pred_p2 (+lh): median {np.nanmedian(pred_p2_lh):+.2f}")
    print(f"  pred_p2 (+P1): median {np.nanmedian(pred_p2_final):+.2f}, min {np.nanmin(pred_p2_final):.2f}")

    p2_bh = np.array([pred_p2_final[find_pixel(r['lat'], r['lon'])] for _, r in boreholes.iterrows()])

    # ===== Сравнение =====
    print("\n" + "=" * 70)
    print(f"СРАВНЕНИЕ vs 33 БУРЕНИЯ (year 2023, observed median {np.median(obs):+.2f}°C)")
    print("=" * 70)
    print(f"{'Model':<35} {'RMSE':>8} {'Bias':>10} {'MAE':>8} {'min':>8}")
    print("-" * 70)

    summary = {}
    for name, pred_bh, pred_map in [
        ('P4 (14 лет, 2010-2023)', p4_bh, pred_p4_final),
        ('P2 (21 год, 2003-2023)', p2_bh, pred_p2_final),
    ]:
        valid = ~np.isnan(pred_bh)
        err = pred_bh[valid] - obs[valid]
        rmse = float(np.sqrt((err**2).mean()))
        bias = float(err.mean())
        mae = float(np.abs(err).mean())
        print(f"{name:<35} {rmse:>8.3f} {bias:>+10.3f} {mae:>8.3f} {float(np.nanmin(pred_map)):>8.2f}")
        summary[name] = {'rmse': rmse, 'bias': bias, 'mae': mae,
                          'min_pred': float(np.nanmin(pred_map))}

    # Сохраняем
    with open(METRICS_DIR / 'p2_vs_p4_boreholes_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nСохранено: {METRICS_DIR / 'p2_vs_p4_boreholes_comparison.json'}")


if __name__ == '__main__':
    main()
