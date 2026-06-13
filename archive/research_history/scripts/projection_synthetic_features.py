"""
P2-projection через synthetic features.

В отличие от прямой линейной экстраполяции MAGT, здесь мы:
1. Экстраполируем динамические фичи из P2-модели (как в ablation winner)
   pixel-wise линейно на 2025-2030
2. Создаём synthetic 22-year тензор (реальные 2003-2024 + synthetic 2025-2030)
3. Подаём в P2 ConvLSTM, которая делает inference
4. Получаем MAGT через нашу модель, а не через эмпирическую формулу

Это методически сильнее чем direct extrapolation MAGT:
- Модель учитывает физическую связь features → MAGT нелинейно
- Если slope features близок к slope MAGT — мы валидируем линейность
- Если расходятся — есть нелинейный эффект в данных

Вход:
  data/tensor_01deg_extended_22y.npz       (X 22 года 2003-2024)
  data/y_new_rk_landcover_extended_22y.npz (target 22 года)
  models/convlstm_ttop_extended_21years.pt (P2 модель)
  data/maps_lh_v3.npz                       (delta_t для lh correction)

Выход:
  data/tensor_01deg_extended_28y_synthetic.npz   (X 28 лет 2003-2030)
  data/y_new_rk_landcover_extended_28y_synthetic.npz (target 28 лет)
  results/maps/p2_projection_synthetic.npz       (карты 2025-2030 через ConvLSTM)
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
from src.ablation import features_by_names, features_from_checkpoint
from src.landcover import compute_ttop_target

DATA_DIR = BASE_DIR / 'data'
MAPS_DIR = BASE_DIR / 'results' / 'maps'
MODELS_DIR = BASE_DIR / 'models'

P1_SHIFT = 1.985
PROJ_YEARS = list(range(2025, 2031))   # 2025-2030

OUTPUT = MAPS_DIR / 'p2_projection_synthetic.npz'


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # ========== Шаг 1: загружаем существующий 22-year тензор ==========
    print("[Шаг 1] Загружаем 22-year тензор (2003-2024)...")
    tensor_22 = np.load(DATA_DIR / 'tensor_01deg_extended_22y.npz')
    X_22 = tensor_22['X']                    # (22, H, W, 20)
    years_22 = list(tensor_22['years'])      # 2003..2024
    lats = tensor_22['lats']
    lons = tensor_22['lons']
    feature_names = list(tensor_22['feature_names'])
    print(f"  X: {X_22.shape}, годы {years_22[0]}..{years_22[-1]}")

    # ========== Шаг 2: pixel-wise extrapolation КЛИМАТИЧЕСКИХ фичей ==========
    print(f"\n[Шаг 2] Экстраполяция климатических фичей на {PROJ_YEARS[0]}..{PROJ_YEARS[-1]}...")

    # Используем годы 2007-2024 для fit (как и в основной экстраполяции MAGT)
    # input window=4 → первые 4 года в input, target начинается с index=4 (year 2007)
    fit_start_idx = years_22.index(2007)
    fit_years = np.array(years_22[fit_start_idx:])  # 2007..2024 (18 лет)
    X_fit = X_22[fit_start_idx:]                     # (18, H, W, 20)

    T_fit, H, W, F = X_fit.shape
    print(f"  Fit: годы {fit_years[0]}..{fit_years[-1]} ({T_fit} лет)")

    ckpt = torch.load(MODELS_DIR / 'convlstm_ttop_extended_21years.pt',
                       map_location='cpu', weights_only=False)
    model_feats = features_from_checkpoint(ckpt)
    core_idx = features_by_names(model_feats)
    print(f"  Экстраполируем {len(model_feats)} фичей из P2 чекпоинта: {model_feats}")

    # Подготовим extension tensor (6 годов 2025-2030, все 20 фичей, статичные = последний год)
    n_proj = len(PROJ_YEARS)
    X_proj = np.zeros((n_proj, H, W, F), dtype=np.float32)

    # Статичные фичи (рельеф, почва, MAP) — берём из последнего года
    # Динамичные климатические фичи — экстраполируем
    static_idx = [i for i in range(F) if i not in core_idx]
    print(f"  Статичные фичи (берём 2024 = last): {len(static_idx)} штук")

    for i, year in enumerate(PROJ_YEARS):
        X_proj[i, :, :, static_idx] = X_22[-1, :, :, static_idx]   # копия 2024

    # Линейная экстраполяция 6 climate фичей
    x = fit_years.astype(np.float64)
    x_mean = x.mean()
    dx = x - x_mean
    var_x = (dx ** 2).sum()

    print(f"\n  Pixel-wise linear fit для каждой климат-фичи:")
    t0 = time.time()
    for feat_i, feat_name in zip(core_idx, model_feats):
        # y_fit: (T_fit, H, W) — значения этой фичи за 2007-2024
        y_fit = X_fit[..., feat_i].astype(np.float64)

        valid_mask = ~np.isnan(y_fit).any(axis=0)
        y_flat = y_fit.reshape(T_fit, -1)
        valid_flat = valid_mask.flatten()
        y_valid = y_flat[:, valid_flat]

        y_mean_per_pixel = y_valid.mean(axis=0)
        dy = y_valid - y_mean_per_pixel[np.newaxis, :]
        slope_valid = (dx[:, np.newaxis] * dy).sum(axis=0) / var_x
        intercept_valid = y_mean_per_pixel - slope_valid * x_mean

        # Заполняем 6 годов проекции
        for i, year in enumerate(PROJ_YEARS):
            y_year_valid = slope_valid * year + intercept_valid
            X_proj_year = np.full((H, W), np.nan, dtype=np.float32)
            X_proj_year.flat[valid_flat] = y_year_valid.astype(np.float32)
            X_proj[i, :, :, feat_i] = X_proj_year

        median_slope = np.median(slope_valid)
        median_2024 = np.nanmedian(X_22[-1, :, :, feat_i])
        median_2030 = np.nanmedian(X_proj[-1, :, :, feat_i])
        print(f"    {feat_name:14s}: median slope {median_slope:+.4f}/год, "
              f"2024={median_2024:+.2f} → 2030={median_2030:+.2f}")
    print(f"  Время: {time.time()-t0:.1f} сек")

    # ========== Шаг 3: объединяем real + synthetic в 28-year тензор ==========
    print(f"\n[Шаг 3] Собираем 28-year synthetic тензор...")
    X_28 = np.concatenate([X_22, X_proj], axis=0)
    years_28 = years_22 + [int(y) for y in PROJ_YEARS]
    print(f"  X_28: {X_28.shape}, годы {years_28[0]}..{years_28[-1]}")

    # Сохраняем тензор для воспроизводимости
    out_tensor = DATA_DIR / 'tensor_01deg_extended_28y_synthetic.npz'
    np.savez_compressed(
        out_tensor,
        X=X_28.astype(np.float32),
        years=np.array(years_28),
        lats=lats, lons=lons,
        feature_names=np.array(feature_names),
        description='28-year tensor: real 2003-2024 + synthetic 2025-2030 (linear extrapolation of climate features)',
    )
    print(f"  Сохранён: {out_tensor.name} ({out_tensor.stat().st_size/1e6:.0f} МБ)")

    # ========== Шаг 4: пересчёт target через src.landcover ==========
    print(f"\n[Шаг 4] Пересчёт TTOP target для 28 лет...")
    y_old_npz = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old_npz['landcover']
    rk_map = y_old_npz['rk_map']

    y_28 = compute_ttop_target(X_28, landcover, fdd_idx=7, tdd_idx=8)
    print(f"  y_28: {y_28.shape}")

    print(f"  TTOP target по годам (verify synthetic):")
    for i, year in enumerate(years_28):
        median = float(np.nanmedian(y_28[i]))
        marker = " [synthetic]" if year >= 2025 else ""
        if year == 2024 or year >= 2025:
            print(f"    {year}: median {median:+.2f}°C{marker}")

    out_target = DATA_DIR / 'y_new_rk_landcover_extended_28y_synthetic.npz'
    np.savez_compressed(
        out_target,
        y_new=y_28.astype(np.float32),
        landcover=landcover, rk_map=rk_map,
        years=np.array(years_28),
        lats=lats, lons=lons,
    )
    print(f"  Сохранён: {out_target.name}")

    # ========== Шаг 5: Inference P2 модели на synthetic годы ==========
    print(f"\n[Шаг 5] Inference P2 модели на synthetic 2025-2030...")

    ckpt = torch.load(MODELS_DIR / 'convlstm_ttop_extended_21years.pt',
                       map_location=device, weights_only=False)
    model_feats = features_from_checkpoint(ckpt)
    core_idx = features_by_names(model_feats)
    model = ConvLSTMNet(in_ch=len(model_feats)).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"  Модель загружена (val RMSE = {ckpt.get('val_rmse', '?'):.3f}°C, "
          f"{len(model_feats)} фичей)")

    X_core_28 = X_28[..., core_idx]

    # Нормализация по train_idx (как в P2: range(19) = годы 2003..2021)
    train_idx = list(range(19))
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core_28, y_28, train_idx)

    # lh delta_t
    lh = np.load(DATA_DIR / 'maps_lh_v3.npz')
    delta_t = lh['delta_t']

    # Inference на каждый synthetic год (2025-2030)
    raw_proj = np.full((n_proj, H, W), np.nan, dtype=np.float32)
    final_proj = np.full((n_proj, H, W), np.nan, dtype=np.float32)

    print(f"\n  Inference {n_proj} synthetic годов:")
    for i, year in enumerate(PROJ_YEARS):
        t_target = years_28.index(year)
        pred_raw = predict_full_map(
            model, X_n, y_nm, t_target=t_target,
            y_mean=ckpt['y_mean'], y_std=ckpt['y_std'],
            device=device,
        )
        pred_lh = pred_raw + delta_t
        pred_final = pred_lh + P1_SHIFT

        raw_proj[i] = pred_raw.astype(np.float32)
        final_proj[i] = pred_final.astype(np.float32)

        print(f"    {year}: median {np.nanmedian(pred_final):+.2f}°C, "
              f"min {np.nanmin(pred_final):.2f}, max {np.nanmax(pred_final):.2f}")

    # ========== Шаг 6: сохранение и сравнение с линейной экстраполяцией ==========
    print(f"\n[Шаг 6] Сравнение с линейной экстраполяцией MAGT...")

    # Загружаем линейную экстраполяцию для сравнения
    linear_proj = np.load(MAPS_DIR / 'p2_extrapolation_2025_2030.npz')
    magt_linear = linear_proj['magt_extrapolated']

    print(f"\n=== Сравнение: synthetic features ConvLSTM vs linear extrapolation ===")
    print(f"  Год    | ConvLSTM | Linear  | Разница")
    print(f"  -------|----------|---------|--------")
    for i, year in enumerate(PROJ_YEARS):
        m_conv = float(np.nanmedian(final_proj[i]))
        m_lin = float(np.nanmedian(magt_linear[i]))
        diff = m_conv - m_lin
        print(f"  {year}  |  {m_conv:+.2f}  |  {m_lin:+.2f}  |  {diff:+.2f}")

    # Сохраняем
    np.savez_compressed(
        OUTPUT,
        magt_yearly=final_proj,        # (6, H, W) — основной результат
        magt_raw=raw_proj,
        years=np.array(PROJ_YEARS),
        lats=lats, lons=lons,
        intercept=P1_SHIFT,
        description=(
            'P2 model inference on synthetic features 2025-2030. '
            'Climate features linearly extrapolated from 2007-2024 trends, '
            'then fed into trained P2 ConvLSTM. Includes lh correction and pure shift. '
            'NOT a CMIP6 forecast, but model-based projection assuming feature trends continue.'
        ),
    )
    print(f"\nСохранено: {OUTPUT}")
    print(f"  размер: {OUTPUT.stat().st_size/1e6:.1f} МБ")


if __name__ == '__main__':
    main()
