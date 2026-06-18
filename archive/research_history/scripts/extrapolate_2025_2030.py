"""
P2-projection: пиксель-wise линейная экстраполяция тренда MAGT 2025-2030.

Это НЕ прогноз модели ConvLSTM. Это статистическая экстраполяция
наблюдённого тренда 2007-2024. Для каждого пикселя независимо:
  1. Считается линейная регрессия MAGT(year) = slope * year + intercept
  2. Экстраполяция: magt(year) = slope * year + intercept для year in 2025-2030
  3. Confidence interval = ± 2 * residual_std (приблизительно 95% доверия)

Важно для защиты:
- Тренд за 18 лет может не сохраниться следующие 6 лет
- Это экстраполяция, а не прогноз модели
- Неопределённость растёт с расстоянием от 2024

Вход:
  results/maps/p2_yearly_maps.npz (18 годов 2007-2024)

Выход:
  results/maps/p2_extrapolation_2025_2030.npz с ключами:
    - magt_extrapolated: (6, 231, 1501)   для 2025-2030
    - magt_ci_lower:     (6, 231, 1501)   нижняя граница 95% CI
    - magt_ci_upper:     (6, 231, 1501)   верхняя граница 95% CI
    - slope_map:         (231, 1501)      slope тренда по пикселю
    - r2_map:            (231, 1501)      R² по пикселю
    - residual_std_map:  (231, 1501)      σ остатков по пикселю
    - years_extrap:      [2025..2030]
"""

import numpy as np
from pathlib import Path
import sys
import time

BASE_DIR = Path(__file__).resolve().parent.parent
MAPS_DIR = BASE_DIR / 'results' / 'maps'

INPUT_FILE = MAPS_DIR / 'p2_yearly_maps.npz'
OUTPUT = MAPS_DIR / 'p2_extrapolation_2025_2030.npz'

YEARS_EXTRAP = list(range(2025, 2031))   # 2025, 2026, 2027, 2028, 2029, 2030


def pixelwise_linear_fit(magt_history, years_history):
    """
    Векторизованный пиксель-wise linear fit.

    Для каждого пикселя (i, j) считаем slope и intercept через
    least squares formulas. Работает быстро через numpy без scipy loop.

    Parameters
    ----------
    magt_history : (T, H, W) array
        Исторические MAGT карты.
    years_history : (T,) array
        Соответствующие годы.

    Returns
    -------
    slope_map, intercept_map, r2_map, residual_std_map : (H, W) arrays
    """
    T, H, W = magt_history.shape
    x = years_history.astype(np.float64)
    x_mean = x.mean()
    dx = x - x_mean

    # Векторизованный fit: slope, intercept для каждого пикселя
    # Формула: slope = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
    #          intercept = y_mean - slope * x_mean

    # Маска валидных пикселей: пиксели где есть данные во ВСЕ годы
    valid_mask = ~np.isnan(magt_history).any(axis=0)
    print(f"  Валидных пикселей: {valid_mask.sum():,} ({100*valid_mask.mean():.1f}%)")

    # Разворачиваем (T, H, W) → (T, N) где N — число валидных пикселей
    y_flat = magt_history.reshape(T, -1)             # (T, H*W)
    valid_flat = valid_mask.flatten()                 # (H*W,)
    y_valid = y_flat[:, valid_flat]                   # (T, N_valid)

    # Centred-y
    y_mean_per_pixel = y_valid.mean(axis=0)           # (N_valid,)
    dy = y_valid - y_mean_per_pixel[np.newaxis, :]    # (T, N_valid)

    # Slope и intercept
    cov_xy = (dx[:, np.newaxis] * dy).sum(axis=0)     # (N_valid,)
    var_x = (dx ** 2).sum()                            # скаляр
    slope_valid = cov_xy / var_x                       # (N_valid,)
    intercept_valid = y_mean_per_pixel - slope_valid * x_mean  # (N_valid,)

    # R²: 1 - SS_res / SS_tot
    y_pred = slope_valid[np.newaxis, :] * x[:, np.newaxis] + intercept_valid[np.newaxis, :]
    ss_res = ((y_valid - y_pred) ** 2).sum(axis=0)
    ss_tot = (dy ** 2).sum(axis=0)
    r2_valid = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)

    # Residual std (несмещённая оценка)
    df = T - 2  # degrees of freedom
    residual_std_valid = np.sqrt(ss_res / df)

    # Возвращаем в форму (H, W) с NaN на невалидных пикселях
    slope_map = np.full((H, W), np.nan, dtype=np.float32)
    intercept_map = np.full((H, W), np.nan, dtype=np.float32)
    r2_map = np.full((H, W), np.nan, dtype=np.float32)
    residual_std_map = np.full((H, W), np.nan, dtype=np.float32)

    slope_map.flat[valid_flat] = slope_valid
    intercept_map.flat[valid_flat] = intercept_valid
    r2_map.flat[valid_flat] = r2_valid
    residual_std_map.flat[valid_flat] = residual_std_valid

    return slope_map, intercept_map, r2_map, residual_std_map


def main():
    print("=== Pixel-wise линейная экстраполяция 2025-2030 ===\n")

    # Загружаем 18 лет реальных карт
    print(f"Загружаем {INPUT_FILE.name}...")
    data = np.load(INPUT_FILE)
    magt = data['magt_yearly']
    years = data['years'].astype(np.int32)
    lats = data['lats']
    lons = data['lons']
    print(f"  Карт: {magt.shape}")
    print(f"  Годы: {years[0]}..{years[-1]} ({len(years)} лет)")

    # Pixel-wise linear fit
    print(f"\nPixel-wise linear fit...")
    t0 = time.time()
    slope_map, intercept_map, r2_map, residual_std_map = pixelwise_linear_fit(magt, years)
    print(f"  Время: {time.time()-t0:.1f} сек")

    # Статистика по slope
    valid_slope = slope_map[~np.isnan(slope_map)]
    print(f"\n  Slope distribution:")
    print(f"    median: {np.median(valid_slope):+.4f}°C/год")
    print(f"    mean:   {np.mean(valid_slope):+.4f}°C/год")
    print(f"    5%:     {np.percentile(valid_slope, 5):+.4f}")
    print(f"    95%:    {np.percentile(valid_slope, 95):+.4f}")

    # R² distribution
    valid_r2 = r2_map[~np.isnan(r2_map)]
    print(f"\n  R² distribution:")
    print(f"    median: {np.median(valid_r2):.3f}")
    print(f"    mean:   {np.mean(valid_r2):.3f}")
    print(f"    % пикселей с R²>0.5: {(valid_r2 > 0.5).mean() * 100:.1f}%")
    print(f"    % пикселей с R²>0.7: {(valid_r2 > 0.7).mean() * 100:.1f}%")

    # Residual std
    valid_resid = residual_std_map[~np.isnan(residual_std_map)]
    print(f"\n  Residual std distribution:")
    print(f"    median: {np.median(valid_resid):.3f}°C")
    print(f"    95%:    {np.percentile(valid_resid, 95):.3f}°C")

    # Экстраполяция на 2025-2030
    print(f"\n=== Экстраполяция {YEARS_EXTRAP[0]}..{YEARS_EXTRAP[-1]} ===")
    n_extrap = len(YEARS_EXTRAP)
    H, W = slope_map.shape

    magt_extrap = np.full((n_extrap, H, W), np.nan, dtype=np.float32)
    ci_lower = np.full((n_extrap, H, W), np.nan, dtype=np.float32)
    ci_upper = np.full((n_extrap, H, W), np.nan, dtype=np.float32)

    # Для CI используем prediction interval, который растёт с временем
    # PI = ± t * residual_std * sqrt(1 + 1/n + (x - x_mean)^2 / sum((x_i - x_mean)^2))
    # При n=18 и t-stat ~2.1 (95%, df=16):
    # для года years_extrap фактор увеличения = 1 + 1/n + (year - x_mean)^2 / var_x
    n_history = len(years)
    x_mean = years.mean()
    var_x = ((years - x_mean) ** 2).sum()
    t_critical = 2.12  # t-distribution, 95% CI, df=16

    for i, year in enumerate(YEARS_EXTRAP):
        # Central estimate
        magt_year = slope_map * year + intercept_map
        magt_extrap[i] = magt_year

        # Prediction interval factor
        pi_factor = np.sqrt(1.0 + 1.0/n_history + ((year - x_mean) ** 2) / var_x)
        pi_width = t_critical * residual_std_map * pi_factor

        ci_lower[i] = magt_year - pi_width
        ci_upper[i] = magt_year + pi_width

        median = float(np.nanmedian(magt_year))
        median_lower = float(np.nanmedian(ci_lower[i]))
        median_upper = float(np.nanmedian(ci_upper[i]))
        median_width = median_upper - median_lower

        print(f"  {year}: median {median:+.2f}°C, "
              f"95% CI [{median_lower:+.2f}, {median_upper:+.2f}] "
              f"(±{median_width/2:.2f})")

    # Сохраняем
    np.savez_compressed(
        OUTPUT,
        magt_extrapolated=magt_extrap.astype(np.float32),
        magt_ci_lower=ci_lower.astype(np.float32),
        magt_ci_upper=ci_upper.astype(np.float32),
        slope_map=slope_map.astype(np.float32),
        r2_map=r2_map.astype(np.float32),
        residual_std_map=residual_std_map.astype(np.float32),
        years_extrap=np.array(YEARS_EXTRAP),
        years_historical=years,
        lats=lats,
        lons=lons,
        description=(
            'Pixel-wise linear extrapolation of MAGT trend 2007-2024 to 2025-2030. '
            'NOT a ConvLSTM forecast; statistical projection only.'
        ),
    )
    print(f"\nСохранено: {OUTPUT}")
    print(f"  размер: {OUTPUT.stat().st_size/1e6:.1f} МБ")

    # Краткий итог
    print(f"\n=== Итог ===")
    median_2024 = float(np.nanmedian(magt[-1]))
    median_2030 = float(np.nanmedian(magt_extrap[-1]))
    print(f"  2024 (real):       median {median_2024:+.2f}°C")
    print(f"  2030 (extrapol.):  median {median_2030:+.2f}°C")
    print(f"  Δ за 6 лет:        {median_2030 - median_2024:+.2f}°C")
    print(f"  Это {(median_2030 - median_2024)/6 * 100:.1f}% относительно медиана 2024")


if __name__ == '__main__':
    main()
