"""
Метрики качества прогноза.

Содержит:
- all_metrics    : универсальная функция всех метрик (bias, MAE, RMSE, R², Pearson, Spearman, KGE)
- per_zone_metrics: разбивка метрик по зонам мерзлоты
- bias_correction_for_boreholes: коррекция MAGT измерений на временной тренд
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# Границы зон мерзлоты по MAGT (Brown et al., 1998)
PERMAFROST_ZONES = [
    ('Континуальная (<-5°C)',    -np.inf, -5),
    ('Прерывистая (-5..-2°C)',   -5,      -2),
    ('Спорадическая (-2..0°C)',  -2,       0),
    ('Без мерзлоты (>=0°C)',      0,      np.inf),
]


def all_metrics(pred, true, label=''):
    """
    Универсальная функция всех метрик.

    Parameters
    ----------
    pred, true : ndarray
        Массивы предсказаний и истинных значений.
        Могут содержать NaN — они исключаются.
    label : str
        Опциональная метка для печати.

    Returns
    -------
    dict со всеми метриками:
        N, Bias, MAE, MedAE, RMSE, R², Pearson r, Spearman ρ,
        Murphy SS, KGE, std ratio (α), Range ratio
    """
    mask = ~np.isnan(pred) & ~np.isnan(true) & ~np.isinf(pred) & ~np.isinf(true)
    p = pred[mask]
    t = true[mask]
    n = len(p)
    if n < 5:
        return None

    err = p - t
    bias = float(err.mean())
    mae = float(np.abs(err).mean())
    medae = float(np.median(np.abs(err)))
    rmse = float(np.sqrt((err**2).mean()))

    # R² (Murphy SS)
    ss_res = (err**2).sum()
    ss_tot = ((t - t.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else float('nan')

    # Корреляции
    r_p, _ = pearsonr(p, t)
    rho_s, _ = spearmanr(p, t)

    # KGE (Kling-Gupta Efficiency, модифицированная)
    alpha = p.std() / t.std() if t.std() > 0 else float('nan')
    beta_alt = bias / t.std() if t.std() > 0 else float('nan')
    kge = 1 - np.sqrt((r_p - 1)**2 + (alpha - 1)**2 + beta_alt**2)

    # Сжатие диапазона
    range_ratio = (p.max() - p.min()) / (t.max() - t.min()) if t.std() > 0 else float('nan')

    metrics = {
        'N': n,
        'Bias': bias,
        'MAE': mae,
        'MedAE': medae,
        'RMSE': rmse,
        'R²': r2,
        'Pearson r': float(r_p),
        'Spearman ρ': float(rho_s),
        'Murphy SS': r2,
        'KGE': float(kge),
        'std ratio (α)': float(alpha),
        'Range ratio': float(range_ratio),
    }

    if label:
        print(f'\n=== {label} ===')
        for k, v in metrics.items():
            if k == 'N':
                print(f'  {k:<18}: {v:>10,}')
            else:
                print(f'  {k:<18}: {v:>+10.4f}')

    return metrics


def per_zone_metrics(pred, true, zone_classifier=None, zones=PERMAFROST_ZONES):
    """
    Разбивка метрик по зонам мерзлоты.

    Parameters
    ----------
    pred, true : ndarray (H, W) или 1D
        Предсказание и истина.
    zone_classifier : ndarray (H, W), optional
        Если задан — массив, по значениям которого классифицируются зоны.
        Если None — используется true для классификации.
    zones : list[tuple]
        Список (название, low, high) — границы зон по значению.

    Returns
    -------
    DataFrame со строками для каждой зоны и колонками метрик.
    """
    if zone_classifier is None:
        zone_classifier = true

    valid = ~np.isnan(pred) & ~np.isnan(true)
    results = []

    for name, lo, hi in zones:
        zone_mask = (zone_classifier >= lo) & (zone_classifier < hi) & valid
        n = int(zone_mask.sum())
        if n < 10:
            continue

        err = (pred - true)[zone_mask]
        results.append({
            'Зона': name,
            'N': n,
            '% территории': round(100 * n / valid.sum(), 1),
            'Bias °C': round(float(err.mean()), 3),
            'MAE °C': round(float(np.abs(err).mean()), 3),
            'RMSE °C': round(float(np.sqrt((err**2).mean())), 3),
        })

    return pd.DataFrame(results)


def bias_correct_boreholes(df_boreholes, trend=0.085, target_year=2023,
                            magt_col='magt_c', year_col='mid_year'):
    """
    Корректировка измеренной MAGT буров на линейный тренд потепления.

    Буры измерены в 2006-2011, наша модель — для 2023. За эти годы мерзлота
    прогрелась примерно на +1 °C, что нужно учитывать при сравнении.

    Parameters
    ----------
    df_boreholes : DataFrame
        Таблица буров с колонками 'lat', 'lon', magt_col, year_col.
    trend : float
        Тренд потепления в °C/год (по умолчанию 0.085 — собственный yearly анализ).
    target_year : int
        Год, на который коррекция (по умолчанию 2023).

    Returns
    -------
    DataFrame с добавленной колонкой 'magt_adjusted'.
    """
    df = df_boreholes.copy()
    df['years_to_target'] = target_year - df[year_col]
    df['magt_adjusted'] = df[magt_col] + trend * df['years_to_target']
    return df


def extract_borehole_predictions(df_boreholes, pred_map, lons, lats,
                                  out_col='pred'):
    """
    Извлекает значения предсказания на координатах буров.

    Parameters
    ----------
    df_boreholes : DataFrame с колонками 'lat', 'lon'.
    pred_map : ndarray (H, W) — карта предсказаний.
    lons : ndarray (W,) — массив долгот сетки.
    lats : ndarray (H,) — массив широт сетки.
    out_col : str — имя выходной колонки.

    Returns
    -------
    DataFrame с добавленной колонкой `out_col`.
    """
    def find_pixel(lat, lon):
        i = int(np.argmin(np.abs(lats - lat)))
        j = int(np.argmin(np.abs(lons - lon)))
        return i, j

    df = df_boreholes.copy()
    values = []
    for _, row in df.iterrows():
        i, j = find_pixel(row['lat'], row['lon'])
        values.append(pred_map[i, j])
    df[out_col] = values
    return df
