"""
Post-hoc калибровка предсказанной MAGT по in-situ данным бурений.

Используется для устранения cold bias, унаследованного от MODIS LST
(документировано в Westermann 2012, 2017 — остаточный bias ~0.8°C даже после
ERA gap-filling) и от nf-фактора в континентальной Сибири (Obu 2019: занижение
1-6°C, в районе Якутска до 4°C).

Три стратегии калибровки:
    1. Глобальная линейная: y = a*pred + b, одна формула на всю Россию
    2. Per-zone линейная: своя формула для каждой зоны мерзлоты
    3. Piecewise linear по бинам MAGT: своя формула для cold/cool/warm

После калибровки пересчитываются метрики против:
    - in-situ бурений (главная цель — RMSE должен упасть)
    - Obu 2019 (вторично — допускается рост RMSE, это ОЖИДАЕМО)

Использование:
    from src.bias_correction import (
        load_borehole_data,
        fit_correction,
        apply_correction,
        compare_strategies,
    )

    boreholes = load_borehole_data(boreholes_csv)
    paired = pair_predictions_with_boreholes(magt_pred, boreholes, lats, lons)
    results = compare_strategies(paired)
    best_corrector = results['per_zone']['corrector']  # выбрана как лучшая
    magt_corrected = apply_correction(magt_pred, best_corrector, zones=zones_map)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------------------------
# Загрузка in-situ данных
# ---------------------------------------------------------------------------

def load_borehole_data(csv_path: str) -> pd.DataFrame:
    """
    Загружает таблицу 33 чистых бурений.

    Ожидаемые колонки: lat, lon, и одна из {magt_obs, magt_adjusted, magt}.
    Если magt_obs отсутствует, но есть magt_adjusted — переименовывает.
    """
    df = pd.read_csv(csv_path)

    # Обеспечиваем наличие колонки magt_obs (как алиас)
    if 'magt_obs' not in df.columns:
        for candidate in ('magt_adjusted', 'magt', 'MAGT'):
            if candidate in df.columns:
                df['magt_obs'] = df[candidate]
                break

    required = {'lat', 'lon', 'magt_obs'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Колонки {missing} отсутствуют в {csv_path}. "
            f"Имеющиеся: {list(df.columns)}"
        )
    return df


def assign_zone_by_magt(magt: np.ndarray) -> np.ndarray:
    """
    Зональная классификация по MAGT (по Ran 2022, Table 3):
        cold:  MAGT <= -3.0
        cool:  -3.0 < MAGT <= -1.5
        warm:  -1.5 < MAGT <= 0
        no_pf: MAGT > 0

    Возвращает массив int одинаковой формы.
    """
    z = np.full_like(magt, fill_value=3, dtype=np.int8)  # no_pf по умолчанию
    z[magt <= -3.0] = 0  # cold
    z[(magt > -3.0) & (magt <= -1.5)] = 1  # cool
    z[(magt > -1.5) & (magt <= 0.0)] = 2  # warm
    return z


# ---------------------------------------------------------------------------
# Пэйринг предсказаний и in-situ
# ---------------------------------------------------------------------------

def pair_predictions_with_boreholes(
    magt_pred: np.ndarray,
    boreholes: pd.DataFrame,
    lats: np.ndarray,
    lons: np.ndarray,
) -> pd.DataFrame:
    """
    Для каждой точки бурения находит ближайший пиксель в карте magt_pred.

    Args:
        magt_pred: 2D массив (H, W) предсказанной MAGT
        boreholes: DataFrame с колонками lat, lon, magt_obs
        lats: 1D массив широт по оси H (длина H)
        lons: 1D массив долгот по оси W (длина W)

    Returns:
        DataFrame с добавленными колонками magt_pred, i, j, zone_pred
    """
    out = boreholes.copy()
    pred_vals, idx_i, idx_j = [], [], []

    for _, row in out.iterrows():
        i = int(np.argmin(np.abs(lats - row.lat)))
        j = int(np.argmin(np.abs(lons - row.lon)))
        pred_vals.append(float(magt_pred[i, j]))
        idx_i.append(i)
        idx_j.append(j)

    out['magt_pred'] = pred_vals
    out['i'] = idx_i
    out['j'] = idx_j
    out['zone_pred'] = assign_zone_by_magt(out['magt_pred'].values)
    # отбрасываем NaN (могло быть пустое предсказание)
    out = out.dropna(subset=['magt_pred', 'magt_obs']).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Стратегии калибровки
# ---------------------------------------------------------------------------

@dataclass
class GlobalLinearCorrector:
    """Y = slope * pred + intercept, одна формула на всю карту."""
    slope: float = 1.0
    intercept: float = 0.0

    def __call__(self, pred: np.ndarray, **kwargs) -> np.ndarray:
        return self.slope * pred + self.intercept


@dataclass
class PerZoneLinearCorrector:
    """Своя linear regression на зону. Зона определяется по pred."""
    params: dict = field(default_factory=dict)  # zone_id -> (slope, intercept)

    def __call__(self, pred: np.ndarray, zones: Optional[np.ndarray] = None,
                 **kwargs) -> np.ndarray:
        if zones is None:
            zones = assign_zone_by_magt(pred)
        out = pred.copy().astype(np.float32)
        for z, (s, b) in self.params.items():
            mask = (zones == z)
            out[mask] = s * pred[mask] + b
        return out


@dataclass
class PiecewiseLinearCorrector:
    """
    Кусочно-линейная коррекция по бинам исходного предсказания.
    Используется когда per-zone недостаточно (узкие переходные диапазоны).
    """
    breakpoints: np.ndarray = field(default_factory=lambda: np.array([-10, -5, -2, 0]))
    params: list = field(default_factory=list)  # список (slope, intercept) на сегмент

    def __call__(self, pred: np.ndarray, **kwargs) -> np.ndarray:
        out = pred.copy().astype(np.float32)
        bins = np.digitize(pred, self.breakpoints)
        for i, (s, b) in enumerate(self.params):
            mask = (bins == i)
            out[mask] = s * pred[mask] + b
        return out


# ---------------------------------------------------------------------------
# Подгонка
# ---------------------------------------------------------------------------

def fit_global_linear(paired: pd.DataFrame) -> GlobalLinearCorrector:
    """y_obs = slope * y_pred + intercept, обычный OLS."""
    x = paired['magt_pred'].values.reshape(-1, 1)
    y = paired['magt_obs'].values
    lr = LinearRegression().fit(x, y)
    return GlobalLinearCorrector(
        slope=float(lr.coef_[0]),
        intercept=float(lr.intercept_),
    )


def fit_per_zone_linear(paired: pd.DataFrame,
                        min_points_per_zone: int = 4) -> PerZoneLinearCorrector:
    """
    Своя linear regression на каждую зону.
    Если в зоне < min_points_per_zone — fallback на identity (slope=1, intercept=0).
    """
    params = {}
    for z in [0, 1, 2, 3]:
        sub = paired[paired['zone_pred'] == z]
        if len(sub) < min_points_per_zone:
            params[z] = (1.0, 0.0)
            continue
        x = sub['magt_pred'].values.reshape(-1, 1)
        y = sub['magt_obs'].values
        lr = LinearRegression().fit(x, y)
        params[z] = (float(lr.coef_[0]), float(lr.intercept_))
    return PerZoneLinearCorrector(params=params)


def fit_piecewise_linear(paired: pd.DataFrame,
                         breakpoints: np.ndarray = None) -> PiecewiseLinearCorrector:
    """Своя linear regression на каждый бин исходного предсказания."""
    if breakpoints is None:
        breakpoints = np.array([-10, -5, -2, 0])
    bins = np.digitize(paired['magt_pred'].values, breakpoints)
    params = []
    for i in range(len(breakpoints) + 1):
        sub = paired[bins == i]
        if len(sub) < 3:
            params.append((1.0, 0.0))
            continue
        x = sub['magt_pred'].values.reshape(-1, 1)
        y = sub['magt_obs'].values
        lr = LinearRegression().fit(x, y)
        params.append((float(lr.coef_[0]), float(lr.intercept_)))
    return PiecewiseLinearCorrector(breakpoints=breakpoints, params=params)


# ---------------------------------------------------------------------------
# Применение к полной карте
# ---------------------------------------------------------------------------

def apply_correction(
    magt_pred: np.ndarray,
    corrector: Callable,
    zones: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Применяет калибровщик к полной карте.

    Для PerZoneLinearCorrector желательно передать zones — либо реальные зоны
    мерзлоты, либо None и тогда они посчитаются по pred.
    """
    if zones is not None:
        return corrector(magt_pred, zones=zones)
    return corrector(magt_pred)


# ---------------------------------------------------------------------------
# Сравнение стратегий: RMSE / bias / R² через k-fold CV на бурениях
# ---------------------------------------------------------------------------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        'rmse': float(np.sqrt(np.mean(err ** 2))),
        'bias': float(np.mean(err)),
        'mae': float(np.mean(np.abs(err))),
        'r2': float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        'n': int(len(y_true)),
    }


def compare_strategies(paired: pd.DataFrame, n_folds: int = 5,
                       random_state: int = 42) -> dict:
    """
    K-fold cross-validation на 33 бурениях, чтобы выбрать лучшую стратегию
    без оверфита на тех же точках, по которым калибровали.

    Возвращает dict со всеми метриками + объектом-калибровщиком, обученным
    на ВСЕХ бурениях (для применения к карте).
    """
    from sklearn.model_selection import KFold

    n = len(paired)
    if n < n_folds:
        n_folds = max(2, n // 3)

    strategies = {
        'identity':  (lambda df: GlobalLinearCorrector(slope=1.0, intercept=0.0)),
        'global':    fit_global_linear,
        'per_zone':  fit_per_zone_linear,
        'piecewise': fit_piecewise_linear,
    }

    results = {}
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    indices = np.arange(n)

    for name, fitter in strategies.items():
        fold_metrics = []
        for train_idx, val_idx in kf.split(indices):
            train = paired.iloc[train_idx]
            val = paired.iloc[val_idx]
            corrector = fitter(train)
            y_corrected = corrector(val['magt_pred'].values,
                                    zones=val['zone_pred'].values)
            fold_metrics.append(_metrics(val['magt_obs'].values, y_corrected))

        cv = {
            'rmse_mean': np.mean([m['rmse'] for m in fold_metrics]),
            'rmse_std':  np.std([m['rmse'] for m in fold_metrics]),
            'bias_mean': np.mean([m['bias'] for m in fold_metrics]),
            'mae_mean':  np.mean([m['mae'] for m in fold_metrics]),
        }

        # Финальная подгонка на ВСЕХ точках — этим объектом потом корректируем карту
        final_corrector = fitter(paired)
        train_metrics = _metrics(
            paired['magt_obs'].values,
            final_corrector(paired['magt_pred'].values,
                            zones=paired['zone_pred'].values),
        )

        results[name] = {
            'cv': cv,
            'train': train_metrics,
            'corrector': final_corrector,
        }

    return results


def pick_best_strategy(results: dict,
                       prefer_simpler_when_close: float = 0.05) -> str:
    """
    Выбирает стратегию с минимальным CV RMSE.
    Если две стратегии в пределах prefer_simpler_when_close°C — выбираем простейшую
    (identity < global < piecewise < per_zone по сложности).

    Этот тиебрейк защищает от защиты «вы переоверфитили на 33 точках».
    """
    complexity_order = ['identity', 'global', 'piecewise', 'per_zone']
    sorted_by_rmse = sorted(results.items(), key=lambda kv: kv[1]['cv']['rmse_mean'])
    best_name, best_metrics = sorted_by_rmse[0]
    best_rmse = best_metrics['cv']['rmse_mean']

    # Ищем простейшую стратегию в пределах prefer_simpler_when_close
    for name in complexity_order:
        if name in results and results[name]['cv']['rmse_mean'] <= best_rmse + prefer_simpler_when_close:
            return name
    return best_name


# ---------------------------------------------------------------------------
# Bootstrap для неопределённости параметров (вместо доверительных интервалов)
# ---------------------------------------------------------------------------

def bootstrap_corrector(paired: pd.DataFrame, fitter: Callable,
                        n_boot: int = 1000,
                        random_state: int = 42) -> dict:
    """
    Bootstrap для оценки неопределённости параметров калибровщика.
    Возвращает распределение коэффициентов.
    """
    rng = np.random.default_rng(random_state)
    n = len(paired)
    slopes, intercepts = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sub = paired.iloc[idx]
        try:
            c = fitter(sub)
            if isinstance(c, GlobalLinearCorrector):
                slopes.append(c.slope)
                intercepts.append(c.intercept)
        except Exception:
            continue

    return {
        'slope_mean': np.mean(slopes) if slopes else None,
        'slope_std':  np.std(slopes) if slopes else None,
        'intercept_mean': np.mean(intercepts) if intercepts else None,
        'intercept_std':  np.std(intercepts) if intercepts else None,
        'n_successful_resamples': len(slopes),
    }
