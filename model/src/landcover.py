"""
Классификация landcover и физически обоснованные коррекции.

Содержит:
- classify_landcover : построение карты 5 классов (торфяники, тайга, тундра,
                       голая почва, горы) из тензора признаков
- build_rk_map       : карта коэффициента r_k для формулы TTOP по landcover
- build_delta_t_map  : карта поправки ΔT на латентную теплоту по landcover
- compute_alt        : расчёт ALT по формуле Стефана
- compute_ttop_target: расчёт target по формуле TTOP с rk(landcover)

Значения r_k взяты из независимых физических публикаций:
- Smith & Riseborough (2002), Table 1
- Romanovsky & Osterkamp (1995, 1997)
- Obu et al. (2019), Table 1
"""

import numpy as np


# Таблица r_k по landcover (Smith&Riseborough 2002, Romanovsky&Osterkamp 1995, Obu 2019)
RK_TABLE = {
    0: 0.70,   # не определено — дефолт (среднее)
    1: 0.55,   # торфяники
    2: 0.85,   # тайга
    3: 0.65,   # тундра
    4: 0.95,   # голая почва / скалы
    5: 0.95,   # горы
}

# Таблица поправок ΔT на латентную теплоту (°C)
DELTA_T_TABLE = {
    0: 2.0,    # не определено
    1: 4.0,    # торфяники
    2: 2.5,    # тайга
    3: 1.5,    # тундра
    4: 0.5,    # голая почва
    5: 1.0,    # горы
}

# Edaphic factor E для формулы Стефана (м/√К·сут), Romanovsky & Osterkamp 1997
E_TABLE = {
    0: 0.030,  # не определено
    1: 0.015,  # торфяники
    2: 0.025,  # тайга
    3: 0.035,  # тундра
    4: 0.045,  # голая почва
    5: 0.040,  # горы
}


def classify_landcover(X, indices=None):
    """
    Классификация landcover на 5 классов через средние значения признаков за 14 лет.

    Используются признаки (по умолчанию):
        NDVI (idx 0), NDWI (idx 1), soil_oc (idx 13),
        elevation (idx 10), slope (idx 11)

    Алгоритм (версия v3 — финальная):
        5 (горы)      : elevation > 800 ИЛИ slope > 8
        1 (торфяники) : soc > 75-й перц И ndwi > 50-й перц И НЕ горы
        2 (тайга)     : ndvi > 0.55 И НЕ торф И НЕ горы
        3 (тундра)    : 0.2 <= ndvi <= 0.55 И НЕ торф И НЕ горы
        4 (голая)     : ndvi < 0.2 И НЕ торф И НЕ горы
        0 (не определено): NaN в NDVI (вне России)

    Parameters
    ----------
    X : ndarray (T, H, W, C=20)
        Тензор признаков из tensor_01deg_v2.npz.
    indices : dict, optional
        Индексы признаков, если структура тензора отличается.

    Returns
    -------
    landcover : ndarray (H, W) — массив int с классами 0-5
    """
    if indices is None:
        indices = {'ndvi': 0, 'ndwi': 1, 'soil_oc': 13, 'elev': 10, 'slope': 11}

    X_mean = np.nanmean(X, axis=0)
    ndvi = X_mean[..., indices['ndvi']]
    ndwi = X_mean[..., indices['ndwi']]
    soc = X_mean[..., indices['soil_oc']]
    elev = X_mean[..., indices['elev']]
    slope = X_mean[..., indices['slope']]

    soc_p75 = np.nanpercentile(soc, 75)
    ndwi_p50 = np.nanpercentile(ndwi, 50)

    landcover = np.full(ndvi.shape, 0, dtype=np.int8)
    mountains = (elev > 800) | (slope > 8)
    peat = (soc > soc_p75) & (ndwi > ndwi_p50) & ~mountains
    forest = (ndvi > 0.55) & ~peat & ~mountains
    tundra = (ndvi >= 0.2) & (ndvi <= 0.55) & ~peat & ~mountains
    bare = (ndvi < 0.2) & ~peat & ~mountains

    landcover[peat] = 1
    landcover[forest] = 2
    landcover[tundra] = 3
    landcover[bare] = 4
    landcover[mountains] = 5
    landcover[np.isnan(ndvi)] = 0

    return landcover


def build_rk_map(landcover, table=None):
    """Карта r_k по landcover."""
    if table is None:
        table = RK_TABLE
    rk_map = np.full(landcover.shape, table[0], dtype=np.float32)
    for cls, val in table.items():
        rk_map[landcover == cls] = val
    return rk_map


def build_delta_t_map(landcover, lats=None, table=None, lat_correction=True):
    """
    Карта поправки ΔT на латентную теплоту по landcover.

    Если задан lats и lat_correction=True, на широтах > 70°N
    применяется доп. снижение на 0.3 °C (но не ниже 0.5).
    """
    if table is None:
        table = DELTA_T_TABLE
    delta_t = np.zeros(landcover.shape, dtype=np.float32)
    for cls, val in table.items():
        delta_t[landcover == cls] = val

    if lats is not None and lat_correction:
        # Создаём 2D массив широт под форму landcover
        LAT = np.tile(lats[:, None], (1, landcover.shape[1]))
        high_lat = LAT > 70
        delta_t[high_lat] = np.maximum(delta_t[high_lat] - 0.3, 0.5)

    return delta_t


def compute_ttop_target(X, landcover, fdd_idx=7, tdd_idx=8, rk_table=None):
    """
    Расчёт target TTOP с варьируемым r_k(landcover).

    Формула: y = (r_k(landcover) · TDD - FDD) / 365

    Parameters
    ----------
    X : ndarray (T, H, W, C)
        Тензор признаков. FDD и TDD должны быть в исходных К·сут (не нормализованы).
    landcover : ndarray (H, W)
        Карта классов landcover.
    fdd_idx, tdd_idx : int
        Индексы FDD и TDD в тензоре.

    Returns
    -------
    y_new : ndarray (T, H, W)
        Новый target в °C.
    """
    if rk_table is None:
        rk_table = RK_TABLE
    rk_map = build_rk_map(landcover, rk_table)
    FDD = X[..., fdd_idx]
    TDD = X[..., tdd_idx]
    y_new = (rk_map[None, :, :] * TDD - FDD) / 365.0
    return y_new


def compute_alt(TDD_year, landcover, magt_for_mask=None, e_table=None):
    """
    Расчёт ALT по формуле Стефана: ALT = E(landcover) · √TDD

    Parameters
    ----------
    TDD_year : ndarray (H, W)
        TDD для конкретного года в К·сут.
    landcover : ndarray (H, W)
    magt_for_mask : ndarray (H, W), optional
        Если задан — ALT маскируется как NaN где MAGT >= 0 (нет мерзлоты).
    e_table : dict, optional
        Таблица edaphic factor.

    Returns
    -------
    ALT : ndarray (H, W) — толщина деятельного слоя в метрах.
    """
    if e_table is None:
        e_table = E_TABLE
    E_map = np.full(landcover.shape, e_table[0], dtype=np.float32)
    for cls, val in e_table.items():
        E_map[landcover == cls] = val

    TDD_safe = np.maximum(TDD_year, 0)
    ALT = E_map * np.sqrt(TDD_safe)

    if magt_for_mask is not None:
        ALT = np.where(magt_for_mask < 0, ALT, np.nan)

    return ALT


def compute_vulnerability(magt, alt, trend_per_year, masked_by_magt=True):
    """
    Композитный индекс уязвимости мерзлоты.

    Composite = (F1 + F2 + F3) / 3, где:
        F1 = clip((magt + 15) / 15, 0, 1)     — близость к 0 °C
        F2 = clip((alt - 0.3) / 2.0, 0, 1)    — толщина ALT
        F3 = clip(trend / 0.3, 0, 1)          — тренд потепления

    Categories:
        1 (низкая)     : < 0.25
        2 (умеренная)  : 0.25 - 0.45
        3 (высокая)    : 0.45 - 0.65
        4 (оч.высокая) : >= 0.65
    """
    F1 = np.clip((magt + 15) / 15, 0, 1)
    F2 = np.clip((alt - 0.3) / 2.0, 0, 1)
    F3 = np.clip(trend_per_year / 0.3, 0, 1)
    vulnerability = (F1 + F2 + F3) / 3

    if masked_by_magt:
        vulnerability = np.where(magt < 0, vulnerability, np.nan)

    categories = np.full_like(vulnerability, np.nan)
    categories[(vulnerability >= 0) & (vulnerability < 0.25)] = 1
    categories[(vulnerability >= 0.25) & (vulnerability < 0.45)] = 2
    categories[(vulnerability >= 0.45) & (vulnerability < 0.65)] = 3
    categories[vulnerability >= 0.65] = 4

    return vulnerability, categories
