"""
ESA CCI Permafrost: загрузка, репроекция, создание target для обучения.

Использует ESA CCI Permafrost CRDP v04 (Obu et al. 2021) как ground truth
вместо самовычисленного TTOP. Это меняет всю парадигму обучения:
    - старое: train_target = TTOP по своей формуле (наследует cold bias)
    - новое:  train_target = ESA CCI MAGT (валидированный продукт, ~2°C RMSE
              против бурений документировано)

Использование:
    from src.esa_cci import (
        load_esa_cci_year,
        reproject_to_grid,
        build_esa_target_tensor,
    )

    target_2010 = load_esa_cci_year(esa_cci_dir / 'magt' / 'ESACCI-...-2010-fv04.0.nc',
                                     bbox=(30, 55, 180, 78))
    target_reprojected = reproject_to_grid(target_2010, target_lats, target_lons)

    # Для batch-обработки всех лет:
    target_tensor = build_esa_target_tensor(
        esa_cci_dir=DATA_DIR / 'esa_cci' / 'magt',
        years=range(2003, 2020),
        target_lats=lats,
        target_lons=lons,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

try:
    from scipy.interpolate import RegularGridInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_esa_cci_year(
    nc_path: Path,
    variable: str = 'MAGT',
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает один годовой файл ESA CCI Permafrost.

    Args:
        nc_path: путь к NetCDF
        variable: какую переменную брать (MAGT, ALT, PFR)
        bbox: (lon_min, lat_min, lon_max, lat_max) для обрезки — иначе очень
              большой массив

    Returns:
        (magt_array, lats, lons) — все 2D arrays
    """
    if not HAS_XARRAY:
        raise ImportError("Установи xarray: pip install xarray netcdf4")

    ds = xr.open_dataset(nc_path)
    print(f"Переменные в файле: {list(ds.data_vars)}")

    # Имя переменной может варьироваться: MAGT, magt, mean_annual_ground_temperature
    candidates = [variable, variable.lower(), 'magt', 'MAGT',
                  'mean_annual_ground_temperature']
    var = None
    for cand in candidates:
        if cand in ds.data_vars:
            var = cand
            break
    if var is None:
        raise KeyError(f"Не нашёл переменную {variable} в {nc_path}. "
                       f"Доступны: {list(ds.data_vars)}")

    # Обрезка по bbox
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        # Имена координат могут быть lat/lon или latitude/longitude
        lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
        lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
        # ESA CCI обычно идёт от северо-западного угла → широта убывает
        if ds[lat_name][0] > ds[lat_name][-1]:
            ds = ds.sel({lat_name: slice(lat_max, lat_min), lon_name: slice(lon_min, lon_max)})
        else:
            ds = ds.sel({lat_name: slice(lat_min, lat_max), lon_name: slice(lon_min, lon_max)})

    magt = ds[var].values.astype(np.float32)
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Если по широте убывает — переворачиваем для совместимости с np.argmin/argmax-логикой
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        magt = magt[::-1, :]

    print(f"  Загружено: {magt.shape}, lats [{lats[0]:.2f}, {lats[-1]:.2f}], "
          f"lons [{lons[0]:.2f}, {lons[-1]:.2f}]")
    return magt, lats, lons


# ---------------------------------------------------------------------------
# Репроекция на нашу 0.1° сетку
# ---------------------------------------------------------------------------

def reproject_to_grid(
    source_array: np.ndarray,
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    method: str = 'linear',
) -> np.ndarray:
    """
    Билинейная интерполяция ESA CCI (1 км) на нашу 0.1° (~11 км) сетку.

    Args:
        source_array: 2D (H_src, W_src) ESA CCI массив
        source_lats, source_lons: 1D координаты source
        target_lats, target_lons: 1D координаты нашей grid
        method: 'linear' | 'nearest' | 'cubic'

    Returns:
        2D массив (len(target_lats), len(target_lons))
    """
    if not HAS_SCIPY:
        raise ImportError("Установи scipy: pip install scipy")

    # Заменяем NaN на дефолтное значение для интерполятора (потом маскируем обратно)
    valid_mask = ~np.isnan(source_array)
    source_filled = np.where(valid_mask, source_array, 0.0)

    interp = RegularGridInterpolator(
        (source_lats, source_lons),
        source_filled,
        method=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Создаём сетку target координат
    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
    out = interp(points).reshape(len(target_lats), len(target_lons))

    # Также интерполируем mask, чтобы возвращать NaN там, где не было данных
    mask_interp = RegularGridInterpolator(
        (source_lats, source_lons),
        valid_mask.astype(np.float32),
        method='nearest',
        bounds_error=False,
        fill_value=0.0,
    )
    mask_out = mask_interp(points).reshape(len(target_lats), len(target_lons))
    out[mask_out < 0.5] = np.nan

    return out


# ---------------------------------------------------------------------------
# Batch: построение target-тензора для всех лет
# ---------------------------------------------------------------------------

def build_esa_target_tensor(
    esa_cci_dir: Path,
    years: List[int],
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    bbox: Tuple[float, float, float, float] = (30, 55, 180, 78),
    save_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Собирает target-тензор (T, H, W), где T = годы, H/W = размер нашей сетки.

    Args:
        esa_cci_dir: папка с *.nc файлами для всех годов
        years: какие годы включить
        target_lats, target_lons: наша сетка
        bbox: для обрезки source при загрузке
        save_path: куда сохранить npz (если задано)

    Returns:
        тензор (T, H, W), float32
    """
    H, W = len(target_lats), len(target_lons)
    target_tensor = np.full((len(years), H, W), np.nan, dtype=np.float32)

    for i, year in enumerate(years):
        # Ищем файл по году
        nc_files = list(Path(esa_cci_dir).glob(f'*{year}*.nc'))
        if not nc_files:
            print(f"  ! Год {year}: файл не найден в {esa_cci_dir}, оставляем NaN")
            continue
        nc_path = nc_files[0]

        print(f"\nГод {year}: {nc_path.name}")
        try:
            magt_src, lats_src, lons_src = load_esa_cci_year(nc_path, bbox=bbox)
            magt_reproj = reproject_to_grid(magt_src, lats_src, lons_src,
                                             target_lats, target_lons)
            target_tensor[i] = magt_reproj
            print(f"  Reprojected: valid pixels = {np.sum(~np.isnan(magt_reproj))}/{H*W}")
        except Exception as e:
            print(f"  ! Ошибка обработки {year}: {e}")

    if save_path:
        np.savez_compressed(
            save_path,
            target_magt=target_tensor,
            years=np.array(years),
            lats=target_lats,
            lons=target_lons,
        )
        print(f"\nСохранено: {save_path}")

    return target_tensor


# ---------------------------------------------------------------------------
# Сопоставление temporal alignment с feature тензором
# ---------------------------------------------------------------------------

def align_features_with_esa_target(
    feature_tensor: np.ndarray,   # (T_feat, F, H, W)
    feature_years: np.ndarray,    # (T_feat,)
    esa_target_tensor: np.ndarray,  # (T_esa, H, W)
    esa_years: np.ndarray,        # (T_esa,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Оставляет только годы, которые есть и в features, и в ESA target.

    Returns:
        aligned_features: (T_common, F, H, W)
        aligned_target:   (T_common, H, W)
        common_years:     (T_common,)
    """
    common = np.intersect1d(feature_years, esa_years)
    print(f"Общих годов: {len(common)} — {common}")

    feat_indices = [int(np.where(feature_years == y)[0][0]) for y in common]
    esa_indices = [int(np.where(esa_years == y)[0][0]) for y in common]

    return (
        feature_tensor[feat_indices],
        esa_target_tensor[esa_indices],
        common,
    )
