"""
Перепроекция Obu et al. 2019 MAGT TIFF из EPSG:3995 в нашу сетку (EPSG:4326).

Входной файл (1.8 ГБ):
    data/raw/Obu/UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif

Выходной файл (1.3 МБ):
    data/obu_2019_reprojected.npz
        obu_on_grid: (231, 1501) — MAGT на нашей сетке
        lons, lats — координатные массивы

Использование:
    python scripts/reproject_obu.py
"""

import sys
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
except ImportError:
    print("ОШИБКА: rasterio не установлен. Установите: pip install rasterio")
    sys.exit(1)


# Параметры нашей сетки (см. data/README.md)
LON_MIN, LON_MAX = 30.0, 180.0
LAT_MIN, LAT_MAX = 55.0, 78.0
H_OUT, W_OUT = 231, 1501


def main():
    base_dir = Path(__file__).parent.parent
    obu_path = base_dir / 'data' / 'raw' / 'Obu' / 'UiO_PEX_MAGTM_5.0_20181127_2000_2016_NH.tif'
    out_path = base_dir / 'data' / 'obu_2019_reprojected.npz'

    if not obu_path.exists():
        print(f'ОШИБКА: входной файл не найден: {obu_path}')
        print('Скачайте Obu MAGT по ссылке в data/external_links.md')
        sys.exit(1)

    print(f'Открываем: {obu_path}')
    with rasterio.open(obu_path) as src:
        print(f'  CRS: {src.crs}')
        print(f'  Size: {src.width} x {src.height}')
        print(f'  Bounds: {src.bounds}')

    # Целевая трансформация (наша сетка)
    dst_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, W_OUT, H_OUT)
    dst_crs = 'EPSG:4326'

    print('\nПерепроекция из EPSG:3995 в EPSG:4326...')
    obu_on_grid = np.full((H_OUT, W_OUT), np.nan, dtype=np.float32)

    with rasterio.open(obu_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=obu_on_grid,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    # Y-ось в rasterio идёт сверху вниз, наша lats — снизу вверх
    obu_on_grid = obu_on_grid[::-1]

    # Маска валидных
    valid = ~np.isnan(obu_on_grid) & ~np.isinf(obu_on_grid)
    print(f'  Валидных пикселей: {valid.sum():,} ({100*valid.sum()/obu_on_grid.size:.1f}%)')
    print(f'  Диапазон: [{obu_on_grid[valid].min():+.2f}, {obu_on_grid[valid].max():+.2f}]°C')
    print(f'  Среднее: {obu_on_grid[valid].mean():+.2f}°C')

    # Координатные массивы (для сохранения)
    lons = np.linspace(LON_MIN, LON_MAX, W_OUT)
    lats = np.linspace(LAT_MIN, LAT_MAX, H_OUT)

    np.savez(out_path, obu_on_grid=obu_on_grid, lons=lons, lats=lats)
    print(f'\nСохранено: {out_path} ({out_path.stat().st_size / 1024**2:.1f} МБ)')


if __name__ == '__main__':
    main()
