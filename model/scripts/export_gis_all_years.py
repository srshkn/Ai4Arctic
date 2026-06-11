"""
Склейка MAGT 2007–2035 и экспорт для GIS.

Источники:
  results/maps/p2_yearly_maps.npz           — 2007–2024 (real P2)
  results/maps/p2_real_2025.npz            — 2025 (real P2)
  results/maps/p3_projection_2026_2035_*.npz — 2026–2035 (прогноз P3)

Выход:
  results/maps/magt_all_years_2007_2035.npz   — единый стек (29, H, W)
  results/gis/MAGT_<year>.tif                 — GeoTIFF по годам (опционально)

Использование:
  python3 scripts/export_gis_all_years.py
  python3 scripts/export_gis_all_years.py --projection modelB --export-tiff
  python3 scripts/export_gis_all_years.py --export-tiff --tiff-dir results/gis
"""

import argparse
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

MAPS_DIR = BASE_DIR / 'results' / 'maps'
DEFAULT_OUT_NPZ = MAPS_DIR / 'magt_all_years_2007_2035.npz'
DEFAULT_TIFF_DIR = BASE_DIR / 'results' / 'gis'

PROJECTION_FILES = {
    'modelA': MAPS_DIR / 'p3_projection_2026_2035_modelA.npz',
    'modelB': MAPS_DIR / 'p3_projection_2026_2035_modelB.npz',
}


def load_all_years(projection: str):
    """Собрать MAGT (years, H, W) и координаты."""
    hist_path = MAPS_DIR / 'p2_yearly_maps.npz'
    y2025_path = MAPS_DIR / 'p2_real_2025.npz'
    proj_path = PROJECTION_FILES[projection]

    for p in (hist_path, y2025_path, proj_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Не найден: {p}\n"
                "Запустите inference_p2_history.py и projection_2026_2035.py."
            )

    hist = np.load(hist_path)
    years_hist = list(hist['years'])
    magt_hist = hist['magt_yearly']
    lats, lons = hist['lats'], hist['lons']

    y2025 = np.load(y2025_path)
    magt_2025 = y2025['magt'][None, ...]  # (1, H, W)

    proj = np.load(proj_path)
    years_proj = list(proj['years'])
    magt_proj = proj['magt']

    magt_all = np.concatenate([magt_hist, magt_2025, magt_proj], axis=0)
    years_all = years_hist + [int(y2025['year'])] + years_proj

    segments = {
        '2007_2024': 'P2 real (p2_yearly_maps.npz)',
        '2025': 'P2 real (p2_real_2025.npz)',
        '2026_2035': f'P3 forecast ({proj_path.name})',
    }
    return magt_all, np.array(years_all), lats, lons, segments, proj_path.name


def write_geotiff(path: Path, array_hw: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    import rasterio
    from rasterio.transform import from_bounds

    h, w = array_hw.shape
    west = float(lons.min()) - 0.05
    east = float(lons.max()) + 0.05
    south = float(lats.min()) - 0.05
    north = float(lats.max()) + 0.05
    transform = from_bounds(west, south, east, north, w, h)

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=h,
        width=w,
        count=1,
        dtype='float32',
        crs='EPSG:4326',
        transform=transform,
        nodata=np.nan,
        compress='lzw',
    ) as dst:
        dst.write(array_hw.astype(np.float32), 1)
        dst.set_band_description(1, 'MAGT_degC')


def main():
    parser = argparse.ArgumentParser(description='Экспорт MAGT 2007–2035 для GIS')
    parser.add_argument(
        '--projection',
        choices=['modelA', 'modelB'],
        default='modelB',
        help='Какой прогноз 2026–2035 использовать (default: modelB)',
    )
    parser.add_argument(
        '--out-npz',
        type=Path,
        default=DEFAULT_OUT_NPZ,
        help='Путь к объединённому .npz',
    )
    parser.add_argument(
        '--export-tiff',
        action='store_true',
        help='Дополнительно сохранить GeoTIFF на каждый год',
    )
    parser.add_argument(
        '--tiff-dir',
        type=Path,
        default=DEFAULT_TIFF_DIR,
        help='Папка для GeoTIFF (default: results/gis/)',
    )
    args = parser.parse_args()

    print('Загрузка и склейка 2007–2035...')
    magt, years, lats, lons, segments, proj_name = load_all_years(args.projection)
    print(f"  Форма: {magt.shape}  ({years[0]}..{years[-1]}, {len(years)} лет)")
    print(f"  2007–2024: {segments['2007_2024']}")
    print(f"  2025:      {segments['2025']}")
    print(f"  2026–2035: {segments['2026_2035']}")

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        magt=magt.astype(np.float32),
        years=years,
        lats=lats,
        lons=lons,
        crs='EPSG:4326',
        units='degC',
        description=(
            'MAGT 2007-2035: P2 real 2007-2025 + P3 projection 2026-2035 '
            f'({proj_name}), with lh correction and P1 shift'
        ),
        source_history='p2_yearly_maps.npz + p2_real_2025.npz',
        source_projection=proj_name,
    )
    print(f"\nСохранено: {args.out_npz} ({args.out_npz.stat().st_size / 1e6:.1f} МБ)")

    if args.export_tiff:
        print(f"\nЭкспорт GeoTIFF → {args.tiff_dir}/")
        for i, year in enumerate(years):
            tiff_path = args.tiff_dir / f'MAGT_{year}.tif'
            write_geotiff(tiff_path, magt[i], lats, lons)
            print(f"  {tiff_path.name}")
        print(f"\nГотово: {len(years)} файлов. Откройте в QGIS (CRS: EPSG:4326).")
    else:
        print("\nДля GeoTIFF добавьте флаг: --export-tiff")


if __name__ == '__main__':
    main()
