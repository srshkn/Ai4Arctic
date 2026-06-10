# -*- coding: utf-8 -*-
"""
Универсальный GEE export годовых признаков для любых годов.

Заменяет:
  - gee_export_2003_2009.py  (был YEARS = 2003..2024)
  - gee_export_2025.py        (был YEARS = [2025])

Использование:
  # Один год:
  python3 scripts/gee_export.py --years 2025

  # Диапазон:
  python3 scripts/gee_export.py --years 2003-2025

  # Несколько отдельных годов:
  python3 scripts/gee_export.py --years 2024,2025

Drive folder: GEE_exports_features_01deg_v2_extended/

Что выгружается:
  20 признаков: NDVI, NDWI, NDMI, SAVI, LST_summer, LST_winter, LST_annual,
                FDD, TDD, snow_days, elevation, slope, aspect,
                soil_oc, soil_bd, soil_clay, MAAT, MAP, era5_temp, era5_precip

Время:
  Один год = 4 поддиапазона = 4 GEE tasks (~30-60 минут в очереди)
  Диапазон 2003-2025 = 23 года × 4 = 92 задачи (~несколько часов)
"""

import argparse
import sys
import ee


# ===== Константы pipeline =====
STEP_DEG = 0.1
EXPORT_FOLDER = 'GEE_exports_features_01deg_v2_extended'
SOUTH, NORTH = 55, 78

BANDS = [
    ('band0',      30, 80),
    ('band1_west', 80, 105),
    ('band1_east', 105, 130),
    ('band2',      130, 180),
]


def parse_years(years_str):
    """
    Парсит строку --years в список годов.

    Примеры:
      "2025"       -> [2025]
      "2003-2025"  -> [2003, 2004, ..., 2025]
      "2024,2025"  -> [2024, 2025]
    """
    years = []
    for part in years_str.split(','):
        part = part.strip()
        if '-' in part:
            # диапазон, например "2003-2025"
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            if start > end:
                raise ValueError(f"Неверный диапазон {part}: start > end")
            years.extend(range(start, end + 1))
        else:
            # один год
            years.append(int(part))

    # Уникальные и сортированные
    years = sorted(set(years))

    # Sanity checks
    if not years:
        raise ValueError("Не указано ни одного года")
    if min(years) < 2000:
        raise ValueError(f"Год {min(years)} раньше 2000 — Landsat 5 не поддерживается до 1985")
    if max(years) > 2030:
        raise ValueError(f"Год {max(years)} в будущем — данных ещё нет в GEE")

    return years


def get_landsat(year, region):
    """Унификация Landsat 5/7/8/9 по эпохам."""
    if year <= 2012:
        l5 = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2') if year <= 2011 else None
        l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
        if l5 is not None:
            coll = l5.merge(l7)
        else:
            coll = l7
        coll = (coll.filterDate(f'{year}-06-01', f'{year}-08-31')
                    .filterBounds(region)
                    .filter(ee.Filter.lt('CLOUD_COVER', 60)))
        bands_src = ['SR_B3', 'SR_B4', 'SR_B2', 'SR_B5']
    elif year <= 2020:
        coll = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                .filterDate(f'{year}-06-01', f'{year}-08-31')
                .filterBounds(region)
                .filter(ee.Filter.lt('CLOUD_COVER', 60)))
        bands_src = ['SR_B4', 'SR_B5', 'SR_B3', 'SR_B6']
    else:
        l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
              .filterDate(f'{year}-06-01', f'{year}-08-31')
              .filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', 60)))
        l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
              .filterDate(f'{year}-06-01', f'{year}-08-31')
              .filterBounds(region)
              .filter(ee.Filter.lt('CLOUD_COVER', 60)))
        coll = l8.merge(l9)
        bands_src = ['SR_B4', 'SR_B5', 'SR_B3', 'SR_B6']

    def renamebands(img):
        return img.select(bands_src, ['RED', 'NIR', 'GREEN', 'SWIR']).multiply(2.75e-5).add(-0.2)

    img = coll.map(renamebands).median()
    ndvi = img.expression('(NIR - RED) / (NIR + RED)',
                          {'NIR': img.select('NIR'), 'RED': img.select('RED')}).rename('NDVI')
    ndwi = img.expression('(GREEN - NIR) / (GREEN + NIR)',
                          {'GREEN': img.select('GREEN'), 'NIR': img.select('NIR')}).rename('NDWI')
    ndmi = img.expression('(NIR - SWIR) / (NIR + SWIR)',
                          {'NIR': img.select('NIR'), 'SWIR': img.select('SWIR')}).rename('NDMI')
    savi = img.expression('1.2 * (NIR - RED) / (NIR + RED + 0.2)',
                          {'NIR': img.select('NIR'), 'RED': img.select('RED')}).rename('SAVI')
    return ee.Image.cat([ndvi, ndwi, ndmi, savi])


def get_lst_features(year, region):
    """
    LST признаки. ВАЖНО:
    - MODIS Terra (LST_summer) работал до 15 октября 2025
    - ERA5-Land доступна за все годы 2003-настоящее
    """
    modis_raw = (ee.ImageCollection('MODIS/061/MOD11A1')
                 .filterDate(f'{year}-06-01', f'{year}-08-31')
                 .select('LST_Day_1km'))

    era5_raw = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
                .filterDate(f'{year}-01-01', f'{year}-12-31')
                .select('temperature_2m'))

    def to_celsius_modis(col):
        return col.mean().multiply(0.02).subtract(273.15)

    def to_celsius_era(col):
        return col.mean().subtract(273.15)

    lst_summer = to_celsius_modis(modis_raw).rename('LST_summer')
    era_winter = era5_raw.filter(ee.Filter.calendarRange(12, 2, 'month'))
    lst_winter = to_celsius_era(era_winter).rename('LST_winter')
    lst_annual = to_celsius_era(era5_raw).rename('LST_annual')

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    fdd = ee.Image(0)
    tdd = ee.Image(0)
    for i, m in enumerate(range(1, 13)):
        t = to_celsius_era(era5_raw.filter(ee.Filter.calendarRange(m, m, 'month')))
        d = days_in_month[i]
        fdd = fdd.add(t.multiply(-1).max(0).multiply(d))
        tdd = tdd.add(t.max(0).multiply(d))

    return ee.Image.cat([lst_summer, lst_winter, lst_annual,
                         fdd.rename('FDD'), tdd.rename('TDD')])


def get_snow(year):
    snow = (ee.ImageCollection('MODIS/061/MOD10A1')
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .select('NDSI_Snow_Cover'))
    return snow.map(lambda i: i.gt(40)).sum().rename('snow_days')


def get_terrain():
    dem = ee.Image('MERIT/DEM/v1_0_3').rename('elevation')
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    return ee.Image.cat([dem, slope, aspect])


def get_soil():
    oc = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02').select('b0').rename('soil_oc')
    bd = ee.Image('OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02').select('b0').rename('soil_bd')
    clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0').rename('soil_clay')
    return ee.Image.cat([oc, bd, clay])


def get_climate(year):
    wc = ee.Image('WORLDCLIM/V1/BIO')
    maat = wc.select('bio01').rename('MAAT')
    mAp = wc.select('bio12').rename('MAP')
    era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
            .filterDate(f'{year}-01-01', f'{year}-12-31'))
    era5_t = era5.select('temperature_2m').mean().subtract(273.15).rename('era5_temp')
    era5_p = era5.select('total_precipitation_sum').mean().rename('era5_precip')
    return ee.Image.cat([maat, mAp, era5_t, era5_p])


def export_year_subband(year, band_name, west, east):
    region = ee.Geometry.Rectangle([west, SOUTH, east, NORTH], proj='EPSG:4326', geodesic=False)
    lons = ee.List.sequence(west, east, STEP_DEG)
    lats = ee.List.sequence(SOUTH, NORTH, STEP_DEG)

    def make_row(lat):
        lat = ee.Number(lat)
        return lons.map(lambda lon: ee.Feature(ee.Geometry.Point([ee.Number(lon), lat])))
    grid = ee.FeatureCollection(lats.map(make_row).flatten())

    img = ee.Image.cat([
        get_landsat(year, region),
        get_lst_features(year, region),
        get_snow(year),
        get_terrain(),
        get_soil(),
        get_climate(year)
    ])
    sampled = img.sampleRegions(collection=grid, scale=10000, geometries=True)
    fname = f'RussiaGrid_0.1deg_v2_{year}_{band_name}'
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=fname,
        fileNamePrefix=fname,
        folder=EXPORT_FOLDER,
        fileFormat='GeoJSON',
    )
    task.start()
    print(f'  Запущена: {fname}')
    return task


def main():
    parser = argparse.ArgumentParser(
        description='GEE export годовых признаков для прогноза мерзлоты',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Примеры:
  python3 scripts/gee_export.py --years 2025
  python3 scripts/gee_export.py --years 2003-2025
  python3 scripts/gee_export.py --years 2024,2025
  python3 scripts/gee_export.py --years 2003-2010,2025

Для каждого года запускается 4 задачи (по поддиапазонам долготы).
Проверить статус задач: https://code.earthengine.google.com/tasks
""")
    parser.add_argument('--years', required=True,
                         help='Годы: "2025", "2003-2025", "2024,2025" или "2003-2010,2025"')
    parser.add_argument('--project', default='clean-outcome-446214-a4',
                         help='Google Cloud project ID (default: clean-outcome-446214-a4)')
    args = parser.parse_args()

    # Парсим годы
    try:
        years = parse_years(args.years)
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

    # Аутентификация GEE
    print(f"Подключение к Google Earth Engine (project: {args.project})...")
    ee.Authenticate()
    ee.Initialize(project=args.project)
    print('GEE OK')

    # Информация о задаче
    print(f'\n{"="*60}')
    print(f'GEE Export — годовые признаки 0.1° для России')
    print(f'{"="*60}')
    print(f'Шаг сетки: {STEP_DEG}°')
    print(f'Годы:      {years[0]}..{years[-1]} ({len(years)} лет)')
    if len(years) <= 10:
        print(f'           {years}')
    print(f'Поддиапазонов по долготе: {len(BANDS)}')
    print(f'Всего задач: {len(years)} × {len(BANDS)} = {len(years)*len(BANDS)}')
    print(f'Drive folder: {EXPORT_FOLDER}')
    print(f'{"="*60}\n')

    # Запуск задач
    tasks = []
    for year in years:
        print(f'\n=== ГОД {year} ===')
        for band_name, w, e in BANDS:
            try:
                t = export_year_subband(year, band_name, w, e)
                tasks.append((year, band_name, t))
            except Exception as exc:
                print(f'  ОШИБКА для {year} {band_name}: {exc}')

    # Итог
    print(f'\n{"="*60}')
    print(f'Всего запущено: {len(tasks)} задач из {len(years)*len(BANDS)} ожидаемых')
    print(f'Следите за прогрессом: https://code.earthengine.google.com/tasks')
    print(f'Файлы появятся в: Google Drive / {EXPORT_FOLDER}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
