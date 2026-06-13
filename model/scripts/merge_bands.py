"""
Универсальное объединение GEE-выгрузок: 4 поддиапазона → 1 годовой geojson.

Заменяет:
  - merge_bands_extended.py  (был YEARS = 2003..2024)
  - merge_bands_2025.py       (был YEAR = 2025)

Вход:  data/gee/RussiaGrid_0.1deg_v2_<year>_<band>.geojson  (4 файла на год)
Выход: data/gee/RussiaGrid_0.1deg_v2_<year>.geojson         (1 файл на год)

Использование:
  # Один год
  python3 scripts/merge_bands.py --years 2025

  # Диапазон
  python3 scripts/merge_bands.py --years 2003-2025

  # Несколько отдельных годов
  python3 scripts/merge_bands.py --years 2024,2025

  # С перезаписью существующих годовых файлов
  python3 scripts/merge_bands.py --years 2025 --overwrite

Что делает:
  1. Проверяет наличие всех 4 поддиапазонов для каждого года
  2. Загружает features из каждого файла (band0, band1_west, band1_east, band2)
  3. Объединяет в один FeatureCollection
  4. Сохраняет как годовой geojson (~150 МБ на год)
"""

import argparse
import json
import sys
import time
from pathlib import Path


# ===== Константы pipeline =====
FEAT_DIR = Path("data/gee")
BANDS = ["band0", "band1_west", "band1_east", "band2"]


def parse_years(years_str):
    """Парсит --years в список годов (см. gee_export.py)."""
    years = []
    for part in years_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            if start > end:
                raise ValueError(f"Неверный диапазон {part}: start > end")
            years.extend(range(start, end + 1))
        else:
            years.append(int(part))
    years = sorted(set(years))
    if not years:
        raise ValueError("Не указано ни одного года")
    return years


def load_features(fpath):
    """Загружает features из одного geojson файла."""
    with open(fpath) as f:
        gj = json.load(f)
    return gj.get("features", [])


def check_all_files_exist(years):
    """Проверяет наличие всех 4 поддиапазонов для каждого года."""
    missing = []
    for year in years:
        for band in BANDS:
            fpath = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{year}_{band}.geojson"
            if not fpath.exists():
                missing.append((year, band, fpath.name))
    return missing


def merge_one_year(year, overwrite=False):
    """
    Объединяет 4 поддиапазона одного года в единый geojson.
    
    Returns:
        (success: bool, message: str)
    """
    # Проверка выходного файла
    out_path = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{year}.geojson"
    if out_path.exists() and not overwrite:
        size_mb = out_path.stat().st_size / 1e6
        return True, f"уже есть, пропускаю ({size_mb:.0f} МБ). Используй --overwrite для перезаписи"

    # Загружаем features из всех 4 поддиапазонов
    all_features = []
    t0 = time.time()
    band_stats = []

    for band in BANDS:
        fpath = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{year}_{band}.geojson"
        if not fpath.exists():
            return False, f"отсутствует файл {fpath.name}"
        try:
            feats = load_features(fpath)
            band_stats.append((band, len(feats)))
            all_features.extend(feats)
        except json.JSONDecodeError as e:
            return False, f"некорректный JSON в {fpath.name}: {e}"

    if not all_features:
        return False, "не загружено ни одной точки"

    # Объединённый geojson
    out_gj = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    with open(out_path, 'w') as f:
        json.dump(out_gj, f)

    elapsed = time.time() - t0
    out_size_mb = out_path.stat().st_size / 1e6

    # Детальная статистика
    stats_str = ", ".join([f"{b}={n:,}" for b, n in band_stats])
    return True, (
        f"{len(all_features):,} точек ({stats_str})\n"
        f"      сохранён: {out_path.name} ({out_size_mb:.0f} МБ, время {elapsed:.1f}с)"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Объединение 4 поддиапазонов GEE-выгрузки в годовые файлы',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Примеры:
  python3 scripts/merge_bands.py --years 2025
  python3 scripts/merge_bands.py --years 2003-2025
  python3 scripts/merge_bands.py --years 2024,2025
  python3 scripts/merge_bands.py --years 2003-2025 --overwrite

Перед запуском убедись:
  1. Все файлы выгружены из GEE (см. gee_export.py)
  2. Файлы скачаны из Google Drive в data/gee/
  3. Существуют data/gee/RussiaGrid_0.1deg_v2_<year>_<band>.geojson
""")
    parser.add_argument('--years', required=True,
                         help='Годы: "2025", "2003-2025", "2024,2025" или "2003-2010,2025"')
    parser.add_argument('--overwrite', action='store_true',
                         help='Перезаписать существующие годовые файлы')
    args = parser.parse_args()

    # Парсим годы
    try:
        years = parse_years(args.years)
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)

    # Проверка существования папки
    if not FEAT_DIR.exists():
        print(f"ОШИБКА: папка не найдена: {FEAT_DIR}")
        print(f"  Сначала запусти gee_export.py и скачай файлы из Google Drive в эту папку")
        sys.exit(1)

    # Информация
    print(f'\n{"="*60}')
    print(f'Merge GEE поддиапазонов в годовые файлы')
    print(f'{"="*60}')
    print(f'Папка:      {FEAT_DIR}/')
    print(f'Годы:       {years[0]}..{years[-1]} ({len(years)} лет)')
    if len(years) <= 10:
        print(f'            {years}')
    print(f'Поддиапазонов: {len(BANDS)} ({", ".join(BANDS)})')
    print(f'Overwrite:  {"да" if args.overwrite else "нет (пропуск существующих)"}')
    print(f'{"="*60}\n')

    # Этап 1: проверка всех файлов
    print('[Этап 1] Проверка наличия всех 4 поддиапазонов...')
    missing = check_all_files_exist(years)
    if missing:
        print(f'\nОШИБКА: отсутствует {len(missing)} файлов:')
        for year, band, fname in missing[:10]:
            print(f'  - {fname}')
        if len(missing) > 10:
            print(f'  ... и ещё {len(missing) - 10} файлов')
        print('\nЛибо запусти GEE export, либо проверь что файлы в правильной папке.')
        sys.exit(1)
    print(f'  ✓ Все {len(years) * len(BANDS)} файлов на месте\n')

    # Этап 2: merge по годам
    print(f'[Этап 2] Объединение по годам...\n')
    succeeded = 0
    failed = 0
    skipped = 0

    for i, year in enumerate(years, 1):
        ok, msg = merge_one_year(year, overwrite=args.overwrite)
        if not ok:
            print(f'  [{i}/{len(years)}] {year}: ❌ {msg}')
            failed += 1
        elif "уже есть" in msg:
            print(f'  [{i}/{len(years)}] {year}: ⊘ {msg}')
            skipped += 1
        else:
            print(f'  [{i}/{len(years)}] {year}: ✓ {msg}')
            succeeded += 1

    # Итог
    print(f'\n{"="*60}')
    print(f'Итого:')
    print(f'  ✓ Создано:    {succeeded}')
    print(f'  ⊘ Пропущено:  {skipped}')
    print(f'  ❌ Ошибок:     {failed}')
    print(f'{"="*60}')

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
