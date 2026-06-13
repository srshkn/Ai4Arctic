"""
Merge 4 поддиапазонов 2025 года в один годовой geojson.

Вход:  data/gee/RussiaGrid_0.1deg_v2_2025_<band>.geojson  (4 файла)
Выход: data/gee/RussiaGrid_0.1deg_v2_2025.geojson         (1 файл, ~140 МБ)

Минимальная версия — только для 2025, чтобы не трогать рабочий merge_bands_extended.py.
"""
from pathlib import Path
import json
import time

FEAT_DIR = Path("data/gee")
YEAR = 2025
BANDS = ["band0", "band1_west", "band1_east", "band2"]


def load_features(fpath):
    with open(fpath) as f:
        gj = json.load(f)
    return gj.get("features", [])


def main():
    if not FEAT_DIR.exists():
        print(f"ОШИБКА: папка не найдена: {FEAT_DIR}")
        return

    print(f"Merge 4 поддиапазонов 2025 года")
    print(f"Папка: {FEAT_DIR}/\n")

    # Проверка наличия всех 4 файлов
    missing = []
    for band in BANDS:
        fpath = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{YEAR}_{band}.geojson"
        if not fpath.exists():
            missing.append(fpath.name)
        else:
            size_mb = fpath.stat().st_size / 1e6
            print(f"  [OK] {fpath.name} ({size_mb:.1f} МБ)")

    if missing:
        print(f"\nОШИБКА: отсутствует {len(missing)} файлов:")
        for m in missing:
            print(f"  - {m}")
        return

    # Проверка что годовой уже не существует
    out_path = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{YEAR}.geojson"
    if out_path.exists():
        print(f"\nВНИМАНИЕ: {out_path.name} уже существует ({out_path.stat().st_size/1e6:.0f} МБ)")
        ans = input("Перезаписать? (y/N): ").strip().lower()
        if ans != 'y':
            print("Отмена.")
            return

    # Объединение features
    print(f"\nОбъединение features из 4 поддиапазонов...")
    all_features = []
    t0 = time.time()

    for band in BANDS:
        fpath = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{YEAR}_{band}.geojson"
        feats = load_features(fpath)
        print(f"  {band}: {len(feats):,} точек")
        all_features.extend(feats)

    print(f"\nВсего точек: {len(all_features):,}")

    # Сохранение
    out_gj = {
        "type": "FeatureCollection",
        "features": all_features,
    }
    with open(out_path, 'w') as f:
        json.dump(out_gj, f)

    elapsed = time.time() - t0
    out_size_mb = out_path.stat().st_size / 1e6
    print(f"\nСохранено: {out_path.name}")
    print(f"  Размер: {out_size_mb:.1f} МБ")
    print(f"  Время: {elapsed:.1f} сек")


if __name__ == '__main__':
    main()
