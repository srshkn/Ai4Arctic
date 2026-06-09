"""
Склейка 4 поддиапазонов в годовые файлы для расширенного набора (2003-2023).

Вход:  data/gee/RussiaGrid_0.1deg_v2_<year>_<band>.geojson  (84 файла)
Выход: data/gee/RussiaGrid_0.1deg_v2_<year>.geojson         (21 файл)
"""

from pathlib import Path
import json
import time

FEAT_DIR = Path("data/gee")
YEARS = list(range(2003, 2025))  # 2003..2024 (22 года)
BANDS = ["band0", "band1_west", "band1_east", "band2"]


def load_features(fpath):
    with open(fpath) as f:
        gj = json.load(f)
    return gj.get("features", [])


def main():
    if not FEAT_DIR.exists():
        print(f"ОШИБКА: папка не найдена: {FEAT_DIR}")
        return

    print(f"Сканирую {FEAT_DIR}/")
    print(f"Обрабатываю {len(YEARS)} лет (2003-2023)\n")

    missing = []
    for year in YEARS:
        for band in BANDS:
            fpath = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{year}_{band}.geojson"
            if not fpath.exists():
                missing.append(fpath.name)

    if missing:
        print(f"ВНИМАНИЕ: отсутствует {len(missing)} файлов:")
        for m in missing[:10]:
            print(f"  - {m}")
        return

    total_features = 0
    for year in YEARS:
        # Проверим, не сделан ли уже годовой файл
        out_path = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{year}.geojson"
        if out_path.exists():
            print(f"[{year}] годовой файл уже есть, пропускаю ({out_path.stat().st_size/1e6:.0f} МБ)")
            continue
        
        print(f"=== {year} ===")
        all_features = []
        for band in BANDS:
            fpath = FEAT_DIR / f"RussiaGrid_0.1deg_v2_{year}_{band}.geojson"
            t0 = time.time()
            feats = load_features(fpath)
            print(f"  {band}: {len(feats):>7} точек ({time.time() - t0:.1f}с)")
            all_features.extend(feats)

        # Дедупликация
        seen = set()
        dedup = []
        for ft in all_features:
            if ft.get("geometry") is None:
                continue
            coords = tuple(ft["geometry"]["coordinates"])
            if coords in seen:
                continue
            seen.add(coords)
            dedup.append(ft)
        n_dup = len(all_features) - len(dedup)
        if n_dup > 0:
            print(f"  дубликатов удалено: {n_dup}")

        out = {"type": "FeatureCollection", "features": dedup}
        with open(out_path, "w") as f:
            json.dump(out, f)
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"  → {out_path.name}: {len(dedup):>7} точек, {size_mb:.1f} МБ\n")
        total_features += len(dedup)

    print("=" * 50)
    print(f"ИТОГО: {total_features:,} точек по {len(YEARS)} годам")


if __name__ == "__main__":
    main()
