"""
Склейка долготных полос (band0/1/2) в единый файл за год.

После выгрузки по полосам имеем файлы вида:
  RussiaGrid_0.25deg_2003_band0.geojson
  RussiaGrid_0.25deg_2003_band1.geojson
  RussiaGrid_0.25deg_2003_band2.geojson
Склеиваем в RussiaGrid_0.25deg_2003.geojson

Запуск:
    python3 merge_bands.py
"""

from pathlib import Path
import glob
import json

FEAT_DIR = Path("data/features")
YEARS = [2003, 2010, 2013, 2015, 2017, 2019, 2021, 2023]
STEP = "0.25"


def merge_year(year):
    pattern = str(FEAT_DIR / f"RussiaGrid_{STEP}deg_{year}_band*.geojson")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  {year}: полос не найдено, пропуск")
        return
    print(f"  {year}: склеиваю {len(files)} полос")
    all_features = []
    for f in files:
        with open(f) as fh:
            gj = json.load(fh)
        all_features.extend(gj["features"])
        print(f"    {Path(f).name}: {len(gj['features'])} точек")

    merged = {"type": "FeatureCollection", "features": all_features}
    out = FEAT_DIR / f"RussiaGrid_{STEP}deg_{year}.geojson"
    with open(out, "w") as fh:
        json.dump(merged, fh)
    print(f"    -> {out.name}: {len(all_features)} точек всего")


def main():
    print("Склейка полос по годам:")
    for y in YEARS:
        merge_year(y)
    print("\nГотово. Проверьте итоговые файлы RussiaGrid_0.25deg_<год>.geojson")


if __name__ == "__main__":
    main()