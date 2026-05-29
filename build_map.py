"""
Построение карты мерзлоты России.

Применяет обученную модель (models_scatter_xgb.joblib) к регулярной сетке точек,
рисует карту предсказанного MAGT и карту 3 классов.

Запуск:
    python3 build_map.py
"""

from pathlib import Path
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import joblib

GRID_GEOJSON = glob.glob("data/features/RussiaGrid_*.geojson")
MODEL = "models_scatter_xgb.joblib"
REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "NDVI", "NDWI", "NDMI", "SAVI",
    "LST_summer", "LST_winter", "LST_annual", "FDD", "TDD",
    "snow_days", "elevation", "slope", "aspect",
    "soil_oc", "soil_bd", "soil_clay",
    "MAAT", "MAP", "era5_temp", "era5_precip",
]


def load_grid():
    path = GRID_GEOJSON[0]
    print(f"Читаю сетку: {path}")
    with open(path) as f:
        gj = json.load(f)
    rows, lons, lats = [], [], []
    for ft in gj["features"]:
        if ft.get("geometry") is None:
            continue
        rows.append(ft["properties"])
        lon, lat = ft["geometry"]["coordinates"]
        lons.append(lon); lats.append(lat)
    df = pd.DataFrame(rows)
    df["lon"] = lons; df["lat"] = lats
    print(f"Точек в сетке: {len(df)}")
    return df


def main():
    df = load_grid()
    model = joblib.load(MODEL)

    # Оставляем точки с полным набором фич
    n0 = len(df)
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"Точек с полными фичами: {len(df)}/{n0}")

    # Предсказание
    df["pred_magt"] = model.predict(df[FEATURE_COLS].values)

    # Классы
    def to_class(v):
        if v > 0: return 0
        if v > -1.5: return 1
        return 2
    df["pred_class"] = df["pred_magt"].apply(to_class)

    print(f"\nПредсказанный MAGT: {df['pred_magt'].min():.1f} ... {df['pred_magt'].max():.1f} °C")
    print("Распределение классов:")
    for c, name in [(0, "нет/тёплая"), (1, "деградирующая"), (2, "устойчивая")]:
        n = (df["pred_class"] == c).sum()
        print(f"  Класс {c} ({name}): {n} ({100*n/len(df):.1f}%)")

    # === Карта MAGT (непрерывная) ===
    fig, ax = plt.subplots(figsize=(16, 8))
    sc = ax.scatter(df["lon"], df["lat"], c=df["pred_magt"],
                    cmap="coolwarm_r", s=8, vmin=-12, vmax=2)
    ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
    ax.set_title("Предсказанная среднегодовая температура грунта (MAGT) — Россия, 2021")
    plt.colorbar(sc, ax=ax, label="MAGT на 10 м, °C", shrink=0.7)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(REPORTS / "russia_magt_map.png", dpi=130)
    plt.close()
    print(f"\nКарта MAGT: {REPORTS / 'russia_magt_map.png'}")

    # === Карта 3 классов ===
    fig, ax = plt.subplots(figsize=(16, 8))
    cmap3 = mcolors.ListedColormap(["#2c7bb6", "#fdae61", "#d7191c"][::-1])
    sc = ax.scatter(df["lon"], df["lat"], c=df["pred_class"],
                    cmap=cmap3, s=8, vmin=-0.5, vmax=2.5)
    ax.set_xlabel("Долгота"); ax.set_ylabel("Широта")
    ax.set_title("Классы мерзлоты — Россия, 2021")
    cbar = plt.colorbar(sc, ax=ax, ticks=[0, 1, 2], shrink=0.7)
    cbar.ax.set_yticklabels(["нет/тёплая", "деградирующая", "устойчивая"])
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(REPORTS / "russia_class_map.png", dpi=130)
    plt.close()
    print(f"Карта классов: {REPORTS / 'russia_class_map.png'}")

    # Сохраняем результат для дальнейшего (Kepler.gl и т.д.)
    df[["lon", "lat", "pred_magt", "pred_class"]].to_csv(
        REPORTS / "russia_map_data.csv", index=False)
    print(f"Данные карты: {REPORTS / 'russia_map_data.csv'}")


if __name__ == "__main__":
    main()