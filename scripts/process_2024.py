"""
P2-extension: добавление 2024 года в pipeline.

Делает за один проход:
1. Merge 4 поддиапазонов 2024 в один годовой geojson
2. Расширяет X-тензор: tensor_01deg_extended.npz → tensor_01deg_extended_22y.npz
3. Расширяет target: y_new_rk_landcover_extended.npz → y_new_rk_landcover_extended_22y.npz
4. Применяет P2-модель на 2024
5. Добавляет 2024 в результаты p2_yearly_maps.npz

Вход:
  data/gee/RussiaGrid_0.1deg_v2_2024_{band}.geojson (4 файла)
  data/tensor_01deg_extended.npz                     (21 год)
  data/y_new_rk_landcover_extended.npz               (21 год)
  models/convlstm_ttop_extended_21years.pt           (P2 модель)
  data/maps_lh_v3.npz                                (delta_t)
  results/maps/p2_yearly_maps.npz                    (17 лет 2007-2023)

Выход:
  data/gee/RussiaGrid_0.1deg_v2_2024.geojson         (годовой merge)
  data/tensor_01deg_extended_22y.npz                  (22 года)
  data/y_new_rk_landcover_extended_22y.npz            (22 года)
  results/maps/p2_yearly_maps.npz                     (ПЕРЕЗАПИСАН — 18 лет 2007-2024)
"""

import json
import time
import numpy as np
import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.model import ConvLSTMNet
from src.data import normalize_features
from src.inference import predict_full_map
from src.ablation import features_by_names
from src.landcover import compute_ttop_target

DATA_DIR = BASE_DIR / 'data'
MAPS_DIR = BASE_DIR / 'results' / 'maps'
MODELS_DIR = BASE_DIR / 'models'

FEAT_DIR = DATA_DIR / 'gee'
BANDS = ['band0', 'band1_west', 'band1_east', 'band2']
YEAR = 2024
P1_SHIFT = 1.985

STEP = 0.1
LON_MIN, LON_MAX = 30.0, 180.0
LAT_MIN, LAT_MAX = 55.0, 78.0

FEATURE_COLS = [
    'NDVI', 'NDWI', 'NDMI', 'SAVI',
    'LST_summer', 'LST_winter', 'LST_annual', 'FDD', 'TDD',
    'snow_days', 'elevation', 'slope', 'aspect',
    'soil_oc', 'soil_bd', 'soil_clay',
    'MAAT', 'MAP', 'era5_temp', 'era5_precip',
]


def step1_merge_2024():
    """Шаг 1: склейка 4 поддиапазонов 2024."""
    out_path = FEAT_DIR / f'RussiaGrid_0.1deg_v2_{YEAR}.geojson'
    if out_path.exists():
        print(f"[Шаг 1] Годовой файл {YEAR} уже есть, пропускаю")
        return out_path

    print(f"[Шаг 1] Merge 4 поддиапазонов {YEAR}...")
    all_features = []
    for band in BANDS:
        fpath = FEAT_DIR / f'RussiaGrid_0.1deg_v2_{YEAR}_{band}.geojson'
        if not fpath.exists():
            raise FileNotFoundError(f"Нет файла: {fpath}")
        t0 = time.time()
        with open(fpath) as f:
            gj = json.load(f)
        feats = gj.get('features', [])
        print(f"    {band}: {len(feats):,} точек ({time.time()-t0:.1f}с)")
        all_features.extend(feats)

    # Дедупликация
    seen = set()
    dedup = []
    for ft in all_features:
        if ft.get('geometry') is None:
            continue
        coords = tuple(ft['geometry']['coordinates'])
        if coords in seen:
            continue
        seen.add(coords)
        dedup.append(ft)
    print(f"    дубликатов удалено: {len(all_features) - len(dedup)}")

    out = {'type': 'FeatureCollection', 'features': dedup}
    with open(out_path, 'w') as f:
        json.dump(out, f)
    print(f"    → {out_path.name}: {len(dedup):,} точек, "
          f"{out_path.stat().st_size/1024/1024:.1f} МБ")
    return out_path


def step2_extend_tensor():
    """Шаг 2: добавить 2024 к X-тензору."""
    out_tensor = DATA_DIR / 'tensor_01deg_extended_22y.npz'

    print(f"\n[Шаг 2] Расширяем X-тензор 21→22 года...")
    # Загружаем существующий 21-летний тензор
    tensor_21 = np.load(DATA_DIR / 'tensor_01deg_extended.npz')
    X_21 = tensor_21['X']
    years_21 = list(tensor_21['years'])
    lats = tensor_21['lats']
    lons = tensor_21['lons']

    print(f"    Существующий X: {X_21.shape}, годы {years_21[0]}..{years_21[-1]}")

    # Растеризуем 2024
    fpath = FEAT_DIR / f'RussiaGrid_0.1deg_v2_{YEAR}.geojson'
    H, W = len(lats), len(lons)
    n_feat = len(FEATURE_COLS)
    X_2024 = np.full((H, W, n_feat), np.nan, dtype=np.float32)

    print(f"    Растеризуем {fpath.name}...")
    t0 = time.time()
    with open(fpath) as f:
        gj = json.load(f)

    n_ok = 0
    for ft in gj['features']:
        geom = ft.get('geometry')
        if geom is None:
            continue
        lo, la = geom['coordinates']
        j = round((lo - lons[0]) / STEP)
        i = round((la - lats[0]) / STEP)
        if 0 <= i < H and 0 <= j < W:
            props = ft['properties']
            for k, col in enumerate(FEATURE_COLS):
                v = props.get(col)
                if v is not None:
                    X_2024[i, j, k] = v
            n_ok += 1
    print(f"    Записано точек: {n_ok:,} за {time.time()-t0:.1f}с")

    # Конкатенация
    X_22 = np.concatenate([X_21, X_2024[np.newaxis]], axis=0)
    years_22 = years_21 + [YEAR]
    print(f"    Новый X: {X_22.shape}, годы {years_22[0]}..{years_22[-1]}")

    np.savez_compressed(
        out_tensor,
        X=X_22,
        lons=lons, lats=lats,
        years=np.array(years_22),
        feature_names=np.array(FEATURE_COLS),
    )
    print(f"    Сохранено: {out_tensor.name} ({out_tensor.stat().st_size/1e6:.0f} МБ)")
    return out_tensor, X_22, years_22, lats, lons


def step3_extend_target(X_22, years_22, lats, lons):
    """Шаг 3: расширить target до 22 лет."""
    out_target = DATA_DIR / 'y_new_rk_landcover_extended_22y.npz'

    print(f"\n[Шаг 3] Пересчёт target для 22 лет...")
    y_old_npz = np.load(DATA_DIR / 'y_new_rk_landcover.npz')
    landcover = y_old_npz['landcover']
    rk_map = y_old_npz['rk_map']

    y_22 = compute_ttop_target(X_22, landcover, fdd_idx=7, tdd_idx=8)
    print(f"    y_22: {y_22.shape}")
    print(f"    Медиана 2024: {np.nanmedian(y_22[-1]):+.2f}°C "
          f"(сравним: 2023 = {np.nanmedian(y_22[-2]):+.2f})")

    np.savez_compressed(
        out_target,
        y_new=y_22.astype(np.float32),
        landcover=landcover,
        rk_map=rk_map,
        years=np.array(years_22),
        lats=lats, lons=lons,
    )
    print(f"    Сохранено: {out_target.name} ({out_target.stat().st_size/1e6:.1f} МБ)")
    return out_target, y_22


def step4_inference_2024(X_22, y_22, years_22, lats, lons):
    """Шаг 4: inference модели на 2024 + lh + shift."""
    print(f"\n[Шаг 4] Inference P2 модели на 2024...")

    device = ('cuda' if torch.cuda.is_available()
              else 'mps' if torch.backends.mps.is_available()
              else 'cpu')
    print(f"    Device: {device}")

    # Модель
    ckpt = torch.load(MODELS_DIR / 'convlstm_ttop_extended_21years.pt',
                       map_location=device, weights_only=False)
    model = ConvLSTMNet(in_ch=6).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']
    core_idx = features_by_names(CORE_6)
    X_core = X_22[..., core_idx]

    # Нормализация по train years (как в P2: range(19) для 21-year tensor)
    # Но теперь tensor стал 22 года, train_idx должны быть теми же годами 2003-2021
    train_idx = list(range(19))  # 2003..2021
    X_n, _, _, _, _, _, y_nm = normalize_features(X_core, y_22, train_idx)

    # Inference на 2024 (последний год, index = 21)
    t_target = 21
    pred_raw = predict_full_map(
        model, X_n, y_nm, t_target=t_target,
        y_mean=ckpt['y_mean'], y_std=ckpt['y_std'],
        device=device,
    )

    # lh + shift
    lh = np.load(DATA_DIR / 'maps_lh_v3.npz')
    delta_t = lh['delta_t']
    pred_lh = pred_raw + delta_t
    pred_final = pred_lh + P1_SHIFT

    print(f"    pred_2024 raw:  median {np.nanmedian(pred_raw):+.2f}, min {np.nanmin(pred_raw):.2f}")
    print(f"    pred_2024 final: median {np.nanmedian(pred_final):+.2f}, min {np.nanmin(pred_final):.2f}, max {np.nanmax(pred_final):.2f}")

    return pred_final.astype(np.float32), pred_raw.astype(np.float32)


def step5_update_yearly_maps(pred_2024_final, pred_2024_raw, lats, lons):
    """Шаг 5: добавить 2024 в p2_yearly_maps.npz."""
    out_maps = MAPS_DIR / 'p2_yearly_maps.npz'

    print(f"\n[Шаг 5] Добавляем 2024 в {out_maps.name}...")
    existing = np.load(out_maps)
    magt_17 = existing['magt_yearly']  # (17, 231, 1501) для 2007-2023
    raw_17 = existing['magt_raw']
    years_17 = list(existing['years'])

    print(f"    Существующих годов: {len(years_17)} ({years_17[0]}..{years_17[-1]})")

    # Добавляем 2024
    magt_18 = np.concatenate([magt_17, pred_2024_final[np.newaxis]], axis=0)
    raw_18 = np.concatenate([raw_17, pred_2024_raw[np.newaxis]], axis=0)
    years_18 = years_17 + [YEAR]

    print(f"    Новый набор: {len(years_18)} годов ({years_18[0]}..{years_18[-1]})")

    np.savez_compressed(
        out_maps,
        magt_yearly=magt_18,
        magt_raw=raw_18,
        years=np.array(years_18),
        lats=lats, lons=lons,
        intercept=P1_SHIFT,
        description='P2 model inference for 2007-2024, with lh correction and pure shift',
    )
    print(f"    Сохранено: {out_maps} ({out_maps.stat().st_size/1e6:.1f} МБ)")

    # Сводный тренд по годам
    print(f"\n=== Полный тренд медианы MAGT 2007-2024 ===")
    for y, m in zip(years_18, magt_18):
        print(f"  {y}: {float(np.nanmedian(m)):+.2f}°C")


def main():
    print(f"=== Pipeline для добавления {YEAR} ===\n")
    step1_merge_2024()
    _, X_22, years_22, lats, lons = step2_extend_tensor()
    _, y_22 = step3_extend_target(X_22, years_22, lats, lons)
    pred_final, pred_raw = step4_inference_2024(X_22, y_22, years_22, lats, lons)
    step5_update_yearly_maps(pred_final, pred_raw, lats, lons)
    print("\n=== Готово ===")
    print("Следующий шаг: python3 scripts/plot_yearly_trend_p2.py")


if __name__ == '__main__':
    main()
