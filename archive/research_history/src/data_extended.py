"""
Загрузка и подготовка расширенного временного ряда 2001-2020 для ConvLSTM.

Ключевая идея: то же окно 5-1 (5 лет на вход, 1 год на выход), но теперь
у нас 20 лет данных вместо 5, что даёт 15 (input_window, target_year) пар
вместо 1. Это в 15 раз больше тренировочных данных + позволяет временной
held-out test.

Архитектурно это совместимо с существующей model.py — меняется только
DataLoader.

Разбиение времени:
    train:   target_year ∈ [2005, 2018]   (14 пар)
    val:     target_year ∈ [2019, 2020]   (2 пары)
    test:    target_year ∈ [2021, 2023]   (3 пары) — если есть данные

Использование:
    from src.data_extended import (
        load_extended_tensor,
        make_sliding_windows,
        temporal_split,
        ExtendedTensorDataset,
    )

    tensor, lats, lons, years = load_extended_tensor(DATA_DIR / 'tensor_01deg_extended.npz')
    windows = make_sliding_windows(tensor, window_size=5)
    splits = temporal_split(windows, years, train_until=2018, val_until=2020)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_extended_tensor(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает тензор (T, F, H, W), где T — все доступные годы.

    Ожидаемые ключи в npz:
        tensor: (T, F, H, W)
        lats: (H,)
        lons: (W,)
        years: (T,) — массив годов, должен быть монотонно возрастающим
    """
    data = np.load(npz_path)
    required = {'tensor', 'lats', 'lons', 'years'}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"В {npz_path} нет ключей: {missing}. Есть: {list(data.files)}")
    tensor = data['tensor']
    lats = data['lats']
    lons = data['lons']
    years = data['years']

    # Проверки целостности
    assert tensor.ndim == 4, f"Ожидался 4D тензор (T,F,H,W), получено {tensor.shape}"
    assert tensor.shape[0] == len(years), "T не совпадает с длиной years"
    assert tensor.shape[-2] == len(lats), "H не совпадает с lats"
    assert tensor.shape[-1] == len(lons), "W не совпадает с lons"
    assert np.all(np.diff(years) > 0), "years должны быть строго монотонно возрастающими"

    return tensor, lats, lons, years


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

@dataclass
class Window:
    """Одно тренировочное окно: input_years -> target_year."""
    input_years: List[int]   # например, [2014, 2015, 2016, 2017, 2018]
    target_year: int          # например, 2019
    input_indices: List[int]  # индексы по оси T тензора
    target_index: int


def make_sliding_windows(years: np.ndarray, window_size: int = 5) -> List[Window]:
    """
    Создаёт sliding окна: каждое окно — это window_size предыдущих лет + 1 год target.

    Например, для years=[2001..2020] и window_size=5:
        окно 1: input [2001-2005] -> target 2006
        окно 2: input [2002-2006] -> target 2007
        ...
        окно 15: input [2015-2019] -> target 2020

    Возвращает список Window-объектов.
    """
    windows = []
    for i in range(window_size, len(years)):
        input_indices = list(range(i - window_size, i))
        target_index = i
        windows.append(Window(
            input_years=[int(years[j]) for j in input_indices],
            target_year=int(years[target_index]),
            input_indices=input_indices,
            target_index=target_index,
        ))
    return windows


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def temporal_split(
    windows: List[Window],
    train_until: int = 2018,
    val_until: int = 2020,
) -> Dict[str, List[Window]]:
    """
    Разбивает окна на train/val/test по target_year.

        train: target_year <= train_until
        val:   train_until < target_year <= val_until
        test:  target_year > val_until
    """
    splits = {'train': [], 'val': [], 'test': []}
    for w in windows:
        if w.target_year <= train_until:
            splits['train'].append(w)
        elif w.target_year <= val_until:
            splits['val'].append(w)
        else:
            splits['test'].append(w)
    return splits


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ExtendedTensorDataset(Dataset):
    """
    Dataset для extended timeseries.

    Возвращает пары (X, y), где:
        X: тензор (window_size, F, H, W) — входное окно
        y: тензор (H, W) — целевая MAGT в target_year

    Опционально режет на тайлы (patch_size) — полезно для GPU с малой памятью.
    """

    def __init__(
        self,
        tensor: np.ndarray,
        target: np.ndarray,  # (T, H, W) — таргет на каждый год
        windows: List[Window],
        patch_size: Optional[int] = None,
        patches_per_window: int = 4,
    ):
        self.tensor = tensor
        self.target = target
        self.windows = windows
        self.patch_size = patch_size
        self.patches_per_window = patches_per_window

        if patch_size is not None:
            # каждый sample = (window_id, patch_top_y, patch_top_x)
            H, W = tensor.shape[-2], tensor.shape[-1]
            self.samples = []
            rng = np.random.default_rng(42)
            for wid in range(len(windows)):
                for _ in range(patches_per_window):
                    y0 = rng.integers(0, H - patch_size + 1)
                    x0 = rng.integers(0, W - patch_size + 1)
                    self.samples.append((wid, int(y0), int(x0)))
        else:
            self.samples = [(wid, None, None) for wid in range(len(windows))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wid, y0, x0 = self.samples[idx]
        w = self.windows[wid]
        X = self.tensor[w.input_indices, :, :, :]  # (window_size, F, H, W)
        y = self.target[w.target_index, :, :]      # (H, W)

        if y0 is not None:
            ps = self.patch_size
            X = X[:, :, y0:y0+ps, x0:x0+ps]
            y = y[y0:y0+ps, x0:x0+ps]

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Слияние extended и v2 тензоров (если v2 короткий 2018-2022)
# ---------------------------------------------------------------------------

def merge_extended_with_v2(
    extended_npz: Path,
    v2_npz: Path,
    out_npz: Path,
    feature_alignment: Optional[Dict[str, str]] = None,
):
    """
    Объединяет extended (2001-2020) и v2 (2018-2022) в один длинный тензор.

    Если есть overlap (2018-2020) — берём из extended (более полный).
    Если каналы признаков отличаются — отображаем через feature_alignment.

    Args:
        extended_npz: путь к выгрузке extended (после rasterize)
        v2_npz: путь к существующему tensor_01deg_v2.npz
        out_npz: путь сохранения объединённого
        feature_alignment: dict {v2_feature_name: extended_feature_name}, если нужно
    """
    ext = np.load(extended_npz)
    v2 = np.load(v2_npz)

    ext_years = ext['years']
    v2_years = v2.get('years', np.array([2018, 2019, 2020, 2021, 2022]))

    # Все уникальные годы
    all_years = np.unique(np.concatenate([ext_years, v2_years]))

    # Берём пространственную решётку из v2 (она каноничная)
    lats = v2['lats']
    lons = v2['lons']
    F = v2['tensor'].shape[1]
    H, W = lats.shape[0], lons.shape[0]

    merged = np.full((len(all_years), F, H, W), np.nan, dtype=np.float32)

    for i, year in enumerate(all_years):
        if year in ext_years:
            ei = int(np.where(ext_years == year)[0][0])
            # ВАЖНО: убедись, что порядок каналов совпадает
            merged[i] = ext['tensor'][ei]
        elif year in v2_years:
            vi = int(np.where(v2_years == year)[0][0])
            merged[i] = v2['tensor'][vi]

    np.savez_compressed(
        out_npz,
        tensor=merged,
        lats=lats,
        lons=lons,
        years=all_years,
    )
    print(f"Сохранено: {out_npz}")
    print(f"  Tensor: {merged.shape}, years: {all_years}")
