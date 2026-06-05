"""
Загрузка и обработка данных для обучения ConvLSTM.

Содержит:
- load_tensor      : загрузка tensor_01deg_v2.npz и его проверка
- normalize_features: стандартизация 20 признаков
- make_pairs       : нарезка на тайлы 64x64 + формирование (X_input, y_target) пар
- TileDataset      : PyTorch Dataset для обучения
- masked_mse       : функция потерь с маской невалидных пикселей
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# Константы из работы
TILE = 64      # размер тайла на обучении
STEP = 32      # шаг между тайлами (50% перекрытие)
INPUT_LEN = 4  # количество лет на вход


def load_tensor(path):
    """
    Загружает tensor_01deg_v2.npz и проверяет его структуру.

    Parameters
    ----------
    path : str
        Путь к .npz файлу с тензором.

    Returns
    -------
    dict со следующими ключами:
        X      : (14, 231, 1501, 20) — признаки
        y      : (14, 231, 1501) — target по формуле TTOP
        years  : массив лет [2010..2023]
        lons   : массив долгот (1501)
        lats   : массив широт (231)
        feature_names : список из 20 названий признаков
    """
    data = np.load(path, allow_pickle=True)
    print(f'X shape: {data["X"].shape}')
    print(f'y shape: {data["y"].shape}')
    print(f'Годы: {list(data["years"])}')
    print(f'Признаков: {len(data["feature_names"])}')
    return data


def normalize_features(X_full, y_full, train_year_indices):
    """
    Стандартизация признаков и target по обучающим годам.

    Статистика считается ТОЛЬКО по обучающим годам (без валидационного 2023),
    чтобы избежать утечки данных.

    Parameters
    ----------
    X_full : ndarray (14, H, W, 20)
        Исходный тензор признаков.
    y_full : ndarray (14, H, W)
        Исходный тензор target.
    train_year_indices : list[int]
        Индексы обучающих годов (например, range(13) для 2010..2022).

    Returns
    -------
    X_norm : ndarray (14, H, W, 20) — нормализованные признаки
    y_norm : ndarray (14, H, W) — нормализованный target
    x_mean : ndarray (20,) — среднее по каналам
    x_std  : ndarray (20,) — std по каналам
    y_mean : float — среднее target
    y_std  : float — std target
    y_nan_mask : ndarray (14, H, W) — где target неизвестен
    """
    X_train = X_full[train_year_indices]
    x_mean = np.nanmean(X_train, axis=(0, 1, 2))
    x_std = np.nanstd(X_train, axis=(0, 1, 2))
    x_std[x_std < 1e-6] = 1.0  # защита от деления на 0

    y_train = y_full[train_year_indices]
    y_mean = float(np.nanmean(y_train))
    y_std = float(np.nanstd(y_train))

    # Применяем нормализацию ко всему тензору
    X_norm = (X_full - x_mean[None, None, None, :]) / x_std[None, None, None, :]
    y_norm = (y_full - y_mean) / y_std

    # NaN → 0 после нормализации, маску храним отдельно
    X_norm = np.nan_to_num(X_norm, nan=0.0).astype(np.float32)
    y_nan_mask = np.isnan(y_norm)
    y_norm = np.nan_to_num(y_norm, nan=0.0).astype(np.float32)

    return X_norm, y_norm, x_mean, x_std, y_mean, y_std, y_nan_mask


def make_pairs(X_norm, y_norm, y_nan_mask, year_targets,
               tile=TILE, step=STEP, input_len=INPUT_LEN, min_valid_frac=0.3):
    """
    Нарезка тензора на тайлы и формирование (X_input, y_target) пар.

    Каждая пара — это:
        X: 4 года признаков, тайл 64x64x20
        y: target следующего года, тайл 64x64

    Тайлы с менее min_valid_frac валидных пикселей в target отбрасываются.

    Parameters
    ----------
    X_norm : ndarray (T, H, W, C)
        Нормализованные признаки.
    y_norm : ndarray (T, H, W)
        Нормализованный target.
    y_nan_mask : ndarray (T, H, W)
        True где target неизвестен.
    year_targets : list[int]
        Индексы target-годов (например, range(4, 13) для обучения).

    Returns
    -------
    pairs_X : ndarray (N, input_len, tile, tile, C)
    pairs_y : ndarray (N, tile, tile)
    pairs_mask : ndarray (N, tile, tile) — bool, True = валидный пиксель
    """
    H, W = X_norm.shape[1], X_norm.shape[2]
    pairs_X, pairs_y, pairs_mask = [], [], []

    for t_target in year_targets:
        t_in_start = t_target - input_len
        if t_in_start < 0:
            continue
        X_in = X_norm[t_in_start:t_target]
        y_tg = y_norm[t_target]
        mask_tg = y_nan_mask[t_target]

        for i in range(0, H - tile + 1, step):
            for j in range(0, W - tile + 1, step):
                X_tile = X_in[:, i:i+tile, j:j+tile, :]
                y_tile = y_tg[i:i+tile, j:j+tile]
                mask_tile = mask_tg[i:i+tile, j:j+tile]

                valid_frac = 1.0 - mask_tile.mean()
                if valid_frac < min_valid_frac:
                    continue

                pairs_X.append(X_tile)
                pairs_y.append(y_tile)
                pairs_mask.append(~mask_tile)

    return np.array(pairs_X), np.array(pairs_y), np.array(pairs_mask)


class TileDataset(Dataset):
    """
    PyTorch Dataset для тайлов ConvLSTM.

    Преобразует numpy-массивы (N, T, H, W, C) в torch-тензоры (N, T, C, H, W),
    как ожидает PyTorch (Channel-first).
    """

    def __init__(self, X, y, mask):
        self.X = torch.from_numpy(X.transpose(0, 1, 4, 2, 3)).float()
        self.y = torch.from_numpy(y).float()
        self.mask = torch.from_numpy(mask.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


def masked_mse(pred, target, mask):
    """
    MSE-loss с маской: считается только на валидных пикселях.

    Используется как loss function во время обучения.
    """
    diff = (pred - target) ** 2 * mask
    return diff.sum() / (mask.sum() + 1e-8)


def masked_metrics(pred, target, mask):
    """
    MAE, RMSE, R² на валидных пикселях. Работает с torch-тензорами.
    """
    valid = mask > 0.5
    if valid.sum() == 0:
        return float('nan'), float('nan'), float('nan')
    p = pred[valid]
    t = target[valid]
    mae = (p - t).abs().mean().item()
    rmse = ((p - t) ** 2).mean().sqrt().item()
    ss_res = ((p - t) ** 2).sum().item()
    ss_tot = ((t - t.mean()) ** 2).sum().item() + 1e-8
    r2 = 1 - ss_res / ss_tot
    return mae, rmse, r2
