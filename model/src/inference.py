"""
Inference и MC Dropout uncertainty estimation.

Содержит:
- predict_full_map: tile-based прогноз для одного года с усреднением по перекрытиям
- enable_dropout : включает Dropout в режиме eval (для MC Dropout)
- mc_dropout_predict: запускает N прогонов модели и возвращает mean + std карты
"""

import numpy as np
import torch
import torch.nn as nn

from .data import TILE, STEP, INPUT_LEN


def predict_full_map(model, X_norm, y_nan_mask, t_target,
                     y_mean, y_std, device='cuda',
                     tile=TILE, step=STEP, input_len=INPUT_LEN):
    """
    Tile-based прогноз для одного года.

    Карта 231x1501 разбивается на тайлы 64x64 с шагом 32 (50% перекрытие).
    Прогнозы суммируются и усредняются по перекрытиям — это даёт эффект
    мини-ансамбля и устраняет artifacts на границах тайлов.

    Parameters
    ----------
    model : ConvLSTMNet
        Обученная модель в режиме eval.
    X_norm : ndarray (T, H, W, C)
        Нормализованные признаки.
    y_nan_mask : ndarray (T, H, W)
        True где target неизвестен (вне России).
    t_target : int
        Индекс целевого года (например, 13 для 2023).
    y_mean, y_std : float
        Статистики для денормализации обратно в °C.
    device : str
        'cuda' или 'cpu'.

    Returns
    -------
    pred_celsius : ndarray (H, W)
        Прогноз MAGT в °C с маской NaN вне России.
    """
    t_in_start = t_target - input_len
    H, W = X_norm.shape[1], X_norm.shape[2]
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, H - tile + 1, step):
            for j in range(0, W - tile + 1, step):
                X_tile = X_norm[t_in_start:t_target, i:i+tile, j:j+tile, :]
                X_t = (torch.from_numpy(X_tile.transpose(0, 3, 1, 2))
                       .float().unsqueeze(0).to(device))
                p = model(X_t).cpu().numpy()[0]
                pred_sum[i:i+tile, j:j+tile] += p
                pred_cnt[i:i+tile, j:j+tile] += 1

    pred_norm = pred_sum / np.maximum(pred_cnt, 1)
    pred_celsius = pred_norm * y_std + y_mean

    # Маскируем где target неизвестен
    pred_celsius = np.where(y_nan_mask[t_target], np.nan, pred_celsius)
    return pred_celsius


def predict_year_extended(model, X_norm, t_target,
                          y_mean, y_std, device='cuda',
                          tile=TILE, step=STEP, input_len=INPUT_LEN):
    """
    Прогноз для года, для которого может не быть y_nan_mask (например, 2024).

    Использует расширенный X_norm с дублированием последнего года.
    """
    H, W = X_norm.shape[1], X_norm.shape[2]
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)

    t_in_start = t_target - input_len
    model.eval()
    with torch.no_grad():
        for i in range(0, H - tile + 1, step):
            for j in range(0, W - tile + 1, step):
                X_tile = X_norm[t_in_start:t_target, i:i+tile, j:j+tile, :]
                X_t = (torch.from_numpy(X_tile.transpose(0, 3, 1, 2))
                       .float().unsqueeze(0).to(device))
                p = model(X_t).cpu().numpy()[0]
                pred_sum[i:i+tile, j:j+tile] += p
                pred_cnt[i:i+tile, j:j+tile] += 1

    pred_norm = pred_sum / np.maximum(pred_cnt, 1)
    return pred_norm * y_std + y_mean


def enable_dropout(model):
    """
    Включает Dropout-слои в режиме eval. Необходимо для MC Dropout uncertainty:
    обычно при eval() Dropout выключается, но для оценки неопределённости
    его нужно держать активным.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()


def mc_dropout_predict(model, X_norm, y_nan_mask, t_target,
                       y_mean, y_std, device='cuda',
                       n_samples=30, verbose=True,
                       tile=TILE, step=STEP, input_len=INPUT_LEN):
    """
    Monte Carlo Dropout: запускает N прогонов модели с активным Dropout
    и возвращает среднее, std и доверительные интервалы.

    Parameters
    ----------
    n_samples : int
        Количество прогонов (по умолчанию 30 — компромисс между точностью и временем).

    Returns
    -------
    dict со следующими ключами:
        mean_pred : (H, W) — среднее по N прогонам в °C
        std_pred  : (H, W) — std по N прогонам (неопределённость) в °C
        ci_low    : (H, W) — 2.5 перцентиль (нижняя граница 95% CI)
        ci_high   : (H, W) — 97.5 перцентиль
        all_preds : (N, H, W) — все N прогонов (опционально для анализа)
    """
    H, W = X_norm.shape[1], X_norm.shape[2]
    all_preds = np.zeros((n_samples, H, W), dtype=np.float32)
    t_in_start = t_target - input_len

    for sample_idx in range(n_samples):
        model.eval()
        enable_dropout(model)  # активируем Dropout

        pred_sum = np.zeros((H, W), dtype=np.float32)
        pred_cnt = np.zeros((H, W), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, H - tile + 1, step):
                for j in range(0, W - tile + 1, step):
                    X_tile = X_norm[t_in_start:t_target, i:i+tile, j:j+tile, :]
                    X_t = (torch.from_numpy(X_tile.transpose(0, 3, 1, 2))
                           .float().unsqueeze(0).to(device))
                    p = model(X_t).cpu().numpy()[0]
                    pred_sum[i:i+tile, j:j+tile] += p
                    pred_cnt[i:i+tile, j:j+tile] += 1
        pred_norm = pred_sum / np.maximum(pred_cnt, 1)
        all_preds[sample_idx] = pred_norm * y_std + y_mean

        if verbose and (sample_idx + 1) % 5 == 0:
            print(f'  Прогон {sample_idx + 1}/{n_samples} готов')

    # Маскируем вне России
    real_mask = y_nan_mask[t_target]
    all_preds = np.where(real_mask[None, ...], np.nan, all_preds)

    return {
        'mean_pred': np.nanmean(all_preds, axis=0),
        'std_pred': np.nanstd(all_preds, axis=0),
        'ci_low': np.nanpercentile(all_preds, 2.5, axis=0),
        'ci_high': np.nanpercentile(all_preds, 97.5, axis=0),
        'all_preds': all_preds,
        'n_samples': n_samples,
    }
