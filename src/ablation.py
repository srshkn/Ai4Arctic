"""
Feature ablation: проверка, нужны ли все 20 признаков.

Подход:
    1. Определяем 5-7 наборов фичей разного размера/состава
    2. Обучаем ConvLSTM на каждом наборе с урезанным бюджетом (30 эпох, ES)
    3. Сравниваем RMSE/R²/Pearson на одном и том же validation
    4. Выбираем «маленькую» конфигурацию в пределах +0.2°C от baseline
    5. Финальную модель переобучаем на полном бюджете эпох

Названия фичей и их индексы должны соответствовать каналам в тензоре.
Адаптируй FEATURE_NAMES под свой actual tensor (порядок каналов).

Использование:
    from src.ablation import FEATURE_GROUPS, run_ablation_study, plot_ablation_results

    results_df = run_ablation_study(
        tensor_path=DATA_DIR / 'tensor_01deg_v2.npz',
        target_path=DATA_DIR / 'y_new_rk_landcover.npz',
        feature_groups=FEATURE_GROUPS,
        train_years=range(2018, 2022),
        val_years=[2022],
        epochs=30,
        device='cuda',
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Каноническое имя -> индекс канала в тензоре
# ---------------------------------------------------------------------------
# ВАЖНО: проверь порядок каналов в своём tensor_01deg_v2.npz и сверь с этим
# списком. Если порядок другой — поменяй индексы здесь.

FEATURE_NAMES: List[str] = [
    'NDVI',         # 0
    'NDWI',         # 1
    'NDMI',         # 2
    'SAVI',         # 3
    'LST_summer',   # 4
    'LST_winter',   # 5
    'LST_annual',   # 6
    'FDD',          # 7
    'TDD',          # 8
    'snow_days',    # 9
    'elevation',    # 10
    'slope',        # 11
    'aspect',       # 12
    'soil_oc',      # 13
    'soil_bd',      # 14
    'soil_clay',    # 15
    'MAAT',         # 16
    'MAP',          # 17
    'era5_temp',    # 18
    'era5_precip',  # 19
]


def features_by_names(names: Sequence[str]) -> List[int]:
    """Конвертирует имена фичей в индексы каналов."""
    indices = []
    for n in names:
        if n not in FEATURE_NAMES:
            raise KeyError(f"Признак '{n}' не найден в FEATURE_NAMES. "
                          f"Доступные: {FEATURE_NAMES}")
        indices.append(FEATURE_NAMES.index(n))
    return indices


# ---------------------------------------------------------------------------
# Наборы фичей для ablation
# ---------------------------------------------------------------------------
FEATURE_GROUPS: Dict[str, List[str]] = {
    # 1. Baseline — все 20 фичей
    'all_20': list(FEATURE_NAMES),

    # 2. Climate-core (6) — топ температурных, проверка гипотезы о достаточности
    'climate_core_6': ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp'],

    # 3. Climate + wetness (8) — climate + индексы растительности/воды
    'climate_wet_8': ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp',
                       'NDVI', 'NDWI'],

    # 4. Climate + soil (9) — climate + snow + soil_oc + MAP
    'climate_soil_9': ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp',
                        'snow_days', 'soil_oc', 'MAP'],

    # 5. Compact (12) — без рельефа (elevation/slope/aspect) и избыточных индексов
    'compact_12': ['MAAT', 'LST_summer', 'LST_winter', 'LST_annual', 'era5_temp',
                    'TDD', 'FDD', 'NDVI', 'NDWI', 'snow_days', 'MAP', 'soil_oc'],

    # 6. No-soil (17) — гипотеза, что почвенные параметры не нужны на 0.1°
    'no_soil_17': [n for n in FEATURE_NAMES if n not in ['soil_oc', 'soil_bd', 'soil_clay']],

    # 7. No-terrain (17) — проверка вклада рельефа (slope/aspect)
    'no_terrain_17': [n for n in FEATURE_NAMES if n not in ['elevation', 'slope', 'aspect']],
}


# ---------------------------------------------------------------------------
# Утилиты тензора
# ---------------------------------------------------------------------------

def subset_features(tensor: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Подвыборка каналов из тензора фичей.

    Args:
        tensor: shape (T, F, H, W) или (N, T, F, H, W) — нужно знать axis F
        indices: список индексов каналов

    Returns:
        тензор с уменьшенным числом каналов
    """
    # Определяем axis канала: F идёт сразу после T (или N, T)
    if tensor.ndim == 4:  # (T, F, H, W)
        return tensor[:, indices, :, :]
    elif tensor.ndim == 5:  # (N, T, F, H, W)
        return tensor[:, :, indices, :, :]
    else:
        raise ValueError(f"Неожиданная размерность тензора: {tensor.shape}")


# ---------------------------------------------------------------------------
# Тренировочный пайплайн для ablation
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    group_name: str
    n_features: int
    feature_names: List[str]
    best_epoch: int
    train_rmse: float
    val_rmse: float
    val_bias: float
    val_r2: float
    val_pearson: float
    train_time_sec: float


def train_one_config(
    model_class,
    X_raw: np.ndarray,        # (T, H, W, F)
    y_raw: np.ndarray,        # (T, H, W) — target в исходных °C
    feature_indices: List[int],
    train_targets: List[int],
    val_targets: List[int],
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 8,
    device: str = 'cuda',
    early_stop_patience: int = 8,
    verbose: bool = True,
    input_len: int = 4,
    tile: int = 64,
    step: int = 32,
) -> Dict:
    """
    Обучает одну конфигурацию ablation.

    Args:
        X_raw: сырой тензор (T, H, W, F) — формат как в tensor_01deg_v2.npz
        y_raw: target (T, H, W) — TTOP/MAGT target в исходных °C
        feature_indices: индексы каналов для подвыборки
        train_targets, val_targets: индексы целевых лет для train/val
        epochs, lr, batch_size, early_stop_patience, input_len, tile, step:
            гиперпараметры обучения

    Returns:
        dict с метриками и историей
    """
    from torch.utils.data import DataLoader, TensorDataset
    from src.data import normalize_features, make_pairs, TileDataset

    # 1. Подвыборка каналов из сырого тензора по последней оси F
    X_sub = X_raw[..., feature_indices]
    n_features = len(feature_indices)

    # 2. Нормализация (по тренировочным годам, без утечки данных)
    #    Твоя normalize_features принимает X и y вместе и возвращает 7 значений
    train_year_indices = list(range(max(train_targets) + 1))
    X_n, y_n, x_mean, x_std, y_mean, y_std, y_nm = normalize_features(
        X_sub, y_raw, train_year_indices
    )

    # 3. Нарезка на тайлы
    Xtr, ytr, mtr = make_pairs(X_n, y_n, y_nm, train_targets,
                                input_len=input_len, tile=tile, step=step)
    Xv, yv, mv = make_pairs(X_n, y_n, y_nm, val_targets,
                             input_len=input_len, tile=tile, step=step)

    if verbose:
        print(f"    Train pairs: {Xtr.shape[0]}, Val pairs: {Xv.shape[0]}")

    train_loader = DataLoader(TileDataset(Xtr, ytr, mtr),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TileDataset(Xv, yv, mv),
                            batch_size=batch_size, shuffle=False)

    # 4. Модель + обучение
    model = model_class(in_ch=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from src.data import masked_mse, masked_metrics

    best_val = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_rmse_phys': []}

    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for xb, yb, mb in train_loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
                pred = model(xb)
                val_losses.append(masked_mse(pred, yb, mb).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        # RMSE в физических единицах °C
        val_rmse_phys = float(np.sqrt(val_loss)) * y_std
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse_phys'].append(val_rmse_phys)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch}, best={best_epoch}, "
                          f"val_rmse={val_rmse_phys:.3f}°C")
                break

        if verbose and epoch % 5 == 0:
            print(f"    Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
                  f"val_rmse={val_rmse_phys:.3f}°C")

    train_time = time.time() - t_start

    # Финальные метрики (в физических единицах)
    best_val_rmse = float(np.sqrt(best_val)) * y_std
    return {
        'val_rmse': best_val_rmse,
        'val_loss': best_val,
        'best_epoch': best_epoch,
        'train_time_sec': train_time,
        'history': history,
        'model_state': best_state,
        'y_mean': y_mean,
        'y_std': y_std,
    }


def run_ablation_study(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    model_class,
    feature_groups: Dict[str, List[str]] = None,
    train_targets: List[int] = None,
    val_targets: List[int] = None,
    epochs: int = 30,
    device: str = 'cuda',
    save_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Запускает ablation по всем конфигурациям, возвращает DataFrame с метриками.

    Args:
        X_raw: сырой тензор (T, H, W, F) из tensor_01deg_v2.npz
        y_raw: target (T, H, W) — y_new из y_new_rk_landcover.npz
        model_class: класс модели (ConvLSTMNet)
        feature_groups: dict с наборами фичей; по умолчанию FEATURE_GROUPS
        train_targets, val_targets: индексы лет (по умолчанию train=[4..12], val=[13])
        epochs: бюджет на конфигурацию (для ablation хватает 30)
        save_dir: куда сохранять чекпоинты (если не None)
    """
    if feature_groups is None:
        feature_groups = FEATURE_GROUPS

    if train_targets is None:
        train_targets = list(range(4, 13))  # 2014..2022
    if val_targets is None:
        val_targets = [13]                  # 2023

    rows = []
    for name, feat_names in feature_groups.items():
        print(f"\n{'='*60}\nКонфигурация: {name} ({len(feat_names)} фичей)")
        feat_idx = features_by_names(feat_names)

        result = train_one_config(
            model_class=model_class,
            X_raw=X_raw,
            y_raw=y_raw,
            feature_indices=feat_idx,
            train_targets=train_targets,
            val_targets=val_targets,
            epochs=epochs,
            device=device,
        )

        rows.append({
            'group': name,
            'n_features': len(feat_names),
            'features': ','.join(feat_names),
            'val_rmse_c': result['val_rmse'],
            'best_epoch': result['best_epoch'],
            'train_time_min': result['train_time_sec'] / 60,
        })

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'state_dict': result['model_state'],
                'feature_names': feat_names,
                'feature_indices': feat_idx,
                'val_rmse_c': result['val_rmse'],
                'y_mean': result['y_mean'],
                'y_std': result['y_std'],
            }, save_dir / f'ablation_{name}.pt')

        print(f"  ИТОГ: val_rmse={result['val_rmse']:.3f}°C, "
              f"best_epoch={result['best_epoch']}, "
              f"time={result['train_time_sec']/60:.1f} мин")

    return pd.DataFrame(rows).sort_values('val_rmse_c').reset_index(drop=True)

# ---------------------------------------------------------------------------
# Выбор победителя
# ---------------------------------------------------------------------------

def pick_ablation_winner(results_df: pd.DataFrame,
                         tolerance: float = 0.2,
                         prefer_fewer: bool = True) -> str:
    """
    Выбирает конфигурацию по правилу:
        - если RMSE в пределах tolerance °C от лучшей — берём с меньшим числом фичей
        - иначе — лучшую по RMSE

    Это защита от защиты «вы переоверфитили, специально подобрав мало фичей».
    """
    best_rmse = results_df['val_rmse_c'].min()
    candidates = results_df[results_df['val_rmse_c'] <= best_rmse + tolerance]
    if prefer_fewer:
        winner = candidates.sort_values('n_features').iloc[0]
    else:
        winner = candidates.sort_values('val_rmse_c').iloc[0]
    return winner['group']


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------

def plot_ablation_results(results_df: pd.DataFrame, save_path: Optional[Path] = None):
    """Bar chart: RMSE vs # features, с метками конфигураций."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: RMSE per configuration
    axes[0].barh(results_df['group'], results_df['val_rmse_c'],
                 color=['green' if i == 0 else 'steelblue'
                        for i in range(len(results_df))])
    axes[0].set_xlabel('Validation RMSE, °C')
    axes[0].set_title('RMSE по конфигурациям (отсортировано)')
    axes[0].invert_yaxis()
    for i, (rmse, n) in enumerate(zip(results_df['val_rmse_c'], results_df['n_features'])):
        axes[0].text(rmse + 0.01, i, f'{rmse:.3f} ({n}f)', va='center')

    # Panel 2: RMSE vs # features (scatter)
    axes[1].scatter(results_df['n_features'], results_df['val_rmse_c'],
                    s=100, alpha=0.7, edgecolor='k')
    for _, row in results_df.iterrows():
        axes[1].annotate(row['group'],
                         (row['n_features'], row['val_rmse_c']),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[1].set_xlabel('# features')
    axes[1].set_ylabel('Validation RMSE, °C')
    axes[1].set_title('RMSE vs # features')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
