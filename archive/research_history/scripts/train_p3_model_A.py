"""
P3 (model A): обучение ConvLSTM с валидацией на 3 годах 2023-2025.

Цель: дать НАДЁЖНУЮ оценку качества модели через independent validation.
Это модель для научной защиты — её val RMSE будет цитироваться в работе.

Train/Val split:
- input window: 4 года
- train targets: 2007..2022 (16 target лет, индексы 4..21)
- val:           2023, 2024, 2025 (3 года, индексы 20, 21, 22)

Сравнительно с P2:
- P2: train 2007..2021 (15 лет), val 2022-2023 (2 года), val RMSE 0.84°C
- P3-A: train 2007..2022 (16 лет), val 2023-2025 (3 года), val RMSE ?

Вход:
  data/tensor_01deg_extended_23y.npz
  data/y_new_rk_landcover_extended_23y.npz

Выход:
  models/convlstm_ttop_p3_model_A_val_2023_2025.pt
"""

import numpy as np
import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ablation import train_one_config, features_by_names, load_ablation_winner
from src.model import ConvLSTMNet
from src.reproducibility import DEFAULT_SEED


# --- Конфигурация ---
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
ABLATION_TABLE = METRICS_DIR / 'ablation_table.csv'

# 23-year тензор: 2003=0, 2004=1, ..., 2024=21, 2025=22
# Train: 2007-2022 (индексы 4..19, 16 target лет)
# Val:   2023, 2024, 2025 (индексы 20, 21, 22)
TRAIN_TARGETS = list(range(4, 20))    # 2007..2022 (16 target years)
VAL_TARGETS = [20, 21, 22]             # 2023, 2024, 2025

EPOCHS = 80
LR = 1e-4
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 20
SEED = DEFAULT_SEED

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT = MODELS_DIR / 'convlstm_ttop_p3_model_A_val_2023_2025.pt'


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # Данные
    print("Загружаем 23-year X-тензор...")
    tensor = np.load(DATA_DIR / 'tensor_01deg_extended_23y.npz')
    X = tensor['X']
    years = list(tensor['years'])
    print(f"  X: {X.shape}")

    print("Загружаем 23-year TTOP target...")
    y_npz = np.load(DATA_DIR / 'y_new_rk_landcover_extended_23y.npz')
    y = y_npz['y_new']
    print(f"  y: {y.shape}")

    print(f"\nYears: {years[0]}..{years[-1]} ({len(years)} лет)")
    print(f"Train target years ({len(TRAIN_TARGETS)}): {[years[i] for i in TRAIN_TARGETS]}")
    print(f"Val target years ({len(VAL_TARGETS)}):    {[years[i] for i in VAL_TARGETS]}")

    if not ABLATION_TABLE.exists():
        raise FileNotFoundError(
            f"Нет {ABLATION_TABLE}. Сначала запустите notebooks/07_feature_ablation.ipynb."
        )
    winner_name, winner_feats = load_ablation_winner(ABLATION_TABLE)
    core_idx = features_by_names(winner_feats)
    print(f"\nFeatures (P4 winner: {winner_name}): {winner_feats}")
    print(f"Indices: {core_idx}\n")

    print(f"Тренировка: lr={LR}, batch={BATCH_SIZE}, epochs={EPOCHS}, patience={EARLY_STOP_PATIENCE}\n")
    result = train_one_config(
        model_class=ConvLSTMNet,
        X_raw=X,
        y_raw=y,
        feature_indices=core_idx,
        train_targets=TRAIN_TARGETS,
        val_targets=VAL_TARGETS,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=device,
        early_stop_patience=EARLY_STOP_PATIENCE,
        seed=SEED,
    )

    print(f"\n=== P3 Model A (val on 2023-2025) ===")
    print(f"  val RMSE на {[years[i] for i in VAL_TARGETS]}: {result['val_rmse']:.3f}°C")
    print(f"  best epoch: {result['best_epoch']}")
    print(f"  time: {result['train_time_sec']/60:.1f} мин")
    print(f"\n  Сравнение:")
    print(f"    P4 winner (14 лет, val=2023):     val RMSE 0.762°C")
    print(f"    P2 (21 год, val=2022-2023):       val RMSE 0.837°C")
    print(f"    P3-A (23 года, val=2023-2024-2025): val RMSE {result['val_rmse']:.3f}°C ⭐")

    torch.save({
        'state_dict': result['model_state'],
        'feature_names': winner_feats,
        'feature_group': winner_name,
        'seed': SEED,
        'feature_indices': core_idx,
        'val_rmse': result['val_rmse'],
        'y_mean': result['y_mean'],
        'y_std': result['y_std'],
        'best_epoch': result['best_epoch'],
        'description': 'P3 Model A: ConvLSTM trained on 23-year tensor (2003-2025), '
                       f'{winner_name}, TTOP target with landcover-varying rk, '
                       'train=2007..2022 (16 years), val=2023-2024-2025 (3 years). '
                       'This is the SCIENCE model — its val RMSE is the official metric.',
        'train_years': [years[i] for i in TRAIN_TARGETS],
        'val_years': [years[i] for i in VAL_TARGETS],
        'years_all': years,
        'epochs_budget': EPOCHS,
        'lr': LR,
        'batch_size': BATCH_SIZE,
    }, MODEL_OUTPUT)
    print(f"\nСохранено: {MODEL_OUTPUT}")


if __name__ == '__main__':
    main()
