"""
P2: переобучение ConvLSTM на расширенном тензоре 2003-2023 (21 год).

Использует тот же ConvLSTM v2 что и в P4 winner модели, с тем же
набором фичей climate_core_6 (6 температурных признаков). Расширение
от 14 лет (2010-2023) до 21 года (2003-2023) даёт +50% обучающих данных.

Train/Val split:
- input window: 4 года (2003-2006 идут в первый input)
- train targets: 2007-2021 (15 target лет)
- val:           2022, 2023 (2 года, для устойчивости оценки)

Вход:
  data/tensor_01deg_extended.npz
  data/y_new_rk_landcover_extended.npz

Выход:
  models/convlstm_ttop_extended_21years.pt

Запуск:
  python3 scripts/train_extended_21years.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ablation import train_one_config, features_by_names
from src.model import ConvLSTMNet


# --- Конфигурация ---
CORE_6 = ['MAAT', 'LST_winter', 'TDD', 'LST_annual', 'FDD', 'era5_temp']

# Years индексы: 2003=0, 2004=1, ..., 2023=20
# Input window для ConvLSTM = 4 года → первые 4 года уходят в input
# Train на target лет 4..18 (2007..2021), val на 19, 20 (2022, 2023)
TRAIN_TARGETS = list(range(4, 19))    # 2007..2021 (15 target years)
VAL_TARGETS = [19, 20]                 # 2022, 2023

EPOCHS = 80
LR = 1e-4
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 20    # увеличено с 15 для большей стабильности

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT = MODELS_DIR / 'convlstm_ttop_extended_21years.pt'


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # Данные
    print("Загружаем расширенный X-тензор (21 год)...")
    tensor_ext = np.load(DATA_DIR / 'tensor_01deg_extended.npz')
    X_ext = tensor_ext['X']
    years_ext = list(tensor_ext['years'])
    print(f"  X: {X_ext.shape}")

    print("Загружаем расширенный TTOP target (21 год)...")
    y_ext_npz = np.load(DATA_DIR / 'y_new_rk_landcover_extended.npz')
    y_ext = y_ext_npz['y_new']
    print(f"  y: {y_ext.shape}")

    print(f"\nYears: {years_ext[0]}..{years_ext[-1]} ({len(years_ext)} лет)")
    print(f"Train target years: {[years_ext[i] for i in TRAIN_TARGETS]}")
    print(f"Val target years:   {[years_ext[i] for i in VAL_TARGETS]}")

    core_idx = features_by_names(CORE_6)
    print(f"\nFeatures (climate_core_6): {CORE_6}")
    print(f"Indices: {core_idx}\n")

    print(f"Тренировка: lr={LR}, batch={BATCH_SIZE}, epochs={EPOCHS}, patience={EARLY_STOP_PATIENCE}\n")
    result = train_one_config(
        model_class=ConvLSTMNet,
        X_raw=X_ext,
        y_raw=y_ext,
        feature_indices=core_idx,
        train_targets=TRAIN_TARGETS,
        val_targets=VAL_TARGETS,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        device=device,
        early_stop_patience=EARLY_STOP_PATIENCE,
    )

    print(f"\n=== Финальная модель P2 (21 год) ===")
    print(f"  val RMSE на {[years_ext[i] for i in VAL_TARGETS]}: {result['val_rmse']:.3f}°C")
    print(f"  best epoch: {result['best_epoch']}")
    print(f"  time: {result['train_time_sec']/60:.1f} мин")
    print(f"\n  Baseline P4 на 14 годе: val RMSE 0.762°C на 2023")

    torch.save({
        'state_dict': result['model_state'],
        'feature_names': CORE_6,
        'feature_indices': core_idx,
        'val_rmse': result['val_rmse'],
        'y_mean': result['y_mean'],
        'y_std': result['y_std'],
        'best_epoch': result['best_epoch'],
        'description': 'ConvLSTM trained on EXTENDED 2003-2023 tensor (P2), '
                       'climate_core_6, TTOP target with landcover-varying rk, '
                       'train=2007..2021, val=2022-2023',
        'train_years': [years_ext[i] for i in TRAIN_TARGETS],
        'val_years': [years_ext[i] for i in VAL_TARGETS],
        'years_all': years_ext,
        'epochs_budget': EPOCHS,
        'lr': LR,
        'batch_size': BATCH_SIZE,
    }, MODEL_OUTPUT)
    print(f"\nСохранено: {MODEL_OUTPUT}")


if __name__ == '__main__':
    main()
