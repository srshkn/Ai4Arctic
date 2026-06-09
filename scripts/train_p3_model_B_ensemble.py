"""
P3 Model B — ENSEMBLE на ВСЕХ данных для финального прогноза 2026-2035.

Цель: использовать максимум данных (включая 2023, 2024, 2025) для обучения.
Эта модель НЕ имеет validation — она просто учится на максимум.
Используется ТОЛЬКО для прогноза 2026-2035 через synthetic features.

Train/Val split:
- train targets: 2007..2025 (19 target лет, индексы 4..22) — ВСЕ доступные
- val:           НЕТ (формально val = последний год, но НЕ используется для early stopping)

Запуски: 3 модели с seeds [42, 123, 456]

Hyperparameters:
- Фиксированное число эпох = 12 (из best_epoch ensemble Model A: median ~ 11)
- early_stop_patience = 999 (отключено, чтобы обучить ровно 12 эпох)

Вход:
  data/tensor_01deg_extended_23y.npz
  data/y_new_rk_landcover_extended_23y.npz

Выход:
  models/convlstm_ttop_p3_model_B_seed_{seed}.pt  (3 файла)
  models/convlstm_ttop_p3_model_B_BEST.pt         (копия первой как BEST)
  results/metrics/p3_model_B_ensemble.csv
"""

import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path
import shutil

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ablation import train_one_config, features_by_names, load_ablation_winner
from src.model import ConvLSTMNet


# --- Конфигурация ---
METRICS_DIR = BASE_DIR / 'results' / 'metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
ABLATION_TABLE = METRICS_DIR / 'ablation_table.csv'

# Все доступные target годы 2007-2025 (индексы 4..22 в 23-летнем тензоре)
# Используем 2007..2024 как train, 2025 как FAKE val для совместимости с train_one_config
# (но НЕ верим в эту val метрику, т.к. 2025 не репрезентативен — холодный outlier)
TRAIN_TARGETS = list(range(4, 22))     # 2007..2024 (18 target years)
VAL_TARGETS = [22]                      # 2025 — формально, но не для научной метрики

SEEDS = [42, 123, 456]

# Фиксированный budget эпох (из ensemble Model A: best_epoch ~ 11 для 2 из 3 моделей)
EPOCHS = 15  # с запасом
LR = 1e-4
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 999  # отключаем early stop — всегда учим 15 эпох

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}\n")

    # Данные
    print("Загружаем 23-year тензор...")
    tensor = np.load(DATA_DIR / 'tensor_01deg_extended_23y.npz')
    X = tensor['X']
    years = list(tensor['years'])

    y_npz = np.load(DATA_DIR / 'y_new_rk_landcover_extended_23y.npz')
    y = y_npz['y_new']

    print(f"  X: {X.shape}, y: {y.shape}")
    print(f"  Years: {years[0]}..{years[-1]}")
    print(f"  Train: {[years[i] for i in TRAIN_TARGETS]}")
    print(f"  Pseudo-val (2025, не используется как метрика): {[years[i] for i in VAL_TARGETS]}")

    if not ABLATION_TABLE.exists():
        raise FileNotFoundError(f"Нет {ABLATION_TABLE}")
    winner_name, winner_feats = load_ablation_winner(ABLATION_TABLE)
    core_idx = features_by_names(winner_feats)
    print(f"\nFeatures: {winner_feats}")
    print(f"\n=== Запуск ensemble из {len(SEEDS)} моделей ===")
    print(f"Seeds: {SEEDS}")
    print(f"Фиксированное число эпох: {EPOCHS}")
    print(f"Ожидаемое время: ~{len(SEEDS) * 4} минут\n")

    results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(SEEDS)}] Обучение с seed={seed}")
        print(f"{'='*60}")

        result = train_one_config(
            model_class=ConvLSTMNet,
            X_raw=X,
            y_raw=y,
            feature_indices=core_idx,
            train_targets=TRAIN_TARGETS,
            val_targets=VAL_TARGETS,  # формально, для совместимости
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            device=device,
            early_stop_patience=EARLY_STOP_PATIENCE,  # 999, фактически отключено
            seed=seed,
        )

        val_rmse = result['val_rmse']  # это не научная метрика, но сохраним
        best_epoch = result['best_epoch']
        time_min = result['train_time_sec'] / 60

        print(f"\n  seed={seed}: time={time_min:.1f} мин, best_epoch={best_epoch}")
        print(f"  (pseudo-val_rmse на 2025 = {val_rmse:.3f}°C, не используется как scientific metric)")

        # Сохраняем
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_B_seed_{seed}.pt'
        torch.save({
            'state_dict': result['model_state'],
            'feature_names': winner_feats,
            'feature_indices': core_idx,
            'pseudo_val_rmse_2025': val_rmse,  # явное переименование чтобы не путать
            'y_mean': result['y_mean'],
            'y_std': result['y_std'],
            'best_epoch': best_epoch,
            'seed': seed,
            'train_years': [int(years[i]) for i in TRAIN_TARGETS],
            'description': (
                f'P3 Model B (для финального прогноза), seed={seed}, '
                'trained on ALL years 2007-2024, '
                'no held-out validation, used ONLY for projection 2026-2035'
            ),
        }, model_path)

        results.append({
            'seed': seed,
            'pseudo_val_rmse_2025': val_rmse,
            'best_epoch': best_epoch,
            'time_min': time_min,
            'model_path': str(model_path.relative_to(BASE_DIR)),
        })

    # ===== Статистика =====
    df = pd.DataFrame(results)
    print(f"\n\n{'='*60}")
    print(f"=== Итоги ensemble Model B ===")
    print(f"{'='*60}")
    print(df[['seed', 'pseudo_val_rmse_2025', 'best_epoch', 'time_min']].to_string(index=False))

    print(f"\n  ВАЖНО: pseudo_val_rmse_2025 — это не научная метрика!")
    print(f"  Это просто RMSE модели на одном году (2025) во время обучения.")
    print(f"  Научная метрика для защиты — val_rmse Model A (0.785°C)")

    # Сохраняем CSV
    csv_path = METRICS_DIR / 'p3_model_B_ensemble.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  CSV: {csv_path.relative_to(BASE_DIR)}")

    # BEST = seed=42 (произвольно — все модели на одних данных)
    best_path = MODELS_DIR / f'convlstm_ttop_p3_model_B_seed_42.pt'
    best_copy = MODELS_DIR / 'convlstm_ttop_p3_model_B_BEST.pt'
    shutil.copy(best_path, best_copy)
    print(f"\n  BEST model (seed=42 как default):")
    print(f"    {best_copy.relative_to(BASE_DIR)}")
    print(f"\n  Используется для прогноза 2026-2035 через synthetic features.")


if __name__ == '__main__':
    main()
