"""
P3 Model A — ENSEMBLE из 3 моделей с разными seeds.

Цель: получить robust оценку val_rmse через множественные запуски.
Если модели дают близкие val_rmse → робастная оценка.
Если сильно разные → проблема со split, нужно искать другое решение.

Train/Val split:
- train targets: 2007..2022 (16 target лет, индексы 4..19)
- val:           2023, 2024, 2025 (3 года, индексы 20, 21, 22)

Запуски: 3 модели с seeds [42, 123, 456]

Вход:
  data/tensor_01deg_extended_23y.npz
  data/y_new_rk_landcover_extended_23y.npz

Выход:
  models/convlstm_ttop_p3_model_A_seed_{seed}.pt  (3 файла)
  models/convlstm_ttop_p3_model_A_BEST.pt         (копия лучшего)
  results/metrics/p3_model_A_ensemble.csv         (статистика)
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

TRAIN_TARGETS = list(range(4, 20))     # 2007..2022 (16 target years)
VAL_TARGETS = [20, 21, 22]              # 2023, 2024, 2025

SEEDS = [42, 123, 456]                  # 3 запуска

EPOCHS = 80
LR = 1e-4
BATCH_SIZE = 8
EARLY_STOP_PATIENCE = 20

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
    print(f"  Val:   {[years[i] for i in VAL_TARGETS]}")

    if not ABLATION_TABLE.exists():
        raise FileNotFoundError(f"Нет {ABLATION_TABLE}")
    winner_name, winner_feats = load_ablation_winner(ABLATION_TABLE)
    core_idx = features_by_names(winner_feats)
    print(f"\nFeatures: {winner_feats}")
    print(f"\n=== Запуск ensemble из {len(SEEDS)} моделей ===")
    print(f"Seeds: {SEEDS}")
    print(f"Ожидаемое время: ~{len(SEEDS) * 8} минут\n")

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
            val_targets=VAL_TARGETS,
            epochs=EPOCHS,
            lr=LR,
            batch_size=BATCH_SIZE,
            device=device,
            early_stop_patience=EARLY_STOP_PATIENCE,
            seed=seed,
        )

        val_rmse = result['val_rmse']
        best_epoch = result['best_epoch']
        time_min = result['train_time_sec'] / 60

        print(f"\n  seed={seed}: val_rmse={val_rmse:.3f}°C, best_epoch={best_epoch}, time={time_min:.1f} мин")

        # Сохраняем
        model_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{seed}.pt'
        torch.save({
            'state_dict': result['model_state'],
            'feature_names': winner_feats,
            'feature_indices': core_idx,
            'val_rmse': val_rmse,
            'y_mean': result['y_mean'],
            'y_std': result['y_std'],
            'best_epoch': best_epoch,
            'seed': seed,
            'train_years': [int(years[i]) for i in TRAIN_TARGETS],
            'val_years': [int(years[i]) for i in VAL_TARGETS],
            'description': f'P3 Model A, seed={seed}, val on 2023-2025',
        }, model_path)

        results.append({
            'seed': seed,
            'val_rmse': val_rmse,
            'best_epoch': best_epoch,
            'time_min': time_min,
            'model_path': str(model_path.relative_to(BASE_DIR)),
        })

    # ===== Статистика =====
    df = pd.DataFrame(results)
    print(f"\n\n{'='*60}")
    print(f"=== Итоги ensemble ===")
    print(f"{'='*60}")
    print(df[['seed', 'val_rmse', 'best_epoch', 'time_min']].to_string(index=False))

    val_rmses = df['val_rmse'].values
    mean = val_rmses.mean()
    std = val_rmses.std()
    min_v = val_rmses.min()
    max_v = val_rmses.max()

    print(f"\n  Mean val_rmse:  {mean:.3f}°C")
    print(f"  Std val_rmse:   {std:.3f}°C")
    print(f"  Min val_rmse:   {min_v:.3f}°C  (seed={df.loc[df['val_rmse'].idxmin(), 'seed']})")
    print(f"  Max val_rmse:   {max_v:.3f}°C")
    print(f"  Range:          {max_v - min_v:.3f}°C")

    # Интерпретация
    print(f"\n  Интерпретация:")
    if std < 0.05:
        print(f"  ✓ ОТЛИЧНО: std={std:.3f}°C < 0.05 — модели очень стабильны")
    elif std < 0.1:
        print(f"  ✓ ХОРОШО: std={std:.3f}°C < 0.1 — модели стабильны")
    elif std < 0.2:
        print(f"  ⚠️ СРЕДНЕ: std={std:.3f}°C — есть зависимость от seed")
    else:
        print(f"  ❌ ПЛОХО: std={std:.3f}°C > 0.2 — модели сильно расходятся, split нестабилен")

    print(f"\n  Для защиты официально цитируем:")
    print(f"    val_rmse = {mean:.2f} ± {std:.2f}°C (ensemble из {len(SEEDS)} моделей)")

    # Сохраняем CSV
    csv_path = METRICS_DIR / 'p3_model_A_ensemble.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  CSV: {csv_path.relative_to(BASE_DIR)}")

    # Сохраняем "BEST" — модель с min val_rmse
    best_idx = df['val_rmse'].idxmin()
    best_seed = df.loc[best_idx, 'seed']
    best_path = MODELS_DIR / f'convlstm_ttop_p3_model_A_seed_{best_seed}.pt'
    best_copy = MODELS_DIR / 'convlstm_ttop_p3_model_A_BEST.pt'
    shutil.copy(best_path, best_copy)
    print(f"\n  BEST model (seed={best_seed}, val_rmse={df.loc[best_idx, 'val_rmse']:.3f}°C):")
    print(f"    {best_copy.relative_to(BASE_DIR)}")


if __name__ == '__main__':
    main()
