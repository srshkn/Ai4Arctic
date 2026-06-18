# Модели

## Финальная модель

**Файл:** `convlstm_ttop_rk_v2.pt` (~0.5 МБ)

ConvLSTM модель, обученная на target по формуле TTOP с rk(landcover).

## Метрики

| Метрика | Значение |
|---|---|
| Best epoch | 20 |
| Val loss (MSE normalized) | 0.0310 |
| Val R² | +0.971 |
| Val MAE (normalized) | 0.146 |
| Val MAE (denormalized) | ~0.78 °C |
| Параметров | 133 921 |

## Архитектура

```
Input  (B, T=4, C=20, H=64, W=64)
   │
   ├─→ ConvLSTMCell (in=20, hidden=32, kernel=3)  ── итерация по 4 годам ──┐
   │                                                                       │
   │                                              h_t: (B, 32, 64, 64) ←──┘
   ▼
BatchNorm2d (32)
   ▼
Dropout2d (p=0.2)
   ▼
Conv2d (32 → 1, kernel=1)
   ▼
Output (B, H=64, W=64)
```

## Гиперпараметры обучения

| Параметр | Значение |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Gradient clipping | max_norm=1.0 |
| Batch size | 8 |
| Epochs (max) | 80 |
| Early stopping patience | 20 |
| Dropout | 0.2 |
| Loss | Masked MSE |
| Random seed | 42 |

## Содержимое чекпоинта

Файл сохранён через `torch.save` и содержит словарь:

```python
{
    'model_state': OrderedDict,    # веса модели
    'x_mean': ndarray (20,),       # среднее по каналам для нормализации
    'x_std':  ndarray (20,),       # std по каналам
    'y_mean': -5.9989,             # среднее target в °C
    'y_std':  5.2419,              # std target
    'best_epoch': 20,
    'val_metrics': {'MAE': ..., 'RMSE': ..., 'R2': ...},
    'history': {...},              # train_loss, val_loss, val_mae, val_r2 по эпохам
    'rk_map': ndarray (231, 1501),  # карта r_k, использованная для target
}
```

## Загрузка модели

```python
from src.model import load_checkpoint

ckpt = load_checkpoint('models/convlstm_ttop_rk_v2.pt', device='cuda')

model = ckpt['model']            # модель в режиме eval
x_mean, x_std = ckpt['x_mean'], ckpt['x_std']
y_mean, y_std = ckpt['y_mean'], ckpt['y_std']
```

## Результаты на эталонах

| Сравнение | RMSE (°C) | Pearson r |
|---|---|---|
| vs Obu 2019 | **2.07** | +0.943 |
| vs ESA CCI 2023 | 3.90 | +0.852 |
| vs 33 GTN-P буров | 2.75 | +0.281 |

**Подробности** — в [`../notebooks/04_validation.ipynb`](../notebooks/04_validation.ipynb)
и [`../docs/model_architecture.md`](../docs/model_architecture.md).

## История обучения

Финальная модель — результат **5 итераций** работы:

1. ConvLSTM с rk=0.7 (без lh): RMSE vs Obu = 4.26 °C
2. + Latent heat correction: RMSE 2.53 °C
3. Переход к rk(landcover): требовал понимания, что FDD/TDD в тензоре без нормализации
4. Первое переобучение, lr=1e-3, best epoch=2: разрушение yearly корреляции
5. **Финальное переобучение, lr=1e-4, best epoch=20: RMSE 2.07 °C ✓**

Подробная хронология — в отчёте дипломной работы.
