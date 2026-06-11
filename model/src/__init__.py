"""
Permafrost Russia ConvLSTM
==========================

Пакет для прогноза MAGT и ALT на территории России с помощью ConvLSTM-модели,
основанной на формуле TTOP с rk(landcover) и коррекцией на латентную теплоту.

Модули:
    model    : Архитектура ConvLSTM (ConvLSTMCell, ConvLSTMNet)
    data     : Загрузка данных, нормализация, формирование обучающих пар
    inference: Tile-based inference, MC Dropout uncertainty
    metrics  : Метрики качества (bias, MAE, RMSE, R², корреляции)
    landcover: Классификация landcover, карты rk и delta_t
"""

__version__ = "2.0.0"
__author__ = "[ВСТАВИТЬ_АВТОРА]"
