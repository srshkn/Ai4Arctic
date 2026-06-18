"""
Архитектура ConvLSTM для прогноза MAGT.

Содержит:
- ConvLSTMCell: одна ячейка ConvLSTM с свёрткой 3x3
- ConvLSTMNet: полная модель (1 слой LSTM + BatchNorm + Dropout + Conv1x1)

Финальная модель имеет 133 921 параметр.

Reference: Shi et al. (2015). Convolutional LSTM Network: A Machine Learning
Approach for Precipitation Nowcasting. NIPS 2015.
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Одна ячейка ConvLSTM с свёрткой kernel x kernel.

    Все полносвязные операции стандартной LSTM заменены на 2D-свёртки,
    что позволяет учитывать пространственный контекст.

    Parameters
    ----------
    in_ch : int
        Количество каналов на входе (для нашей задачи: 20 признаков).
    hidden_ch : int
        Размер скрытого состояния (по умолчанию 32).
    kernel : int
        Размер свёрточного ядра (по умолчанию 3).
    """

    def __init__(self, in_ch, hidden_ch, kernel=3):
        super().__init__()
        pad = kernel // 2
        self.hidden_ch = hidden_ch
        # 4 гейта (input, forget, output, candidate) — одна свёртка с 4*hidden каналами
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=pad)

    def forward(self, x, hc):
        h, c = hc
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)   # input gate
        f = torch.sigmoid(f)   # forget gate
        o = torch.sigmoid(o)   # output gate
        g = torch.tanh(g)      # candidate cell state
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new

    def init_state(self, batch, H, W, device):
        """Инициализация скрытого состояния и состояния ячейки нулями."""
        h = torch.zeros(batch, self.hidden_ch, H, W, device=device)
        c = torch.zeros(batch, self.hidden_ch, H, W, device=device)
        return h, c


class ConvLSTMNet(nn.Module):
    """
    Полная модель ConvLSTM для прогноза MAGT.

    Архитектура:
        вход (B, T, C_in, H, W) → ConvLSTMCell × T шагов → BatchNorm → Dropout
                                → Conv2d(C_hidden → 1) → выход (B, H, W)

    Parameters
    ----------
    in_ch : int
        Количество входных признаков (20).
    hidden_ch : int
        Размер скрытого состояния (32 в финальной модели).
    dropout : float
        Вероятность Dropout (0.2 в финальной модели).
    """

    def __init__(self, in_ch=20, hidden_ch=32, dropout=0.2):
        super().__init__()
        self.lstm = ConvLSTMCell(in_ch, hidden_ch, kernel=3)
        self.bn = nn.BatchNorm2d(hidden_ch)
        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(hidden_ch, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, T, C, H, W) — батч из B последовательностей по T кадров.
        Возвращает: (B, H, W) — прогноз MAGT для следующего года.
        """
        B, T, C, H, W = x.shape
        h, c = self.lstm.init_state(B, H, W, x.device)
        for t in range(T):
            h, c = self.lstm(x[:, t], (h, c))
        h = self.bn(h)
        h = self.dropout(h)
        out = self.head(h)
        return out.squeeze(1)


def count_parameters(model):
    """Утилита: подсчёт количества обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Загружает чекпоинт финальной модели и возвращает model + все метаданные.

    Returns
    -------
    dict со следующими ключами:
        model        : ConvLSTMNet с загруженными весами в режиме eval
        x_mean, x_std: статистики нормализации для 20 признаков
        y_mean, y_std: статистики нормализации target
        best_epoch   : номер лучшей эпохи (= 20 в финальной модели)
        rk_map       : карта r_k (231, 1501)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = ConvLSTMNet(in_ch=20, hidden_ch=32, dropout=0.2).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    return {
        'model': model,
        'x_mean': ckpt.get('x_mean'),
        'x_std': ckpt.get('x_std'),
        'y_mean': float(ckpt.get('y_mean', -5.9989)),
        'y_std': float(ckpt.get('y_std', 5.2419)),
        'best_epoch': ckpt.get('best_epoch', 20),
        'rk_map': ckpt.get('rk_map'),
        'history': ckpt.get('history'),
        'val_metrics': ckpt.get('val_metrics'),
    }
