"""
Фиксация случайности для воспроизводимых экспериментов.

Использование:
    from src.reproducibility import set_global_seed, DEFAULT_SEED

    set_global_seed(42, device='mps')
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch

DEFAULT_SEED = 42


def set_global_seed(
    seed: int = DEFAULT_SEED,
    device: Optional[str] = None,
    deterministic: bool = True,
) -> None:
    """
    Фиксирует seed для Python, NumPy и PyTorch.

    На MPS часть операций может оставаться недетерминированной — тогда
    PyTorch выдаст предупреждение (warn_only=True), но порядок батчей
    и инициализация весов всё равно воспроизводимы.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if deterministic:
        warn_only = device in ('mps', 'cpu', None)
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def train_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Generator для DataLoader(shuffle=True) с фиксированным порядком батчей."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen
