"""
Этап 5: P2 inference на всей истории 2007–2025 (для графиков тренда).

Запускает подряд:
  1. inference_p2_yearly_maps.py  → p2_yearly_maps.npz (2007–2024)
  2. inference_p2_2025.py       → p2_real_2025.npz (2025)

Нужен для plot_ensemble6_final.py и сравнения real vs extrapolation.

Запуск:
    python3 scripts/inference_p2_history.py
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS_DIR.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))


def main():
    from inference_p2_yearly_maps import main as run_yearly
    from inference_p2_2025 import main as run_2025

    print("=" * 60)
    print("ЭТАП 5: P2 inference 2007–2025")
    print("=" * 60)

    print("\n[5a] Годовые карты 2007–2024...\n")
    run_yearly()

    print("\n" + "=" * 60)
    print("[5b] Карта 2025...\n")
    run_2025()

    print("\n" + "=" * 60)
    print("Готово. Для визуализации:")
    print("  results/maps/p2_yearly_maps.npz")
    print("  results/maps/p2_real_2025.npz")
    print("  → python3 scripts/plot_ensemble6_final.py")


if __name__ == '__main__':
    main()
