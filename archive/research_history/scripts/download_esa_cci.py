"""
Скачивание ESA CCI Permafrost — Climate Research Data Package (CRDP).

Продукт: Obu et al. 2021, основан на TTOP-моделировании 2003-2019,
1 км глобальное разрешение, NetCDF формат.

Хостинг: CEDA Archive
    https://catalogue.ceda.ac.uk/uuid/4d3ce4a32cdc4b9ba79c47e8c33bd9cd

Содержит карты:
    - MAGT (Mean Annual Ground Temperature) — наша основная цель
    - PFR (Permafrost Fraction / Probability)
    - ALT (Active Layer Thickness)

Размер: ~200-400 МБ на год по всему миру; ~50-80 МБ на год для России bbox.

Запуск:
    python scripts/download_esa_cci.py --years 2003 2019 --output_dir data/esa_cci/

Авторизация:
    Для CEDA нужен бесплатный аккаунт. Зарегистрируйся:
        https://services.ceda.ac.uk/cedasite/register/start/
    Затем настрой ~/.netrc или используй --user/--password флаги.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: установи requests: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# URL шаблоны CEDA
# ---------------------------------------------------------------------------
# Базовый URL продукта ESA CCI Permafrost CRDP v04
# (может потребоваться обновление если ESA выпустит v05)

CEDA_BASE = (
    "https://data.ceda.ac.uk/neodc/esacci/permafrost/"
    "data/active_layer_thickness/L4/area4/pp/v04.0"
)
CEDA_MAGT = (
    "https://data.ceda.ac.uk/neodc/esacci/permafrost/"
    "data/ground_temperature/L4/area4/pp/v04.0"
)


def build_url(year: int, product: str = 'magt') -> str:
    """
    Формирует URL для конкретного года и продукта.

    product: 'magt' | 'alt' | 'pfr'
    """
    if product == 'magt':
        return f"{CEDA_MAGT}/{year}/ESACCI-PERMAFROST-L4-GTD-MODISLST-AREA4_PP-{year}-fv04.0.nc"
    elif product == 'alt':
        return f"{CEDA_BASE}/{year}/ESACCI-PERMAFROST-L4-ALT-MODISLST-AREA4_PP-{year}-fv04.0.nc"
    else:
        raise ValueError(f"Неизвестный продукт: {product}. Доступны: magt, alt")


# ---------------------------------------------------------------------------
# Скачивание
# ---------------------------------------------------------------------------

def download_file(url: str, output_path: Path, auth: tuple = None,
                  chunk_size: int = 8192) -> bool:
    """Скачивает файл с progress, поддерживает basic auth для CEDA."""
    if output_path.exists():
        print(f"  Уже скачано: {output_path.name}")
        return True

    try:
        with requests.get(url, auth=auth, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            downloaded = 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = output_path.with_suffix(output_path.suffix + '.tmp')

            with open(tmp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        sys.stdout.write(f"\r  {output_path.name}: {pct:.1f}% ({downloaded/1e6:.1f} MB)")
                        sys.stdout.flush()
            tmp_path.rename(output_path)
            print(f"\n  + {output_path.name} ({output_path.stat().st_size/1e6:.1f} MB)")
            return True
    except Exception as e:
        print(f"\n  ! Ошибка для {url}: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, nargs=2, default=[2003, 2019],
                        help='Диапазон годов: --years 2003 2019')
    parser.add_argument('--products', nargs='+', default=['magt'],
                        choices=['magt', 'alt'],
                        help='Какие продукты качать')
    parser.add_argument('--output_dir', type=str, default='data/esa_cci/')
    parser.add_argument('--user', type=str, default=None,
                        help='CEDA username (если нет ~/.netrc)')
    parser.add_argument('--password', type=str, default=None,
                        help='CEDA password (если нет ~/.netrc)')
    args = parser.parse_args()

    auth = None
    if args.user and args.password:
        auth = (args.user, args.password)
    else:
        netrc_path = Path.home() / '.netrc'
        if netrc_path.exists():
            print(f"Используем ~/.netrc для авторизации CEDA")
        else:
            print("ВНИМАНИЕ: нет ~/.netrc и не передан --user/--password.")
            print("Скорее всего скачивание провалится. Регистрация:")
            print("  https://services.ceda.ac.uk/cedasite/register/start/")

    output_dir = Path(args.output_dir)
    start_year, end_year = args.years
    print(f"Скачиваем годы {start_year}-{end_year}, продукты: {args.products}")
    print(f"Куда: {output_dir.resolve()}")

    successes, failures = 0, 0
    for year in range(start_year, end_year + 1):
        for product in args.products:
            url = build_url(year, product)
            filename = url.split('/')[-1]
            output_path = output_dir / product / filename
            print(f"\nГод {year}, продукт {product}:")
            print(f"  URL: {url}")
            if download_file(url, output_path, auth):
                successes += 1
            else:
                failures += 1

    print(f"\n=== Итого: {successes} успех, {failures} провалов ===")
    if failures > 0:
        print("Проверь:")
        print("  1. Авторизацию CEDA (логин/пароль)")
        print("  2. URL — возможно, ESA обновили v04 до v05, проверь")
        print(f"     {CEDA_MAGT}")


if __name__ == '__main__':
    main()
