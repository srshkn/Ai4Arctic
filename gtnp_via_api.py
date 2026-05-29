"""
Загрузка GTN-P boreholes через публичный REST API (data.gtn-p.org).

Что делает:
  1. Для списка стран (Russia + соседи -- мерзлота не знает границ)
     запрашивает list-sites endpoint.
  2. Для каждого site достаёт borehole IDs.
  3. Для каждого borehole скачивает данные температуры.
  4. Усредняет MAGT на глубине нулевой годовой амплитуды (8-25 м).
  5. Сохраняет итоговый CSV: borehole_id, name, lat, lon, country, magt_c, year_min, year_max.

ВАЖНО: я не могу проверить точные имена полей в API без вызова --
поэтому скрипт устроен оборонительно: ловит KeyError и пишет, что не нашлось.
Если что-то не пойдёт -- сначала откройте https://data.gtn-p.org/api/swagger
и посмотрите реальные имена полей в endpoint /api/list-sites и /api/pt/{id}.

Зависимости:
    pip install requests pandas tqdm

Запуск:
    python gtnp_via_api.py
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


# ============================================================
# КОНФИГ
# ============================================================
API_BASE = "https://data.gtn-p.org/api"
OUT_DIR  = Path("./data/labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Страны с мерзлотой, потенциально полезные (страна нужна на английском)
COUNTRIES = [
    "Russia", "Russian Federation",
    # На случай если потом расширите регион -- эти страны имеют активные boreholes
    # "Canada", "United States of America", "Norway", "Sweden", "Finland", "Mongolia",
]

# Глубина измерения, на которой считаем MAGT (zone of zero annual amplitude)
DEPTH_MIN = 8.0
DEPTH_MAX = 25.0

# Пауза между запросами, чтобы не нагружать сервер
SLEEP_SEC = 0.3


# ============================================================
# 1. Список сайтов для страны
# ============================================================
def list_sites_for_country(country: str) -> list:
    """
    GET /api/list-sites?country=<country>&borehole_data=true&metadata=true
    Возвращает list of dict с метаданными сайтов.
    """
    url = f"{API_BASE}/list-sites"
    params = {"country": country, "borehole_data": "true", "metadata": "true"}
    print(f"  Запрос: {url}  country={country}")
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        print(f"   HTTP {r.status_code}: {r.text[:200]}")
        return []
    data = r.json()
    # API может вернуть list или {results: [...]}. Подстраиваемся.
    if isinstance(data, dict):
        sites = data.get("results") or data.get("sites") or data.get("data") or []
    else:
        sites = data
    print(f"   Найдено sites: {len(sites)}")
    return sites


# ============================================================
# 2. Метаданные одного site (или borehole)
# ============================================================
def get_pt_metadata(pt_id) -> dict:
    """GET /api/pt/{id}  -> json с метаданными."""
    url = f"{API_BASE}/pt/{pt_id}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"    Ошибка для pt_id={pt_id}: {e}")
    return {}


# ============================================================
# 3. Парсер одного site -- вытаскиваем borehole_id, lat, lon, depths
# ============================================================
def extract_boreholes_from_site(site: dict) -> list:
    """
    Из объекта site достаёт список boreholes с координатами и доступными глубинами.
    Структура зависит от реального ответа API; обороняемся через .get().
    """
    boreholes = []
    # Координаты часто на уровне site
    lat = site.get("latitude") or site.get("lat") or site.get("LAT")
    lon = site.get("longitude") or site.get("lon") or site.get("LON")
    country = site.get("country", "Russia")
    site_name = site.get("site_name") or site.get("name", "")

    # Список борхол может лежать в site["boreholes"] или site["pt_boreholes"]
    bh_list = (site.get("boreholes") or site.get("pt_boreholes")
               or site.get("borehole_data") or [])
    for bh in bh_list:
        bh_id = bh.get("id") or bh.get("borehole_id") or bh.get("pt_id")
        bh_name = bh.get("name") or bh.get("borehole_name", "")
        bh_lat = bh.get("latitude") or lat
        bh_lon = bh.get("longitude") or lon
        bh_depth = bh.get("depth_max") or bh.get("depth")
        if bh_id is None or bh_lat is None or bh_lon is None:
            continue
        boreholes.append({
            "borehole_id": bh_id,
            "name": bh_name or site_name,
            "lat": float(bh_lat),
            "lon": float(bh_lon),
            "country": country,
            "site_name": site_name,
            "max_depth_m": bh_depth,
        })
    return boreholes


# ============================================================
# 4. Получить временные ряды температуры для одного borehole
# ============================================================
def get_temperature_data(borehole_id):
    """
    Скачивает zipped CSV с данными температуры по глубинам.
    Возвращает DataFrame: depth_m, date, temperature_c.
    """
    url = f"{API_BASE}/data/"
    params = {"pt_data": borehole_id, "combined": "false"}
    try:
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 200:
            return None
        # Ответ может быть zip с CSV; распаковываем
        import io, zipfile
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_files:
            return None
        # Берём первый CSV
        df = pd.read_csv(zf.open(csv_files[0]))
        return df
    except Exception as e:
        print(f"   Ошибка данных pt_id={borehole_id}: {e}")
        return None


# ============================================================
# 5. Расчёт MAGT из временного ряда
# ============================================================
def compute_magt(df: pd.DataFrame, dmin=DEPTH_MIN, dmax=DEPTH_MAX) -> dict | None:
    """
    Усредняет температуру на глубинах в зоне нулевой годовой амплитуды.
    Возвращает {magt_c, year_min, year_max, n_measurements, depth_used}.
    """
    if df is None or df.empty:
        return None

    # Гадаем имена колонок (API может называть по-разному)
    depth_col = next((c for c in df.columns
                      if "depth" in c.lower()), None)
    temp_col = next((c for c in df.columns
                     if "temperature" in c.lower() or "temp" in c.lower()), None)
    date_col = next((c for c in df.columns
                     if "date" in c.lower() or "time" in c.lower()), None)
    if not (depth_col and temp_col):
        return None

    df = df.dropna(subset=[depth_col, temp_col])
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df = df.dropna(subset=[depth_col, temp_col])

    # Глубины в ZAA
    subset = df[(df[depth_col] >= dmin) & (df[depth_col] <= dmax)]
    if subset.empty:
        # Если нет ничего в 8-25 м, берём максимально глубокое
        subset = df[df[depth_col] == df[depth_col].max()]
        if subset.empty:
            return None

    magt = subset[temp_col].mean()
    depth_used = subset[depth_col].median()

    year_min, year_max = None, None
    if date_col:
        try:
            dates = pd.to_datetime(subset[date_col], errors="coerce").dropna()
            if len(dates):
                year_min = int(dates.dt.year.min())
                year_max = int(dates.dt.year.max())
        except Exception:
            pass

    return {
        "magt_c": float(magt),
        "depth_used_m": float(depth_used),
        "n_measurements": int(len(subset)),
        "year_min": year_min,
        "year_max": year_max,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    all_boreholes = []
    for country in COUNTRIES:
        sites = list_sites_for_country(country)
        for site in sites:
            bhs = extract_boreholes_from_site(site)
            all_boreholes.extend(bhs)
        time.sleep(SLEEP_SEC)

    # Дедупликация по borehole_id
    seen = set()
    unique = []
    for bh in all_boreholes:
        if bh["borehole_id"] not in seen:
            seen.add(bh["borehole_id"])
            unique.append(bh)
    print(f"\nУникальных боргол: {len(unique)}")

    # Скачиваем данные температуры
    rows = []
    for bh in tqdm(unique, desc="Скачивание PT-данных"):
        df = get_temperature_data(bh["borehole_id"])
        magt_info = compute_magt(df) if df is not None else None
        row = {**bh}
        if magt_info:
            row.update(magt_info)
        else:
            row["magt_c"] = None
        rows.append(row)
        time.sleep(SLEEP_SEC)

    # Сохраняем CSV
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gtnp_boreholes_raw.csv", index=False)
    print(f"Сырых записей: {len(out)}")

    # С валидным MAGT
    clean = out.dropna(subset=["magt_c"])
    clean.to_csv(OUT_DIR / "gtnp_boreholes.csv", index=False)
    print(f"С валидным MAGT: {len(clean)}")
    print(f"Сохранено: {OUT_DIR / 'gtnp_boreholes.csv'}")

    # Краткая статистика
    if len(clean):
        print("\nРаспределение MAGT:")
        print(clean["magt_c"].describe())


if __name__ == "__main__":
    main()