"""
Финальная версия скрипта загрузки GTN-P boreholes.

Главное отличие от v1:
  Используется combined=true вместо combined=false.
  Это значит, что API сам возвращает агрегированный ground_temperature
  по всем датасетам скважины, отфильтрованный от air/surface temperature.
  Поэтому больше не нужно знать dataset_id заранее.

Ожидаемый результат: ~150-300 hard labels (вместо 55 в v1).

Запуск:
    python3 gtnp_via_api_v2.py
"""

import io
import time
import warnings
import zipfile
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

COUNTRIES = ["Russia"]   # API понимает только "Russia"

# Глубина zone of zero annual amplitude.
# Расширил vs v1 (было 8-25): мерзлота местами может быть thin,
# и температуры на 5-8 м тоже репрезентативны.
DEPTH_MIN = 5.0
DEPTH_MAX = 30.0

# Пауза между запросами, чтобы не нагружать сервер
SLEEP_SEC = 0.3

# Игнорируем pandas warnings
warnings.simplefilter("ignore")


# ============================================================
# 1. Список сайтов для страны
# ============================================================
def list_sites_for_country(country: str) -> list:
    url = f"{API_BASE}/list-sites"
    params = {"country": country, "borehole_data": "true", "metadata": "true"}
    print(f"  Запрос: {url}  country={country}")
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        print(f"   HTTP {r.status_code}: {r.text[:200]}")
        return []
    data = r.json()
    if isinstance(data, dict):
        sites = data.get("results") or data.get("sites") or data.get("data") or []
    else:
        sites = data
    print(f"   Найдено sites: {len(sites)}")
    return sites


# ============================================================
# 2. Извлечение boreholes из site
# ============================================================
def extract_boreholes_from_site(site: dict) -> list:
    boreholes = []
    lat = site.get("latitude") or site.get("lat") or site.get("LAT")
    lon = site.get("longitude") or site.get("lon") or site.get("LON")
    country = site.get("country", "Russia")
    site_name = site.get("site_name") or site.get("name", "")

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
# 3. Получить ground_temperature через combined=true
# ============================================================
def get_ground_temperature_data(borehole_id):
    """
    Скачивает zipped CSV с consolidated ground temperature.
    combined=true сам отфильтровывает air/surface, оставляет только грунт.
    """
    url = f"{API_BASE}/data/"
    params = {"pt_data": borehole_id, "combined": "true"}
    try:
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        if len(r.content) < 200:
            return None, "Пустой ответ"

        zf = zipfile.ZipFile(io.BytesIO(r.content))
        names = zf.namelist()

        # Ищем именно ground_temperature файл
        gt_files = [n for n in names if "ground_temperature" in n.lower()
                                       and n.endswith(".csv")]
        if not gt_files:
            return None, f"Нет ground_temperature в архиве (есть: {names})"

        df = pd.read_csv(zf.open(gt_files[0]))
        return df, "OK"
    except Exception as e:
        return None, f"Exception: {type(e).__name__}: {str(e)[:100]}"


# ============================================================
# 4. Расчёт MAGT
# ============================================================
def compute_magt(df: pd.DataFrame, dmin=DEPTH_MIN, dmax=DEPTH_MAX) -> dict | None:
    """
    Усредняет температуру на глубинах в зоне нулевой годовой амплитуды.
    """
    if df is None or df.empty:
        return None

    # Колонки уже известны из v1: date, depth, temperature, flag
    if not all(c in df.columns for c in ("depth", "temperature")):
        return None

    df = df.copy()
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df = df.dropna(subset=["depth", "temperature"])
    if df.empty:
        return None

    # Глубины в ZAA
    subset = df[(df["depth"] >= dmin) & (df["depth"] <= dmax)]
    if subset.empty:
        # Запасной вариант: берём максимальную доступную глубину
        max_depth = df["depth"].max()
        if max_depth < 1.0:  # совсем поверхностные не годятся
            return None
        subset = df[df["depth"] == max_depth]
        if subset.empty:
            return None

    magt = subset["temperature"].mean()
    depth_used = subset["depth"].median()

    year_min, year_max = None, None
    if "date" in subset.columns:
        try:
            dates = pd.to_datetime(subset["date"], errors="coerce").dropna()
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
    # ----- Сбор всех боргол -----
    all_boreholes = []
    for country in COUNTRIES:
        sites = list_sites_for_country(country)
        for site in sites:
            bhs = extract_boreholes_from_site(site)
            all_boreholes.extend(bhs)
        time.sleep(SLEEP_SEC)

    # Дедупликация
    seen = set()
    unique = []
    for bh in all_boreholes:
        if bh["borehole_id"] not in seen:
            seen.add(bh["borehole_id"])
            unique.append(bh)
    print(f"\nУникальных боргол: {len(unique)}")

    # ----- Скачивание ground temperature -----
    rows = []
    status_counts = {}
    for bh in tqdm(unique, desc="Скачивание ground_temperature"):
        df, status = get_ground_temperature_data(bh["borehole_id"])
        status_counts[status if "OK" in status or "404" in status or "Нет" in status else "other"] = \
            status_counts.get(status if "OK" in status or "404" in status or "Нет" in status else "other", 0) + 1

        magt_info = compute_magt(df) if df is not None else None
        row = {**bh, "status": status}
        if magt_info:
            row.update(magt_info)
        else:
            row["magt_c"] = None
        rows.append(row)
        time.sleep(SLEEP_SEC)

    # ----- Сохранение -----
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "gtnp_boreholes_raw_v2.csv", index=False)
    print(f"\nСырых записей: {len(out)}")

    clean = out.dropna(subset=["magt_c"]).copy()

    # Фильтр аномалий: убираем явно положительные MAGT (это не мерзлота)
    n_before = len(clean)
    clean = clean[clean["magt_c"] <= 1.0]  # +1 С -- запас для шума
    print(f"После фильтра MAGT <= +1 С: {len(clean)}/{n_before}")

    clean.to_csv(OUT_DIR / "gtnp_boreholes_v2.csv", index=False)
    print(f"Финальных скважин с MAGT: {len(clean)}")
    print(f"Сохранено: {OUT_DIR / 'gtnp_boreholes_v2.csv'}")

    if len(clean):
        print("\nРаспределение MAGT:")
        print(clean["magt_c"].describe())
        print("\nРаспределение по регионам (site_name):")
        print(clean["site_name"].value_counts().head(15))

    # Краткая статистика статусов
    print("\nСтатусы скачивания:")
    statuses = out["status"].value_counts()
    for s, n in statuses.head(10).items():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()