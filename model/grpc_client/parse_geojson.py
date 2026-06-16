import json
from pathlib import Path
from typing import Any, Dict, List


def parse_geojson_directory(path: Path) -> List[Dict[str, Any]]:
    """
    Сканирует директорию на наличие .geojson файлов, читает их
    и возвращает список словарей с нужной структурой.
    """

    # Проверяем, существует ли директория
    if not path.is_dir():
        raise ValueError(f"Директория '{path.name}' не найдена.")

    parsed_data_list = []

    # Ищем все файлы с расширением .geojson (регистронезависимый поиск)
    geojson_files = [
        f for f in path.iterdir() if f.is_file() and f.suffix.lower() == ".geojson"
    ]

    if not geojson_files:
        print(f"В директории '{path.name}' не найдено файлов .geojson")
        return parsed_data_list

    for file_path in geojson_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # Загружаем JSON-содержимое
                raw_data = json.load(file)

                parsed_data_list.append(raw_data)
                print(f"Успешно обработан файл: {file_path.name}")

        except json.JSONDecodeError:
            print(f"❌ Ошибка: Файл '{file_path.name}' не является валидным JSON.")
        except Exception as e:
            print(f"❌ Неожиданная ошибка при чтении '{file_path.name}': {e}")

    return parsed_data_list
