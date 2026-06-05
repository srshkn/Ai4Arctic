# Воспроизведение результатов

Пошаговая инструкция для повторения всех результатов работы.

## Сценарий A: только inference (быстро)

**Цель:** загрузить готовую модель, применить к тензору, получить карты MAGT/ALT.

**Время:** ~15 минут

**Что нужно:**
- Python 3.10+, PyTorch
- GPU желательно, но не обязательно
- ~5 ГБ свободного места

**Шаги:**

```bash
# 1. Клонирование репозитория
git clone -b v2-final https://github.com/USER/REPO.git permafrost-russia-convlstm
cd permafrost-russia-convlstm

# 2. Установка зависимостей
pip install -r requirements.txt

# 3. Скачивание данных (см. data/external_links.md)
#    Положите tensor_01deg_v2.npz в data/
#    Положите obu_2019_reprojected.npz в data/

# 4. Запуск ноутбука inference
jupyter notebook notebooks/03_inference.ipynb
```

После выполнения в `results/maps/` появятся:
- `FINAL_maps.npz` — карты MAGT 2014–2024
- `ALT_2023.npz` — карта ALT

---

## Сценарий B: полное воспроизведение (с обучением)

**Цель:** повторить работу с нуля, включая обучение модели.

**Время:** ~2 часа (включая 30 минут обучения)

**Шаги:**

```bash
# 1-3. То же, что в Сценарии A

# 4. Запуск всех 4 ноутбуков по порядку
jupyter notebook notebooks/01_data_preparation.ipynb   # ~5 мин
jupyter notebook notebooks/02_train_model.ipynb        # ~30 мин (требует GPU)
jupyter notebook notebooks/03_inference.ipynb          # ~10 мин
jupyter notebook notebooks/04_validation.ipynb         # ~40 мин (MC Dropout долгий)
```

После выполнения:
- В `models/` — обученная модель `convlstm_ttop_rk_v2.pt`
- В `results/maps/` — все .npz карты
- В `results/metrics/` — все CSV таблицы
- В `results/figures/` — все PNG графики

---

## Сценарий C: подготовка данных с нуля

**Цель:** воссоздать `tensor_01deg_v2.npz` из исходных источников.

**Время:** ~6–10 часов (большая часть — выгрузка из GEE)

**Требования:**
- Аккаунт Google Earth Engine
- ~10 ГБ свободного места

**Шаги:**

```bash
# 1. Регистрация в Google Earth Engine: https://earthengine.google.com/

# 2. Запуск выгрузки (требуется доступ к gee.ipynb отдельно):
#    Запустит 56 export tasks → файлы в Google Drive

# 3. Скачивание geojson файлов из Drive в data/raw/GEE_exports/

# 4. Объединение и растеризация:
python scripts/merge_geojson_bands.py
python scripts/rasterize_to_tensor.py
python scripts/compute_ttop_target.py

# Результат: data/tensor_01deg_v2.npz
```

---

## Проверка воспроизводимости

После запуска `notebooks/04_validation.ipynb` проверьте `results/metrics/metrics_main_table.csv`:

```
                              Сравнение      N   Bias    MAE   RMSE  Pearson
        ConvLSTM_v2 vs Real TTOP target  229745  +0.42  +0.78  +0.98   +0.989
 Modified TTOP_v2 vs Obu MAGT 2000-2016  213258  -1.01  +1.52  +2.07   +0.943
        Modified TTOP_v2 vs ESA CCI 2023  182798  -3.15  +3.31  +3.90   +0.852
        Modified TTOP_v2 vs 33 boreholes      33  -1.02  +2.26  +2.75   +0.281
```

Если RMSE vs Obu = **2.07** ± 0.02 °C — воспроизведение прошло успешно.

---

## Известные проблемы и решения

### Проблема: «CUDA out of memory»

**Причина:** GPU < 8 ГБ VRAM.
**Решение:** уменьшить `batch_size` с 8 до 4 в `notebooks/02_train_model.ipynb`.

### Проблема: ноутбук падает при загрузке тензора

**Причина:** мало RAM.
**Решение:** требуется минимум 16 ГБ RAM. Если у вас 8 ГБ — закройте другие
программы или используйте Colab Pro.

### Проблема: «Best epoch = 2» вместо 20

**Причина:** случайно установлен lr=1e-3 вместо 1e-4.
**Решение:** проверьте гиперпараметры в `02_train_model.ipynb`:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # НЕ 1e-3!
```

### Проблема: финальные метрики отличаются на > 5%

**Причина:** разная среда PyTorch / CUDA.
**Решение:**
1. Проверьте `torch.__version__` (используется 2.0+)
2. Проверьте `random_seed = 42` зафиксирован в начале каждого ноутбука
3. Убедитесь, что `torch.backends.cudnn.deterministic = True`

Допустимое расхождение ± 0.05 °C по RMSE.

---

## Что **нельзя** воспроизвести точно

1. **Колаб-сессии падали** во время оригинальной работы 5+ раз. Каждый перезапуск
   мог немного изменить состояние оптимизатора. Идеальная побитовая воспроизводимость
   не гарантирована.

2. **GEE выгрузки** могут немного отличаться, если используются последние данные
   (например, MODIS обновился). Допустимо расхождение в 0.1 °C на отдельных пикселях.

3. **MC Dropout** — стохастический по своей природе. Каждый запуск даёт чуть
   разный std (±5%).

---

## Контакт для поддержки

Если воспроизведение не получается:
- Создайте issue в репозитории: [ВСТАВИТЬ_ССЫЛКУ]/issues
- Email: [ВСТАВИТЬ_EMAIL]
