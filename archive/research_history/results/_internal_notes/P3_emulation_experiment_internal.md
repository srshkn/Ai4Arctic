# P3 (ESA-trained эмуляция) — внутренняя справка

**Статус:** ЭКСПЕРИМЕНТ, НЕ ВКЛЮЧАЕТСЯ В ЗАЩИТУ

Куратор в прошлой итерации указал, что эмуляция чужих продуктов (ESA-trained,
Obu-trained) не является приемлемым подходом. Основная работа должна 
использовать TTOP-формулу с собственными rk коэффициентами + ConvLSTM.

## Что было проверено (для внутреннего понимания)

Обучили ConvLSTM на ESA CCI T2m как target (Westermann 2024) на:
- climate_core_6 (6 фичей)
- climate_wet_8 (8 фичей)
- climate_soil_9 (9 фичей)
- all_20 (20 фичей)

## Результаты mini-ablation на ESA target (val RMSE на ESA)

| Configuration | val RMSE (ESA) | best epoch | comment |
|---|---|---|---|
| climate_core_6 | 1.409°C | 18 | оптимум для TTOP, не для ESA |
| climate_wet_8 | 1.302°C | 21 | +NDVI, NDWI помогают |
| climate_soil_9 | 1.127°C | 35 | +snow_days, soil_oc, MAP |
| all_20 | 0.895°C → 0.846°C* | 68 | * на 80 эпохах, лучший |

## Сравнение vs 33 бурения

| Model | Bias | RMSE | MAE |
|---|---|---|---|
| TTOP-trained без P1 | -3.73 | 4.57 | 4.11 |
| TTOP+P1 (наша финальная) | **0.00** | **2.51** | 1.84 |
| ESA-trained core_6 | +1.78 | 3.17 | 2.34 |
| ESA-trained all_20 | +1.87 | 3.18 | 2.26 |

## Главные выводы (НЕ для отчёта)

1. ESA-trained требует ВСЕ 20 фичей, в отличие от TTOP-trained (6 хватает).
   Это потому что ESA CryoGrid внутри учитывает landcover, snow, soil — 
   нейросеть должна это репродуцировать.

2. ESA-trained показывает warm bias +1.8°C на бурениях, видимо из-за того
   что ESA T2m относится к 2м, а бурения к 10м.

3. **Наша финальная модель TTOP+P1 лучше эмуляционных моделей** 
   по обоим метрикам (RMSE 2.51 vs 3.17-3.18; bias 0.00 vs +1.78-1.87).
   Это сильный аргумент защиты.

## Файлы эксперимента (для внутреннего использования)

- models/convlstm_esa_target_climate_core_6.pt
- models/convlstm_esa_climate_core_6_more_train.pt
- models/convlstm_esa_all20_full_budget.pt
- data/esa_cci_target_2000_2021.npz (target тензор)
- data/esa_cci/magt/*.nc (исходные NetCDF, 7.6 ГБ)
- results/metrics/p3_final_comparison.json
