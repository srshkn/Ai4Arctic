#!/bin/bash
# Миграция репозитория Ai4Arctic в финальную структуру с папкой ml/.
#
# ИСПОЛЬЗОВАНИЕ:
#   1. Сухой прогон (ничего не меняет, только показывает план):
#        bash migrate_repo.sh --dry-run
#   2. Реальный запуск:
#        bash migrate_repo.sh
#
# ВАЖНО:
#   - Запускайте ИЗ корня репозитория Ai4Arctic/
#   - Должна быть отдельная ветка (не main)
#   - Использует git mv, чтобы сохранить историю файлов

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== СУХОЙ ПРОГОН — ничего не будет изменено ==="
    echo
fi

# === Утилиты ===
run() {
    if $DRY_RUN; then
        echo "DRY: $*"
    else
        echo ">>> $*"
        eval "$@"
    fi
}

git_mv() {
    local src="$1"
    local dst="$2"
    if [[ ! -e "$src" ]]; then
        echo "  ПРОПУСК (нет файла): $src"
        return
    fi
    if $DRY_RUN; then
        echo "DRY: git mv '$src' '$dst'"
    else
        # создаём родительскую папку, если нужно
        mkdir -p "$(dirname "$dst")"
        git mv "$src" "$dst"
    fi
}

# === Проверки перед стартом ===
if [[ ! -d ".git" ]]; then
    echo "ОШИБКА: запускайте из корня репозитория (нет .git/)"
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" == "main" || "$CURRENT_BRANCH" == "master" ]]; then
    echo "ОШИБКА: вы на ветке $CURRENT_BRANCH"
    echo "Создайте отдельную ветку:"
    echo "    git checkout -b feature/ml-pipeline"
    exit 1
fi

if [[ -n "$(git status --porcelain)" ]] && ! $DRY_RUN; then
    echo "ОШИБКА: есть незакоммиченные изменения"
    echo "Закоммитьте или спрячьте (git stash) перед запуском"
    exit 1
fi

echo "Ветка: $CURRENT_BRANCH"
echo "Режим: $([ $DRY_RUN = true ] && echo 'dry-run' || echo 'реальный')"
echo

# ============================================================
# ЭТАП 1: Создание структуры папок
# ============================================================
echo "=== ЭТАП 1: создаю структуру папок ==="

DIRS=(
    "ml/src/permafrost/data"
    "ml/src/permafrost/features"
    "ml/src/permafrost/models"
    "ml/src/permafrost/validation"
    "ml/src/permafrost/analysis"
    "ml/src/permafrost/visualization"
    "ml/experiments/01_boreholes_as_target"
    "ml/experiments/02_single_region"
    "ml/experiments/03_loro_five_regions"
    "ml/experiments/04_long_trend_failed"
    "ml/models"
    "ml/data"
    "ml/reports/figures/maps"
    "ml/reports/figures/validation"
    "ml/reports/figures/trend"
    "ml/reports/figures/forecast"
    "ml/reports/figures/classification"
    "ml/reports/tables"
    "ml/reports/docs"
    "ml/archive"
)

for d in "${DIRS[@]}"; do
    run "mkdir -p '$d'"
done

# Пустые __init__.py для пакетов
INITS=(
    "ml/src/__init__.py"
    "ml/src/permafrost/__init__.py"
    "ml/src/permafrost/data/__init__.py"
    "ml/src/permafrost/features/__init__.py"
    "ml/src/permafrost/models/__init__.py"
    "ml/src/permafrost/validation/__init__.py"
    "ml/src/permafrost/analysis/__init__.py"
    "ml/src/permafrost/visualization/__init__.py"
)
for f in "${INITS[@]}"; do
    run "touch '$f'"
done

# ============================================================
# ЭТАП 2: Перенос финальных скриптов в src/permafrost/
# ============================================================
echo
echo "=== ЭТАП 2: переношу финальные скрипты ==="

# data/
git_mv "sample_esacci_target.py"  "ml/src/permafrost/data/sample_targets.py"
git_mv "merge_bands.py"            "ml/src/permafrost/data/merge_bands.py"

# models/
git_mv "train_scatter.py"          "ml/src/permafrost/models/train_magt.py"
git_mv "train_alt.py"              "ml/src/permafrost/models/train_alt.py"
git_mv "train_multiyear.py"        "ml/src/permafrost/models/train_multiyear.py"

# validation/
git_mv "validate_on_boreholes.py"  "ml/src/permafrost/validation/boreholes.py"
git_mv "compare_map_vs_obu.py"     "ml/src/permafrost/validation/compare_obu.py"
git_mv "verify_trend_esacci.py"    "ml/src/permafrost/validation/verify_trend.py"
git_mv "validation_summary.py"     "ml/src/permafrost/validation/summary.py"

# analysis/
git_mv "classify_permafrost.py"    "ml/src/permafrost/analysis/classify.py"
git_mv "build_trend_multiyear.py"  "ml/src/permafrost/analysis/trend.py"
git_mv "forecast_cmip6.py"         "ml/src/permafrost/analysis/forecast.py"

# visualization/
git_mv "build_gis_maps.py"         "ml/src/permafrost/visualization/gis_maps.py"

# ============================================================
# ЭТАП 3: Перенос моделей с переименованием
# ============================================================
echo
echo "=== ЭТАП 3: переношу обученные модели ==="
git_mv "models_scatter_xgb.joblib"     "ml/models/magt_xgb.joblib"
git_mv "models_alt_xgb.joblib"         "ml/models/alt_xgb.joblib"
git_mv "models_multiyear_xgb.joblib"   "ml/models/magt_multiyear_xgb.joblib"

# ============================================================
# ЭТАП 4: Перенос экспериментов (научная история провалов)
# ============================================================
echo
echo "=== ЭТАП 4: переношу эксперименты ==="

# 01. Скважины как target (провал R²=0.01)
git_mv "gtnp_via_api_v2.py"        "ml/experiments/01_boreholes_as_target/gtnp_via_api.py"
git_mv "compare_obu_vs_gtnp.py"    "ml/experiments/01_boreholes_as_target/compare_obu_vs_gtnp.py"

# v1 удаляем как устаревший
if [[ -f "gtnp_via_api.py" ]]; then
    run "git rm gtnp_via_api.py"
fi

# 02. Один регион (Nadym)
git_mv "train_nadym.py"            "ml/experiments/02_single_region/train_nadym.py"

# 03. LORO 5 регионов + диагностика
git_mv "loro_5regions.py"          "ml/experiments/03_loro_five_regions/loro_5regions.py"
git_mv "step1_climate_leakage.py"  "ml/experiments/03_loro_five_regions/step1_climate_leakage.py"
git_mv "step2_two_regions.py"      "ml/experiments/03_loro_five_regions/step2_two_regions.py"
git_mv "step4_spatial_cv.py"       "ml/experiments/03_loro_five_regions/step4_spatial_cv.py"
git_mv "checkB_features_and_physics.py" "ml/experiments/03_loro_five_regions/checkB_features_and_physics.py"

# 04. Длинный тренд (провал — шум сохранился)
git_mv "build_change_map_clean.py" "ml/experiments/04_long_trend_failed/build_change_map_clean.py"
git_mv "build_trend_long.py"       "ml/experiments/04_long_trend_failed/build_trend_long.py"

# ============================================================
# ЭТАП 5: Архив устаревших версий карт
# ============================================================
echo
echo "=== ЭТАП 5: устаревшие версии в archive/ ==="
git_mv "build_map.py"              "ml/archive/build_map.py"
git_mv "build_quality_map.py"      "ml/archive/build_quality_map.py"
git_mv "build_quality_map_v2.py"   "ml/archive/build_quality_map_v2.py"

# ============================================================
# ЭТАП 6: Notebooks
# ============================================================
echo
echo "=== ЭТАП 6: переношу notebooks ==="
# notebooks/ уже существует в репо (для исследований)
git_mv "gee.ipynb"                 "notebooks/01_features_2021.ipynb"
# cmip6_dt.ipynb на диске может не быть (вы скачивали из /mnt/user-data/outputs)
# если есть локально — раскомментируйте:
# git_mv "cmip6_dt.ipynb"          "notebooks/03_cmip6_delta_t.ipynb"

# ============================================================
# ЭТАП 7: data/ — переносим папку как есть
# ============================================================
echo
echo "=== ЭТАП 7: переношу data/ под ml/ ==="
# data/ почти целиком в .gitignore, переносим что есть
if [[ -d "data" ]]; then
    if $DRY_RUN; then
        echo "DRY: git mv data ml/data_existing  (потом сольём)"
    else
        # если ml/data/ уже создан пустой — переносим содержимое
        if [[ -d "ml/data" ]] && [[ -z "$(ls -A ml/data 2>/dev/null)" ]]; then
            rmdir ml/data
        fi
        git mv data ml/data
    fi
fi

# ============================================================
# ЭТАП 8: reports/ — переносим под ml/ и раскладываем по подпапкам
# ============================================================
echo
echo "=== ЭТАП 8: переношу reports/ ==="
# Сначала переносим всю папку под ml/
if [[ -d "reports" ]]; then
    if $DRY_RUN; then
        echo "DRY: переношу reports/ под ml/reports/"
        echo "DRY: и раскладываю картинки по figures/{maps,validation,trend,forecast,classification}/"
    else
        # Сохраним временно, потом разложим
        if [[ -d "ml/reports" ]] && [[ -z "$(ls -A ml/reports/figures 2>/dev/null)" ]]; then
            # ml/reports пустой — просто переносим содержимое
            git mv reports/* ml/reports/ 2>/dev/null || true
            rmdir reports 2>/dev/null || true

            cd ml/reports

            # Карты состояния и классификации
            for f in gis_magt.png gis_zones.png gis_vulnerability.png \
                     gis_magt_2050.png map_alt_quality.png map_magt_quality.png \
                     map_classes_quality.png russia_magt_map.png \
                     russia_class_map.png vulnerability_index.png \
                     zones_permafrost.png nadym_map.png; do
                [[ -f "$f" ]] && git mv "$f" "figures/maps/$f" || true
            done

            # Валидация
            for f in scatter_spatial_cv.png alt_spatial_cv.png \
                     validate_boreholes.png compare_vs_obu.png \
                     compare_vs_obu_diff.png validation_summary.png \
                     obu_vs_gtnp.png feature_importance.png \
                     step1_climate_leakage.png step1_results.csv \
                     step2_importance.png step2_two_regions_maps.png \
                     step4_spatial_cv.png loro_5regions.png; do
                [[ -f "$f" ]] && git mv "$f" "figures/validation/$f" || true
            done

            # Тренд
            for f in trend_final.png change_final.png trend_magt.png \
                     trend_magt_clean.png trend_magt_long.png \
                     change_magt_2010_2023.png change_magt_clean.png \
                     change_alt_2010_2023.png change_alt_clean.png \
                     verify_trend_esacci.png; do
                [[ -f "$f" ]] && git mv "$f" "figures/trend/$f" || true
            done

            # Прогноз
            for f in forecast_2030.png forecast_2050.png \
                     forecast_magt_2050.png gis_forecast_2050.png; do
                [[ -f "$f" ]] && git mv "$f" "figures/forecast/$f" || true
            done

            # Классификация (boxplot)
            for f in alt_by_zone.png; do
                [[ -f "$f" ]] && git mv "$f" "figures/classification/$f" || true
            done

            # Таблицы (csv) — все в tables/
            for f in *.csv; do
                [[ -f "$f" ]] && git mv "$f" "tables/$f" || true
            done

            cd ../..
        fi
    fi
fi

echo
echo "=== ГОТОВО ==="
if $DRY_RUN; then
    echo "Это был сухой прогон. Если план устраивает, запустите без --dry-run."
else
    echo "Миграция выполнена. Что дальше:"
    echo "  1. Проверить структуру:  tree -L 4 ml/"
    echo "  2. Создать ml/README.md, ml/requirements.txt, ml/experiments/README.md"
    echo "  3. Обновить пути в скриптах (см. отдельный документ)"
    echo "  4. git status — посмотреть, всё ли учтено"
    echo "  5. git commit -m 'restructure: reorganize ML code into ml/ subdirectory'"
fi