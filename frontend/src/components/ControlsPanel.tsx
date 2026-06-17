import { useMemo } from 'react';
import type { MagtClassConfig } from '../types';

interface ControlsPanelProps {
  year: number;
  onYearChange: (year: number) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  magtClasses: MagtClassConfig[];
  layerVisibility: Record<number, boolean>;
  onToggleLayer: (classId: number) => void;
  onSetAllLayersVisible: (visible: boolean) => void;
  isLoading: boolean;
  error: string | null;
}

export function ControlsPanel({
  year,
  onYearChange,
  zoom,
  onZoomChange,
  magtClasses,
  layerVisibility,
  onToggleLayer,
  onSetAllLayersVisible,
  isLoading,
  error,
}: ControlsPanelProps) {
  const MIN_YEAR = 2007;
  const MAX_YEAR = 2035;
  const MIN_ZOOM = 2;
  const MAX_ZOOM = 10;

  const yearPercentage = useMemo(
    () => ((year - MIN_YEAR) / (MAX_YEAR - MIN_YEAR)) * 100,
    [year]
  );

  const zoomPercentage = useMemo(
    () => ((zoom - MIN_ZOOM) / (MAX_ZOOM - MIN_ZOOM)) * 100,
    [zoom]
  );

  // Count how many layers are visible
  const visibleCount = magtClasses.filter((cfg) => layerVisibility[cfg.id]).length;
  const allVisible = magtClasses.length > 0 && visibleCount === magtClasses.length;

  return (
    <div className="absolute top-4 left-4 z-[1000] flex flex-col gap-3 max-w-xs">
      {/* Year Slider */}
      <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-4 transition-all duration-200 hover:shadow-xl">
        <label className="block text-sm font-semibold text-gray-800 mb-2">
          Год: <span className="text-blue-600 text-lg">{year}</span>
        </label>
        <input
          type="range"
          min={MIN_YEAR}
          max={MAX_YEAR}
          value={year}
          onChange={(e) => onYearChange(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          style={{
            background: `linear-gradient(to right, #2563eb 0%, #2563eb ${yearPercentage}%, #e5e7eb ${yearPercentage}%, #e5e7eb 100%)`,
          }}
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>{MIN_YEAR}</span>
          <span>{MAX_YEAR}</span>
        </div>
      </div>

      {/* Zoom Slider */}
      <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-4 transition-all duration-200 hover:shadow-xl">
        <label className="block text-sm font-semibold text-gray-800 mb-2">
          Масштаб: <span className="text-blue-600 text-lg">{zoom.toFixed(1)}x</span>
        </label>
        <input
          type="range"
          min={MIN_ZOOM}
          max={MAX_ZOOM}
          step={0.5}
          value={zoom}
          onChange={(e) => onZoomChange(Number(e.target.value))}
          className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          style={{
            background: `linear-gradient(to right, #2563eb 0%, #2563eb ${zoomPercentage}%, #e5e7eb ${zoomPercentage}%, #e5e7eb 100%)`,
          }}
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>2x</span>
          <span>{MAX_ZOOM}x</span>
        </div>
      </div>

      {/* MAGT Layer Legend */}
      <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-4 transition-all duration-200 hover:shadow-xl">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-gray-800">Температура мерзлоты</h3>
          <div className="flex gap-1">
            <button
              onClick={() => onSetAllLayersVisible(true)}
              className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded hover:bg-green-200 transition-colors"
              title="Выбрать все"
            >
              Все
            </button>
            <button
              onClick={() => onSetAllLayersVisible(false)}
              className="text-xs px-2 py-0.5 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
              title="Снять все"
            >
              Ни одного
            </button>
          </div>
        </div>
        <div className="space-y-0.5 max-h-[40vh] overflow-y-auto pr-1">
          {magtClasses.map((cfg) => {
            const isVisible = layerVisibility[cfg.id] ?? true;
            return (
              <button
                key={cfg.id}
                onClick={() => onToggleLayer(cfg.id)}
                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-all duration-150 ${
                  isVisible
                    ? 'bg-gray-50 text-gray-800'
                    : 'opacity-40 text-gray-400 bg-gray-50/50'
                }`}
              >
                {/* Checkbox */}
                <div className={`w-4 h-4 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors ${
                  isVisible
                    ? 'border-blue-500 bg-blue-500'
                    : 'border-gray-300 bg-white'
                }`}>
                  {isVisible && (
                    <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                {/* Color swatch */}
                <span
                  className="w-5 h-4 rounded border border-gray-300 flex-shrink-0"
                  style={{ backgroundColor: cfg.color }}
                />
                {/* Label */}
                <span className="flex-1 text-left truncate">{cfg.label}</span>
              </button>
            );
          })}
        </div>
        {/* Layer count */}
        <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500 text-center">
          {visibleCount} из {magtClasses.length} слоев
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-3 flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600">Загрузка данных...</span>
        </div>
      )}

      {/* Error indicator */}
      {error && (
        <div className="bg-red-50 backdrop-blur-sm rounded-xl shadow-lg border border-red-200 p-3">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <span className="text-sm text-red-700 font-medium">Ошибка:</span>
              <p className="text-xs text-red-600 mt-1">{error}</p>
              <button
                onClick={() => onYearChange(year)}
                className="text-xs text-blue-600 hover:text-blue-700 mt-1 underline"
              >
                Повторить
              </button>
            </div>
          </div>
        </div>
      )}

      {/* No data message */}
      {!isLoading && !error && magtClasses.length === 0 && (
        <div className="bg-yellow-50 backdrop-blur-sm rounded-xl shadow-lg border border-yellow-200 p-3">
          <span className="text-sm text-yellow-700">Нет данных для отображения. Проверьте консоль разработчика для деталей.</span>
        </div>
      )}
    </div>
  );
}
