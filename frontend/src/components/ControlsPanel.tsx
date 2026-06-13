import { useMemo } from 'react';
import type { PermafrostType } from '../types';
import {
  PERMAFROST_TYPE_LABELS,
  PERMAFROST_TYPE_COLORS,
} from '../types';

interface ControlsPanelProps {
  year: number;
  onYearChange: (year: number) => void;
  zoom: number;
  onZoomChange: (zoom: number) => void;
  selectedType: PermafrostType | 'all';
  onTypeChange: (type: PermafrostType | 'all') => void;
  isLoading: boolean;
}

export function ControlsPanel({
  year,
  onYearChange,
  zoom,
  onZoomChange,
  selectedType,
  onTypeChange,
  isLoading,
}: ControlsPanelProps) {
  const MIN_YEAR = 1990;
  const MAX_YEAR = 2025;
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

      {/* Legend */}
      <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-4 transition-all duration-200 hover:shadow-xl">
        <h3 className="text-sm font-semibold text-gray-800 mb-2">Легенда</h3>
        <div className="space-y-1">
          {/* All types button */}
          <button
            onClick={() => onTypeChange('all')}
            className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors duration-150 ${
              selectedType === 'all'
                ? 'bg-blue-100 text-blue-800 font-medium ring-2 ring-blue-300'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            <span className="w-4 h-4 rounded bg-gradient-to-br from-blue-900 to-sky-200 border border-gray-300" />
            Все типы
          </button>

          {(Object.keys(PERMAFROST_TYPE_LABELS) as PermafrostType[]).map((type) => (
            <button
              key={type}
              onClick={() => onTypeChange(type)}
              className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-sm transition-colors duration-150 ${
                selectedType === type
                  ? 'bg-blue-100 text-blue-800 font-medium ring-2 ring-blue-300'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              <span
                className="w-4 h-4 rounded border border-gray-300"
                style={{ backgroundColor: PERMAFROST_TYPE_COLORS[type] }}
              />
              {PERMAFROST_TYPE_LABELS[type]}
            </button>
          ))}
        </div>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="bg-white/95 backdrop-blur-sm rounded-xl shadow-lg border border-gray-200 p-3 flex items-center gap-2">
          <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
          <span className="text-sm text-gray-600">Загрузка данных...</span>
        </div>
      )}
    </div>
  );
}