import { useEffect } from 'react';
import { useAuthStore } from '../store/authStore';
import { useMapStore } from '../store/mapStore';
import { MapContainer } from '../components/MapContainer';
import { ControlsPanel } from '../components/ControlsPanel';

export function MapPage() {
  const { logout, user } = useAuthStore();
  const year = useMapStore((state) => state.year);
  const zoom = useMapStore((state) => state.zoom);
  const selectedType = useMapStore((state) => state.selectedPermafrostType);
  const isLoading = useMapStore((state) => state.isLoading);
  const setYear = useMapStore((state) => state.setYear);
  const setZoom = useMapStore((state) => state.setZoom);
  const setSelectedType = useMapStore((state) => state.setSelectedPermafrostType);

  // Загружаем данные пользователя при монтировании
  useEffect(() => {
    // Данные уже загружены при логине, но на случай refresh можно добавить
  }, []);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gray-900">
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 z-[400] flex items-center justify-between px-4 py-3 bg-black/50 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <h1 className="text-white font-bold text-lg truncate">Ai4Arctic — Вечная мерзлота</h1>
        </div>
        <div className="flex items-center gap-2">
          {/* User info */}
          {user && (
            <span className="text-white/80 text-sm hidden sm:block">
              {user.name}
            </span>
          )}
          {/* Profile button */}
          <button
            onClick={() => alert('Профиль пользователя')}
            className="flex items-center gap-2 px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white rounded-lg transition-colors"
            title="Профиль"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
            <span className="hidden sm:inline">Профиль</span>
          </button>
          {/* Logout button */}
          <button
            onClick={() => { logout(); }}
            className="flex items-center gap-2 px-3 py-1.5 bg-red-600/80 hover:bg-red-600 text-white rounded-lg transition-colors"
            title="Выйти"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
            <span className="hidden sm:inline">Выход</span>
          </button>
        </div>
      </header>

      {/* Map */}
      <MapContainer center={[65, 100]} zoom={zoom} onZoomChange={setZoom} />

      {/* Controls Panel */}
      <ControlsPanel
        year={year}
        onYearChange={setYear}
        zoom={zoom}
        onZoomChange={setZoom}
        selectedType={selectedType}
        onTypeChange={setSelectedType}
        isLoading={isLoading}
      />
    </div>
  );
}