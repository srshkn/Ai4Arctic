// Система авторизации отключена — authStore больше не используется
import { useMapStore } from '../store/mapStore';
import { MapContainer } from '../components/MapContainer';
import { ControlsPanel } from '../components/ControlsPanel';

export function MapPage() {
  const year = useMapStore((state) => state.year);
  const zoom = useMapStore((state) => state.zoom);
  const isLoading = useMapStore((state) => state.isLoading);
  const error = useMapStore((state) => state.error);
  const geojsonData = useMapStore((state) => state.geojsonData);
  const magtClasses = useMapStore((state) => state.magtClasses);
  const layerVisibility = useMapStore((state) => state.layerVisibility);
  const setYear = useMapStore((state) => state.setYear);
  const setZoom = useMapStore((state) => state.setZoom);
  const toggleLayer = useMapStore((state) => state.toggleLayer);
  const setAllLayersVisible = useMapStore((state) => state.setAllLayersVisible);

  return (
    <div className="relative w-full h-screen overflow-hidden bg-gray-900">
      {/* Header */}
      <header className="absolute top-0 left-0 right-0 z-[1000] flex items-center justify-between px-4 py-3 bg-black/50 backdrop-blur-sm pointer-events-auto">
        <div className="flex items-center gap-3">
          <h1 className="text-white font-bold text-lg truncate">Ai4Arctic — Вечная мерзлота</h1>
        </div>
      </header>

      {/* Map */}
      <MapContainer
        center={[100, 65]}
        zoom={zoom}
        onZoomChange={setZoom}
        geojsonData={geojsonData}
        selectedType="all"
        layerVisibility={layerVisibility}
        year={year}
      />

      {/* Controls Panel */}
      <ControlsPanel
        year={year}
        onYearChange={setYear}
        zoom={zoom}
        onZoomChange={setZoom}
        magtClasses={magtClasses}
        layerVisibility={layerVisibility}
        onToggleLayer={toggleLayer}
        onSetAllLayersVisible={setAllLayersVisible}
        isLoading={isLoading}
        error={error}
      />
    </div>
  );
}
