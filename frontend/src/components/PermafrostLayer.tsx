import { useEffect, useRef } from 'react';
import { useMapStore } from '../store/mapStore';
import type { FeatureCollection, PolygonGeometry, PermafrostFeatureProperties, PermafrostType } from '../types';
import { PERMAFROST_TYPE_COLORS, PERMAFROST_TYPE_LABELS } from '../types';

const LAYER_ID_FILL = 'permafrost-layer-fill';
const LAYER_ID_OUTLINE = 'permafrost-layer-outline';
const SOURCE_ID = 'permafrost-source';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MapLike = any;

export function PermafrostLayer() {
  const mapRef = useRef<MapLike>(null);
  const initializedRef = useRef(false);
  const layerConfigRef = useRef<{ fillAdded: boolean; outlineAdded: boolean }>({
    fillAdded: false,
    outlineAdded: false,
  });

  const geojsonData = useMapStore((s) => s.geojsonData);
  const isLoading = useMapStore((s) => s.isLoading);
  const error = useMapStore((s) => s.error);
  const selectedPermafrostType = useMapStore((s) => s.selectedPermafrostType);

  // Register map instance via custom event
  useEffect(() => {
    const handler = (e: CustomEvent) => {
      mapRef.current = e.detail;
      initializedRef.current = true;
    };

    window.addEventListener('maplibre-map-ready', handler as EventListener);

    return () => {
      window.removeEventListener('maplibre-map-ready', handler as EventListener);
    };
  }, []);

  // Update map layer when geojsonData changes (triggered by year change in store)
  useEffect(() => {
    if (!mapRef.current || !initializedRef.current) return;

    const map = mapRef.current;

    if (geojsonData && geojsonData.features.length > 0) {
      // Update source data if exists, otherwise add new
      if (map.hasSource(SOURCE_ID)) {
        const source = map.getSource(SOURCE_ID);
        source.setData(geojsonData);
      } else {
        addPermafrostLayers(map, geojsonData);
      }

      // Apply filter based on selected type
      applyTypeFilter(map, selectedPermafrostType);
    } else if (geojsonData && geojsonData.features.length === 0) {
      // No features for this year - remove layers
      removePermafrostLayers(map);
    }
  }, [geojsonData, selectedPermafrostType]);

  return null;
}

function addPermafrostLayers(
  map: MapLike,
  geojsonData: FeatureCollection<PolygonGeometry, PermafrostFeatureProperties>
) {
  const config = layerConfigRef.current;

  // Add source
  if (!map.hasSource(SOURCE_ID)) {
    map.addSource(SOURCE_ID, {
      type: 'geojson',
      data: geojsonData,
      buffer: 128,
      tolerance: 0.5,
    });
  }

  // Add fill layer
  if (!config.fillAdded) {
    map.addLayer({
      id: LAYER_ID_FILL,
      type: 'fill',
      source: SOURCE_ID,
      paint: {
        'fill-color': [
          'match',
          ['get', 'type'],
          'continuous', PERMAFROST_TYPE_COLORS.continuous,
          'discontinuous', PERMAFROST_TYPE_COLORS.discontinuous,
          'isolated', PERMAFROST_TYPE_COLORS.isolated,
          'relic', PERMAFROST_TYPE_COLORS.relic,
          '#d4e6f1', // default
        ],
        'fill-opacity': 0.55,
      },
    });

    // Add hover interaction for fill layer
    map.on('mousemove', LAYER_ID_FILL, (e: MapLike) => {
      if (e.features && e.features.length > 0) {
        const feature = e.features[0];
        const props = feature.properties || {};

        map.getCanvas().style.cursor = 'pointer';

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const popup = new (window as unknown as Record<string, any>).maplibregl.Popup({ offset: 10, closeButton: false, closeOnClick: false });
        popup.setLngLat(e.lngLat);
        popup.setHTML(buildTooltipHTML(props));
        popup.addTo(map);
      }
    });

    map.on('mouseleave', LAYER_ID_FILL, () => {
      map.getCanvas().style.cursor = '';
      // Remove all popups
      const style = map.getStyle();
      if (style && style.layers) {
        // Popups are managed internally by maplibregl
      }
    });

    config.fillAdded = true;
  }

  // Add outline layer
  if (!config.outlineAdded) {
    map.addLayer({
      id: LAYER_ID_OUTLINE,
      type: 'line',
      source: SOURCE_ID,
      paint: {
        'line-color': '#ffffff',
        'line-width': 1.2,
        'line-opacity': 0.7,
      },
    });

    config.outlineAdded = true;
  }

  // Reorder: outline on top of fill
  const layers = map.getStyle().layers;
  const firstSymbolLayer = layers.find((l: MapLike) => l.type === 'symbol');
  if (firstSymbolLayer && map.getLayer(LAYER_ID_OUTLINE)) {
    map.moveLayer(LAYER_ID_OUTLINE, firstSymbolLayer.id);
  }
}

function removePermafrostLayers(map: MapLike) {
  const config = layerConfigRef.current;

  if (config.fillAdded) {
    map.removeLayer(LAYER_ID_FILL);
    map.off('mousemove', LAYER_ID_FILL);
    map.off('mouseleave', LAYER_ID_FILL);
    config.fillAdded = false;
  }

  if (config.outlineAdded) {
    map.removeLayer(LAYER_ID_OUTLINE);
    config.outlineAdded = false;
  }

  if (map.hasSource(SOURCE_ID)) {
    map.removeSource(SOURCE_ID);
  }
}

function applyTypeFilter(map: MapLike, selectedType: PermafrostType | 'all') {
  if (!map.hasLayer(LAYER_ID_FILL) && !map.hasLayer(LAYER_ID_OUTLINE)) return;

  if (selectedType === 'all') {
    map.setFilter(LAYER_ID_FILL, null);
    map.setFilter(LAYER_ID_OUTLINE, null);
  } else {
    const filter: unknown[] = ['==', 'type', selectedType];
    map.setFilter(LAYER_ID_FILL, filter);
    map.setFilter(LAYER_ID_OUTLINE, filter);
  }
}

function buildTooltipHTML(props: PermafrostFeatureProperties): string {
  const year = props.year ?? 'N/A';
  const typeRaw = props.type ?? 'N/A';
  const typeLabel = PERMAFROST_TYPE_LABELS[typeRaw as PermafrostType] || typeRaw;
  const typeColor = PERMAFROST_TYPE_COLORS[typeRaw as PermafrostType] || '#d4e6f1';
  const area = props.area_km2 != null ? Number(props.area_km2).toFixed(1) : 'N/A';
  const source = props.source ?? 'N/A';

  return `
    <div class="p-2 min-w-[180px] font-sans">
      <h3 class="font-bold text-sm mb-2 text-gray-800 border-b pb-1">Данные вечной мерзлоты</h3>
      <div class="space-y-1.5 text-xs text-gray-700">
        <div class="flex justify-between">
          <span class="font-semibold">Год:</span>
          <span>${year}</span>
        </div>
        <div class="flex justify-between items-center">
          <span class="font-semibold">Тип:</span>
          <span class="inline-flex items-center gap-1.5">
            <span class="w-3 h-3 rounded-sm inline-block" style="background-color: ${typeColor}"></span>
            ${typeLabel}
          </span>
        </div>
        <div class="flex justify-between">
          <span class="font-semibold">Площадь:</span>
          <span>${area} км²</span>
        </div>
        <div class="flex justify-between">
          <span class="font-semibold">Источник:</span>
          <span class="truncate max-w-[120px]" title="${source}">${source}</span>
        </div>
      </div>
    </div>
  `;
}