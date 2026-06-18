import { useEffect, useMemo } from 'react';
import { MapContainer as LeafletMap, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import type { FeatureCollection, PolygonGeometry, PermafrostFeatureProperties } from '../types';

// Fix default marker icon issue in react-leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
});

interface MapContainerProps {
  center: [number, number];
  zoom: number;
  onZoomChange: (zoom: number) => void;
  geojsonData: FeatureCollection<PolygonGeometry, PermafrostFeatureProperties> | null;
  selectedType: string | 'all';
  layerVisibility: Record<number, boolean>;
  year?: number;
}

// Component to handle zoom changes from map interaction
function MapController({ onZoomChange }: { onZoomChange: (zoom: number) => void }) {
  const map = useMap();

  useEffect(() => {
    map.on('zoomend', () => {
      onZoomChange(map.getZoom());
    });
  }, [map, onZoomChange]);

  return null;
}

/**
 * Get the style for a GeoJSON feature based on its class_id and layer visibility.
 * Features use their own `color` property from the GeoJSON data.
 */
function styleFeature(
  featureProperties: PermafrostFeatureProperties,
  _selectedType: string | 'all',
  layerVisibility: Record<number, boolean>
): L.PathOptions {
  const classId = featureProperties.class_id;

  // If no class_id, make feature invisible
  if (classId == null) {
    return {
      fillColor: 'transparent',
      fillOpacity: 0,
      color: 'transparent',
      weight: 0,
    };
  }

  const numericClassId = Number(classId);

  // Check if this layer is visible
  if (!layerVisibility[numericClassId]) {
    return {
      fillColor: 'transparent',
      fillOpacity: 0,
      color: 'transparent',
      weight: 0,
    };
  }

  // Use the color directly from the GeoJSON feature properties
  const color = (featureProperties.color as string) || '#95a5a6';

  return {
    fillColor: color,
    fillOpacity: 0.75,
    color: 'rgba(255, 255, 255, 0.4)',
    weight: 0.5,
    opacity: 1,
  };
}

// Bounds for Russia territory: [southWestCorner, northEastCorner]
const RUSSIA_BOUNDS: L.LatLngBoundsLiteral = [
  [41, 19],   // South-West (Kaliningrad to Crimea area)
  [82, 180],  // North-East (Chukotka to Arctic)
];

// Zoom limits to prevent seeing the world map multiple times
const MIN_ZOOM = 3;
const MAX_ZOOM = 12;

export function MapContainer({ center, zoom, onZoomChange, geojsonData, selectedType, layerVisibility, year = 2024 }: MapContainerProps) {
  useEffect(() => {
    // Force map to update dimensions after mount
    const timer = setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 100);
    
    return () => clearTimeout(timer);
  }, []);

  // Memoize style function to prevent unnecessary re-renders
  const geoJSONStyle = useMemo(() => (feature: any) => {
    if (!feature) return { fillColor: 'transparent', fillOpacity: 0 };
    const props = feature.properties as PermafrostFeatureProperties;
    return styleFeature(props, selectedType, layerVisibility);
  }, [selectedType, layerVisibility]);

  return (
    <div style={{ width: '100vw', height: '100vh' }}>
      <LeafletMap
        center={center}
        zoom={zoom}
        minZoom={MIN_ZOOM}
        maxZoom={MAX_ZOOM}
        style={{ width: '100%', height: '100%' }}
        zoomControl={false}
        maxBounds={RUSSIA_BOUNDS}
        maxBoundsViscosity={1.0}
      >
        <TileLayer
          attribution='© AI4Arctic 2026'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {geojsonData && (
          <GeoJSON
            key={`geojson-${year}`}
            data={geojsonData}
            style={geoJSONStyle}
            pointToLayer={(geometry, latlng) => {
              // Not used for polygon data but required by react-leaflet
              return L.circleMarker(latlng, { radius: 0 });
            }}
          />
        )}
        <MapController onZoomChange={onZoomChange} />
      </LeafletMap>
    </div>
  );
}