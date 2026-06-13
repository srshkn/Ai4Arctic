import { useEffect, useRef } from 'react';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const maplibregl: any;

interface MapContainerProps {
  center: [number, number];
  zoom: number;
  onZoomChange: (zoom: number) => void;
  children?: React.ReactNode;
}

export function MapContainer({ center, zoom, onZoomChange, children }: MapContainerProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mapInstanceRef = useRef<any | null>(null);
  const initializedRef = useRef(false);

  // Initialize the map once
  useEffect(() => {
    if (!mapRef.current || initializedRef.current || !maplibregl) return;

    const map = new maplibregl.Map({
      container: mapRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            url: 'https://tile.openstreetmap.org/20.json',
            tileSize: 256,
          },
        },
        layers: [
          {
            id: 'osm-tiles',
            type: 'raster',
            source: 'osm',
            minzoom: 0,
            maxzoom: 19,
          },
        ],
      },
      center: center,
      zoom: zoom,
      minZoom: 2,
      maxZoom: 10,
    });

    // Add navigation controls
    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    // Add scale control
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    mapInstanceRef.current = map;
    initializedRef.current = true;

    // Sync zoom with store
    map.on('zoomend', () => {
      const currentZoom = map.getZoom();
      onZoomChange(currentZoom);
    });

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []); // Only run once

  // Sync center changes from store
  useEffect(() => {
    if (!mapInstanceRef.current || !initializedRef.current) return;
    mapInstanceRef.current.flyTo({
      center: center,
      duration: 800,
      essential: true,
    });
  }, [center]);

  // Sync zoom changes from store (when changed via slider)
  const prevZoomRef = useRef(zoom);
  useEffect(() => {
    if (!mapInstanceRef.current || !initializedRef.current) return;
    if (zoom !== prevZoomRef.current) {
      mapInstanceRef.current.zoomTo(zoom, { duration: 300, essential: true });
      prevZoomRef.current = zoom;
    }
  }, [zoom]);

  return (
    <div className="relative w-full h-screen">
      <div ref={mapRef} className="w-full h-full" />
      {children}
    </div>
  );
}