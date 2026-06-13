// Type declarations for MapLibre GL JS CSS import
declare module 'maplibre-gl/dist/maplibre-gl.css' {}

// Extend maplibregl namespace for additional types
declare global {
  namespace maplibregl {
    interface StyleObject {
      version?: number;
      name?: string;
      metadata?: Record<string, unknown>;
      center?: [number, number];
      zoom?: number;
      bearing?: number;
      pitch?: number;
      lightSource?: LightSource;
      models?: Record<string, unknown>;
      sources?: Record<string, Source>;
      sprite?: string | string[];
      glyphs?: string;
      transition?: Transition;
      layers?: Layer[];
      [key: string]: unknown;
    }

    interface LightSource {
      color?: string;
      intensity?: number;
      anchor?: 'map' | 'viewport';
      position?: [number, number, number];
    }

    interface Transition {
      duration?: number;
      delay?: number;
    }

    // Minimal Map instance type for component usage
    interface CustomMapInstance {
      addSource(id: string, source: unknown): void;
      getSource(id: string): unknown;
      hasSource(id: string): boolean;
      removeSource(id: string): void;
      addLayer(layer: unknown): void;
      getLayer(id: string): unknown | null;
      hasLayer(id: string): boolean;
      removeLayer(id: string): void;
      moveLayer(id: string, beforeId?: string): void;
      setFilter(id: string, filter: unknown): void;
      getStyle(): { layers: unknown[] };
      on(event: string, handler: (...args: unknown[]) => void): void;
      off(event: string, handler?: (...args: unknown[]) => void): void;
      flyTo(config: unknown): this;
      fitBounds(bounds: [number, number, number, number], options?: Record<string, unknown>): this;
      project(lngLat: { lng: number; lat: number }): { x: number; y: number };
      unproject(point: { x: number; y: number }): { lng: number; lat: number };
      getCanvas(): HTMLCanvasElement;
    }
  }
}

export {};