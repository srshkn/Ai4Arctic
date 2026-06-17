// ==================== Auth Types ====================

export interface LoginRequest {
  name: string; // maps to "email" in UI
  password: string;
}

export interface RegisterRequest {
  name: string; // maps to "email" in UI
  password: string;
  confirm_password: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export interface UserOut {
  id: string;
  name: string;
}

// ==================== Permafrost Types ====================

export type PermafrostType = 'continuous' | 'discontinuous' | 'isolated' | 'relic';

export const PERMAFROST_TYPE_LABELS: Record<PermafrostType, string> = {
  continuous: 'Сплошная',
  discontinuous: 'Прерывистая',
  isolated: 'Островная',
  relic: 'Реликтовая',
};

export const PERMAFROST_TYPE_COLORS: Record<PermafrostType, string> = {
  continuous: '#1a5276',
  discontinuous: '#2e86c1',
  isolated: '#85c1e9',
  relic: '#d4e6f1',
};

export interface PermafrostProperties {
  year?: number;
  type?: PermafrostType;
  area_km2?: number;
  source?: string;
}

// ==================== MAGT Class Types (for heat regime layers) ====================

/** Configuration for a single MAGT class layer (class_id 1-15). Color is taken from GeoJSON data. */
export interface MagtClassConfig {
  id: number;
  label: string;
  color: string;
  temperatureRange: string;
}

/**
 * Temperature range labels for MAGT classes 1-15.
 * class_id 1 = coldest, class_id 15 = warmest (> 8°C)
 * Color is dynamically loaded from GeoJSON data at runtime.
 */
export const MAGT_CLASS_RANGES: Array<{ id: number; label: string; temperatureRange: string }> = [
  { id: 1, label: 'Класс 1', temperatureRange: '< 0.0°C' },
  { id: 2, label: 'Класс 2', temperatureRange: '0.0..0.5°C' },
  { id: 3, label: 'Класс 3', temperatureRange: '0.5..1.0°C' },
  { id: 4, label: 'Класс 4', temperatureRange: '1.0..1.5°C' },
  { id: 5, label: 'Класс 5', temperatureRange: '1.5..2.0°C' },
  { id: 6, label: 'Класс 6', temperatureRange: '2.0..2.5°C' },
  { id: 7, label: 'Класс 7', temperatureRange: '2.5..3.0°C' },
  { id: 8, label: 'Класс 8', temperatureRange: '3.0..3.5°C' },
  { id: 9, label: 'Класс 9', temperatureRange: '3.5..4.0°C' },
  { id: 10, label: 'Класс 10', temperatureRange: '4.0..4.5°C' },
  { id: 11, label: 'Класс 11', temperatureRange: '4.5..5.0°C' },
  { id: 12, label: 'Класс 12', temperatureRange: '5.0..5.5°C' },
  { id: 13, label: 'Класс 13', temperatureRange: '5.5..6.0°C' },
  { id: 14, label: 'Класс 14', temperatureRange: '6.0..7.0°C' },
  { id: 15, label: 'Класс 15', temperatureRange: '> 8.0°C' },
];

/** Lookup map: class_id → range info */
export const MAGT_CLASS_RANGE_MAP: Record<number, { id: number; label: string; temperatureRange: string }> = {};
MAGT_CLASS_RANGES.forEach((range) => {
  MAGT_CLASS_RANGE_MAP[range.id] = range;
});

/** Build MagtClassConfig[] with colors from GeoJSON feature properties */
export function buildMagtClassConfigs(geojsonData: FeatureCollection<PolygonGeometry, PermafrostFeatureProperties> | null): MagtClassConfig[] {
  if (!geojsonData || !geojsonData.features) return [];

  // Collect the first occurrence color for each class_id
  const colorMap = new Map<number, string>();
  for (const feature of geojsonData.features) {
    const props = feature.properties as PermafrostFeatureProperties;
    const classId = props.class_id;
    const color = props.color;
    if (classId != null && typeof classId === 'number' && Number.isInteger(classId) && color) {
      if (!colorMap.has(classId)) {
        colorMap.set(classId, color as string);
      }
    }
  }

  // Build configs using MAGT_CLASS_RANGES as base + colors from GeoJSON
  const result: MagtClassConfig[] = [];
  for (const range of MAGT_CLASS_RANGES) {
    const color = colorMap.get(range.id);
    if (color) {
      result.push({
        id: range.id,
        label: range.label,
        color,
        temperatureRange: range.temperatureRange,
      });
    }
  }
  return result;
}

// ==================== GeoJSON Types (RFC 7946) ====================

export interface GeoJsonObject {
  type: string;
}

export type Geometry =
  | PointGeometry
  | MultiPointGeometry
  | LineStringGeometry
  | MultiLineStringGeometry
  | PolygonGeometry
  | MultiPolygonGeometry
  | GeometryCollection;

export interface PointGeometry extends GeoJsonObject {
  type: 'Point';
  coordinates: [number, number];
}

export interface MultiPointGeometry extends GeoJsonObject {
  type: 'MultiPoint';
  coordinates: [number, number][];
}

export interface LineStringGeometry extends GeoJsonObject {
  type: 'LineString';
  coordinates: [number, number][];
}

export interface MultiLineStringGeometry extends GeoJsonObject {
  type: 'MultiLineString';
  coordinates: [number, number][][];
}

export interface PolygonGeometry extends GeoJsonObject {
  type: 'Polygon';
  coordinates: [number, number][][];
}

export interface MultiPolygonGeometry extends GeoJsonObject {
  type: 'MultiPolygon';
  coordinates: [number, number][][][];
}

export interface GeometryCollection extends GeoJsonObject {
  type: 'GeometryCollection';
  geometries: Geometry[];
}

export interface GeoJsonProperties {
  [key: string]: unknown;
}

export interface Feature<T extends Geometry = Geometry, P = GeoJsonProperties>
  extends GeoJsonObject {
  type: 'Feature';
  geometry: T;
  properties: P;
  id?: string | number;
}

export interface FeatureCollection<
  T extends Geometry = Geometry,
  P = GeoJsonProperties
> extends GeoJsonObject {
  type: 'FeatureCollection';
  features: Feature<T, P>[];
}

// ==================== Permafrost GeoJSON Types ====================

export interface PermafrostFeatureProperties {
  year?: number;
  type?: string;
  area_km2?: number | string;
  source?: string;
  [key: string]: unknown;
}

export type PermafrostFeature = Feature<PolygonGeometry, PermafrostFeatureProperties>;

// ==================== Map State Types ====================

export interface MapState {
  year: number;
  zoom: number;
  center: [number, number];
  selectedPermafrostType: PermafrostType | 'all';
  isLoading: boolean;
  error: string | null;
}

// ==================== API Response Types ====================

export interface ApiResponse<T = unknown> {
  data: T;
  status: number;
  message?: string;
}

export interface ApiError {
  status: number;
  message: string;
  details?: unknown;
}

// ==================== Component Props Types ====================

export interface SliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  label?: string;
  className?: string;
}

export interface LegendItem {
  type: PermafrostType;
  label: string;
  color: string;
  isActive: boolean;
  onClick: () => void;
}

export interface TooltipData {
  year?: number;
  type?: string;
  area_km2?: number | string;
  source?: string;
}

export interface AuthFormErrors {
  name?: string;
  password?: string;
  general?: string;
}