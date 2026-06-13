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