import { api } from './client';
import type { FeatureCollection, PolygonGeometry, PermafrostFeatureProperties } from '../types';

const ENDPOINTS = {
  permafrost: '/permafrost',
};

export async function getPermafrostData(year: number): Promise<FeatureCollection<PolygonGeometry, PermafrostFeatureProperties>> {
  const response = await api.get<FeatureCollection<PolygonGeometry, PermafrostFeatureProperties>>(
    `${ENDPOINTS.permafrost}/${year}`
  );
  return response.data;
}

export async function getAllPermafrostYears(): Promise<number[]> {
  const response = await api.get<number[]>(`${ENDPOINTS.permafrost}/years`);
  return response.data;
}

export async function loadLocalGeoJSON(filePath: string): Promise<FeatureCollection<PolygonGeometry, PermafrostFeatureProperties>> {
  const response = await fetch(filePath);
  if (!response.ok) {
    throw new Error(`Failed to load GeoJSON from ${filePath}: ${response.statusText}`);
  }
  return response.json();
}