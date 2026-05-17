import { create } from 'zustand';
import type { FeatureCollection, PolygonGeometry, PermafrostFeatureProperties, PermafrostType } from '../types';
import { getPermafrostData } from '../api/permafrost';

interface MapState {
  year: number;
  zoom: number;
  center: [number, number];
  selectedPermafrostType: PermafrostType | 'all';
  isLoading: boolean;
  error: string | null;
  geojsonData: FeatureCollection<PolygonGeometry, PermafrostFeatureProperties> | null;
  availableYears: number[];

  setYear: (year: number) => void;
  setZoom: (zoom: number) => void;
  setCenter: (center: [number, number]) => void;
  setSelectedPermafrostType: (type: PermafrostType | 'all') => void;
  loadPermafrostData: (year: number) => Promise<void>;
  setAvailableYears: (years: number[]) => void;
}

export const useMapStore = create<MapState>((set, get) => ({
  year: 2024,
  zoom: 4,
  center: [100, 65] as [number, number],
  selectedPermafrostType: 'all',
  isLoading: false,
  error: null,
  geojsonData: null,
  availableYears: [],

  setYear: (year: number) => {
    set({ year });
    get().loadPermafrostData(year);
  },

  setZoom: (zoom: number) => {
    set({ zoom });
  },

  setCenter: (center: [number, number]) => {
    set({ center });
  },

  setSelectedPermafrostType: (type: PermafrostType | 'all') => {
    set({ selectedPermafrostType: type });
  },

  loadPermafrostData: async (year: number) => {
    const state = get();
    // Skip if already loading this year's data
    if (state.geojsonData && state.year === year) return;

    set({ isLoading: true, error: null });
    try {
      const data = await getPermafrostData(year);
      set({ geojsonData: data, isLoading: false });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Ошибка загрузки данных';
      set({ error: message, isLoading: false });
    }
  },

  setAvailableYears: (years: number[]) => {
    set({ availableYears: years });
  },
}));