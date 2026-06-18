import { create } from 'zustand';
import type { FeatureCollection, PolygonGeometry, PermafrostFeatureProperties, PermafrostType, MagtClassConfig } from '../types';
import { buildMagtClassConfigs } from '../types';
import { loadLocalGeoJSON } from '../api/permafrost';

interface MapState {
  year: number;
  zoom: number;
  center: [number, number];
  selectedPermafrostType: PermafrostType | 'all';
  isLoading: boolean;
  error: string | null;
  geojsonData: FeatureCollection<PolygonGeometry, PermafrostFeatureProperties> | null;

  // MAGT layer management — only classes present in loaded data with colors from GeoJSON
  magtClasses: MagtClassConfig[];
  layerVisibility: Record<number, boolean>;

  // Internal state for request cancellation and caching
  _abortController: AbortController | null;
  _loadedYears: Set<number>;
  _debounceTimer: ReturnType<typeof setTimeout> | null;

  setYear: (year: number) => void;
  setZoom: (zoom: number) => void;
  setCenter: (center: [number, number]) => void;
  setSelectedPermafrostType: (type: PermafrostType | 'all') => void;
  loadPermafrostData: (year: number, force?: boolean) => Promise<void>;
  toggleLayer: (classId: number) => void;
  setAllLayersVisible: (visible: boolean) => void;
}

export const useMapStore = create<MapState>((set, get) => ({
  year: 2024,
  zoom: 4,
  center: [100, 65] as [number, number],
  selectedPermafrostType: 'all',
  isLoading: false,
  error: null,
  geojsonData: null,

  // MAGT layer management — initialized with all ranges, no visibility
  magtClasses: [],
  layerVisibility: {},

  // Internal state
  _abortController: null,
  _loadedYears: new Set<number>(),
  _debounceTimer: null,

  setYear: (year: number) => {
    const state = get();
    
    console.log(`[MapStore] setYear called: ${year}`);
    
    // Clear any existing debounce timer
    if (state._debounceTimer) {
      clearTimeout(state._debounceTimer);
      console.log(`[MapStore] Cleared debounce timer`);
    }
    
    // If the same year is already selected, no need to reload
    if (state.year === year) {
      console.log(`[MapStore] Year ${year} is already active, skipping`);
      return;
    }
    
    // Debounce: wait 400ms of inactivity before loading
    const timer = setTimeout(() => {
      const currentState = get();
      
      // Cancel any in-flight request for a different year
      if (currentState._abortController) {
        console.log(`[MapStore] Aborting previous controller for year change`);
        currentState._abortController.abort();
      }

      console.log(`[MapStore] Debounced load triggered for year: ${year}`);
      set({ year });
      // Use force=true to ensure data is loaded even if year was previously cached
      currentState.loadPermafrostData(year, true);
    }, 400);

    set({ _debounceTimer: timer });
  },

  setZoom: (zoom: number) => {
    // Clamp zoom between 3 and 12 to prevent seeing the world map multiple times
    const clampedZoom = Math.max(3, Math.min(12, zoom));
    set({ zoom: clampedZoom });
  },

  setCenter: (center: [number, number]) => {
    set({ center });
  },

  setSelectedPermafrostType: (type: PermafrostType | 'all') => {
    set({ selectedPermafrostType: type });
  },

  loadPermafrostData: async (year: number, force: boolean = false) => {
    const state = get();
    
    console.log(`[MapStore] Loading permafrost data for year: ${year}, force: ${force}`);
    
    // If force is true, always reload the data for this year
    // If force is false, skip if already loading or cached (and it's the current year)
    if (!force) {
      if (state.isLoading && state.year === year) {
        console.log(`[MapStore] Skipping load - already loading: ${year}`);
        return;
      }
      if (state._loadedYears.has(year) && state.year === year) {
        console.log(`[MapStore] Skipping load - cached: ${year}`);
        return;
      }
    }

    // If the previous request is for a different year, abort it first
    if (state._abortController && state.year !== year) {
      console.log(`[MapStore] Aborting previous request for different year`);
      state._abortController.abort();
    }

    // Wait a tick for the abort to be processed before starting new request
    const controller = new AbortController();
    
    set({ 
      _abortController: controller, 
      isLoading: true, 
      error: null,
    });

    try {
      const filePath = `/data/magt_smooth_${year}.geojson`;
      console.log(`[MapStore] Fetching from: ${filePath}`);
      
      // Use fetch directly with signal for cancellation support
      const response = await fetch(filePath, { 
        signal: controller.signal,
        headers: {
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        }
      });
      
      console.log(`[MapStore] Response status: ${response.status} ${response.statusText}`);
      
      if (!response.ok) {
        throw new Error(`Failed to load GeoJSON from ${filePath}: ${response.statusText} (HTTP ${response.status})`);
      }
      
      const data = await response.json();
      console.log(`[MapStore] Successfully loaded ${data.features?.length || 0} features for year ${year}`);

      // Check if this is still the relevant year after async operation
      const currentState = get();
      if (currentState.year !== year) {
        console.log(`[MapStore] Year changed during fetch, skipping: ${year} -> ${currentState.year}`);
        return;
      }

      // Build magtClasses with real colors from GeoJSON
      const magtClasses = buildMagtClassConfigs(data);
      
      console.log(`[MapStore] Built ${magtClasses.length} MAGT classes for year ${year}`);
      
      // Build visibility map — preserve existing visibility for overlapping class_ids, default to true
      const newVisibility: Record<number, boolean> = {};
      magtClasses.forEach((cfg: MagtClassConfig) => {
        // Preserve existing visibility if class_id already exists, otherwise default to true
        newVisibility[cfg.id] = currentState.layerVisibility[cfg.id] ?? true;
      });

      console.log(`[MapStore] Layer visibility:`, newVisibility);

      // Update loaded years cache
      const newLoadedYears = new Set(currentState._loadedYears);
      newLoadedYears.add(year);

      set({ 
        geojsonData: data, 
        magtClasses, 
        layerVisibility: newVisibility, 
        isLoading: false, 
        error: null,
        _loadedYears: newLoadedYears,
        _abortController: null,
      });
      
      console.log(`[MapStore] Map updated with data for year ${year}`);
    } catch (error: unknown) {
      // Ignore abort errors — they are expected when year changes
      if (error instanceof DOMException && error.name === 'AbortError') {
        return;
      }
      
      const message = error instanceof Error ? error.message : 'Ошибка загрузки данных';
      set({ 
        error: message, 
        isLoading: false, 
        geojsonData: null, 
        magtClasses: [], 
        layerVisibility: {},
        _abortController: null,
      });
    }
  },

  toggleLayer: (classId: number) => {
    const state = get();
    const currentVisibility = state.layerVisibility[classId] ?? true;
    set({ layerVisibility: { ...state.layerVisibility, [classId]: !currentVisibility } });
  },

  setAllLayersVisible: (visible: boolean) => {
    const state = get();
    const newVisibility: Record<number, boolean> = {};
    state.magtClasses.forEach((cfg: MagtClassConfig) => {
      newVisibility[cfg.id] = visible;
    });
    set({ layerVisibility: newVisibility });
  },
}));
