import { create } from 'zustand';
import type { UserOut } from '../types';
import { login as apiLogin, register as apiRegister, getCurrentUser, setAuthTokens, clearAuthTokens } from '../api/auth';

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  user: UserOut | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  login: (name: string, password: string) => Promise<void>;
  register: (name: string, password: string, confirmPassword: string) => Promise<void>;
  logout: () => void;
  fetchUser: () => Promise<void>;
  checkAuth: () => Promise<boolean>;
  clearError: () => void;
}

// Axios error type guard
function isAxiosError(error: unknown): error is { response?: { data?: { detail?: string } }; message?: string } {
  return typeof error === 'object' && error !== null && 'response' in error;
}

export const useAuthStore = create<AuthState>((set) => ({
  accessToken: localStorage.getItem('access_token'),
  refreshToken: localStorage.getItem('refresh_token'),
  user: null,
  isAuthenticated: !!localStorage.getItem('access_token'),
  isLoading: false,
  error: null,

  login: async (name: string, password: string) => {
    set({ isLoading: true, error: null });
    try {
      const tokens = await apiLogin({ name, password });
      setAuthTokens(tokens.access_token, tokens.refresh_token);
      set({
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        isAuthenticated: true,
        isLoading: false,
      });
      const user = await getCurrentUser();
      set({ user });
    } catch (error: unknown) {
      const message = isAxiosError(error)
        ? error.response?.data?.detail ?? 'Ошибка авторизации'
        : 'Ошибка авторизации';
      set({ error: message as string, isLoading: false });
      throw error;
    }
  },

  register: async (name: string, password: string, confirmPassword: string) => {
    set({ isLoading: true, error: null });
    try {
      const tokens = await apiRegister({ name, password, confirm_password: confirmPassword });
      setAuthTokens(tokens.access_token, tokens.refresh_token);
      set({
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token,
        isAuthenticated: true,
        isLoading: false,
      });
      const user = await getCurrentUser();
      set({ user });
    } catch (error: unknown) {
      const message = isAxiosError(error)
        ? error.response?.data?.detail ?? 'Ошибка регистрации'
        : 'Ошибка регистрации';
      set({ error: message as string, isLoading: false });
      throw error;
    }
  },

  logout: () => {
    clearAuthTokens();
    set({
      accessToken: null,
      refreshToken: null,
      user: null,
      isAuthenticated: false,
      error: null,
    });
  },

  fetchUser: async () => {
    try {
      const user = await getCurrentUser();
      set({ user });
    } catch {
      // User might not be authenticated anymore
    }
  },

  checkAuth: async () => {
    const token = useAuthStore.getState().accessToken;
    if (!token) {
      set({ isAuthenticated: false, user: null });
      return false;
    }
    try {
      const user = await getCurrentUser();
      set({ user, isAuthenticated: true });
      return true;
    } catch {
      clearAuthTokens();
      set({ isAuthenticated: false, accessToken: null, refreshToken: null, user: null });
      return false;
    }
  },

  clearError: () => set({ error: null }),
}));
