import { api } from './client';
import type { LoginRequest, RegisterRequest, TokenPair, UserOut } from '../types';

const ENDPOINTS = {
  login: '/auth/login',
  register: '/auth/register',
  me: '/auth/me',
};

export async function login(data: LoginRequest): Promise<TokenPair> {
  const response = await api.post<TokenPair>(ENDPOINTS.login, data);
  return response.data;
}

export async function register(data: RegisterRequest): Promise<TokenPair> {
  const response = await api.post<TokenPair>(ENDPOINTS.register, data);
  return response.data;
}

export async function getCurrentUser(): Promise<UserOut> {
  const response = await api.get<UserOut>(ENDPOINTS.me);
  return response.data;
}

export function setAuthTokens(access: string, refresh: string): void {
  localStorage.setItem('access_token', access);
  localStorage.setItem('refresh_token', refresh);
}

export function clearAuthTokens(): void {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
}