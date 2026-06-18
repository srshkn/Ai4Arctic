// Система авторизации отключена — все маршруты открыты
import { Outlet } from 'react-router-dom';

export function ProtectedRoute() {
  // Всегда пропускаем без проверки авторизации
  return <Outlet />;
}
