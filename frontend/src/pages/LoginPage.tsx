// Система авторизации отключена
import { Navigate } from 'react-router-dom';

export function LoginPage() {
  // Перенаправляем на главную страницу
  return <Navigate to="/" replace />;
}