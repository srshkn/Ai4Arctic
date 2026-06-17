// Система авторизации отключена
import { Navigate } from 'react-router-dom';

export function RegisterPage() {
  // Перенаправляем на главную страницу
  return <Navigate to="/" replace />;
}