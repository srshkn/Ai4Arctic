import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MapPage } from './pages/MapPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Все маршруты открыты (система авторизации отключена) */}
        <Route path="/" element={<MapPage />} />
        
        {/* Fallback — перенаправляем на главную */}
        <Route path="*" element={<MapPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;