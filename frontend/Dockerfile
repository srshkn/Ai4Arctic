# ====== Stage 1: Build ======
FROM node:20-alpine AS build

WORKDIR /app

# Копируем package.json и yarn.lock для кэширования зависимостей
COPY package.json ./
COPY tsconfig.json ./

# Устанавливаем зависимости
RUN npm install

# Копируем исходный код
COPY . .

# Собираем приложение
RUN npm run build

# ====== Stage 2: Serve with Nginx ======
FROM nginx:alpine AS production

# Копилируем собранные файлы из stage build
COPY --from=build /app/dist /usr/share/nginx/html

# Копируем nginx конфигурацию
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Создаём директорию для данных (GeoJSON)
RUN mkdir -p /usr/share/nginx/data && chown -R nginx:nginx /usr/share/nginx

# Открываем порт 80
EXPOSE 80

# Команда запуска
CMD ["nginx", "-g", "daemon off;"]