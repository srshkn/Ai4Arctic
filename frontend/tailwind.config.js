/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        permafrost: {
          continuous: '#1a3a5c',
          discontinuous: '#2d6da8',
          isolated: '#7ab3e0',
          relic: '#b8d9f0',
        },
        arctic: {
          dark: '#0a1628',
          mid: '#1a2a44',
          light: '#2a3a54',
          accent: '#4fc3f7',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}