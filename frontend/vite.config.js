import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/explain': 'http://127.0.0.1:8000',
      '/models': 'http://127.0.0.1:8000',
      '/severity': 'http://127.0.0.1:8000',
      '/batch': 'http://127.0.0.1:8000',
      '/similarity': 'http://127.0.0.1:8000',
      '/rag': 'http://127.0.0.1:8000',
      '/dashboard': 'http://127.0.0.1:8000',
      '/drift': 'http://127.0.0.1:8000',
      '/rca': 'http://127.0.0.1:8000',
      '/samples': 'http://127.0.0.1:8000',
      '/categories': 'http://127.0.0.1:8000',
      '/reports': 'http://127.0.0.1:8000',
      '/active-learning': 'http://127.0.0.1:8000',
      '/outputs': 'http://127.0.0.1:8000',
      '/stream': 'http://127.0.0.1:8000',
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
