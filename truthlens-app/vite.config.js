import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react()],
    define: {
      // Makes VITE_API_URL available anywhere via import.meta.env.VITE_API_URL
    },
    server: {
      // Local dev proxy: forwards /api calls to Flask on port 5000
      proxy: {
        '/predict': {
          target: env.VITE_API_URL || 'http://localhost:5000',
          changeOrigin: true,
        },
        '/predict-image': {
          target: env.VITE_API_URL || 'http://localhost:5000',
          changeOrigin: true,
        },
      },
    },
  }
})
