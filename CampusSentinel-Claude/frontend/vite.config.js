import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // loadEnv with prefix '' reads ALL variables from .env (including the
  // non-VITE_ BACKEND_PROXY_TARGET). This runs in Node, not in the browser.
  const env = loadEnv(mode, process.cwd(), '');
  const backendTarget = env.BACKEND_PROXY_TARGET || 'http://127.0.0.1:5000';

  return {
    plugins: [react()],
    server: {
      port: 5173,
      strictPort: true, // fail loudly instead of silently moving to 5174
      proxy: {
        // Browser → http://localhost:5173/api/health
        // Vite    → http://127.0.0.1:5000/api/health
        // The browser only ever talks to one origin, so there is no CORS problem.
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
