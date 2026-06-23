import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend runs on :8000 by default (uv run uvicorn webapp.backend.main:app
// --port 8000). Override with BACKEND_URL if you use a different port.
const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: backend, changeOrigin: true },
    },
  },
});
