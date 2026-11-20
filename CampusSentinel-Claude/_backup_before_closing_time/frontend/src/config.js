// Single place where the frontend reads its build-time configuration.
// Vite replaces import.meta.env.VITE_* with the values from .env when it serves/builds.

function positiveInt(value, fallback) {
  const n = Number(value);
  return Number.isInteger(n) && n > 0 ? n : fallback;
}

export const config = Object.freeze({
  apiUrl: import.meta.env.VITE_API_URL || '/api',
  healthPollIntervalMs: positiveInt(import.meta.env.VITE_HEALTH_POLL_INTERVAL_MS, 15000),
});
