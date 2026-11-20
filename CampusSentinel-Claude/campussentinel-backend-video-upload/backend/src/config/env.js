// Loads and validates environment variables once, then exports a frozen config object.
// Every other file imports `config` from here instead of reading process.env directly,
// so all configuration rules live in one place.

import path from 'node:path';
import { fileURLToPath } from 'node:url';
import dotenv from 'dotenv';

// Resolve backend/.env relative to THIS file, so the server finds it
// no matter which folder you start it from.
const envPath = fileURLToPath(new URL('../../.env', import.meta.url));
dotenv.config({ path: envPath, quiet: true });

function required(name) {
  const value = process.env[name];
  if (!value || value.trim() === '') {
    throw new Error(
      `Missing required environment variable "${name}". ` +
        'Copy backend/.env.example to backend/.env and set it.'
    );
  }
  return value.trim();
}

function integer(name, fallback, { min, max }) {
  const raw = process.env[name];
  if (raw === undefined || raw.trim() === '') return fallback;

  const value = Number(raw);
  if (!Number.isInteger(value) || value < min || value > max) {
    throw new Error(
      `Environment variable "${name}" must be an integer between ${min} and ${max} (got "${raw}").`
    );
  }
  return value;
}

// Optional URL setting: must be http(s). Trailing slashes are removed so paths can be appended safely.
function httpUrl(name, fallback) {
  const raw = (process.env[name] || fallback).trim();
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`Environment variable "${name}" must be a valid URL (got "${raw}").`);
  }
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error(`Environment variable "${name}" must start with http:// or https:// (got "${raw}").`);
  }
  return raw.replace(/\/+$/, '');
}

// Temporary upload folder: relative paths are resolved from the backend folder, not the shell's cwd.
const backendRoot = fileURLToPath(new URL('../../', import.meta.url));
const uploadTmpDir = path.resolve(backendRoot, (process.env.UPLOAD_TMP_DIR || 'storage/tmp-uploads').trim());

const nodeEnv = (process.env.NODE_ENV || 'development').trim();
if (!['development', 'production'].includes(nodeEnv)) {
  throw new Error(`NODE_ENV must be "development" or "production" (got "${nodeEnv}").`);
}

export const config = Object.freeze({
  nodeEnv,
  isProduction: nodeEnv === 'production',
  port: integer('PORT', 5000, { min: 1, max: 65535 }),
  mongodb: Object.freeze({
    uri: required('MONGODB_URI'),
    serverSelectionTimeoutMS: integer('MONGODB_SERVER_SELECTION_TIMEOUT_MS', 5000, {
      min: 500,
      max: 60000,
    }),
  }),
  aiService: Object.freeze({
    url: httpUrl('AI_SERVICE_URL', 'http://127.0.0.1:8000'),
    // How long to wait for FastAPI to accept the video and answer (default 10 minutes).
    timeoutMs: integer('AI_SERVICE_TIMEOUT_MS', 600000, { min: 1000, max: 3600000 }),
  }),
  videoUpload: Object.freeze({
    tmpDir: uploadTmpDir,
    maxBytes: integer('VIDEO_UPLOAD_MAX_MB', 500, { min: 1, max: 4096 }) * 1024 * 1024,
  }),
});
