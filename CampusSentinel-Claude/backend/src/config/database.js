// Owns the single Mongoose connection for the whole backend.
// Mongoose keeps a connection pool internally; we connect once at startup
// and every model created later reuses it automatically.

import mongoose from 'mongoose';
import { config } from './env.js';
import { logger } from '../utils/logger.js';

// Mongoose's numeric readyState → readable name.
const READY_STATES = {
  0: 'disconnected',
  1: 'connected',
  2: 'connecting',
  3: 'disconnecting',
};

let listenersAttached = false;

function attachConnectionListeners() {
  if (listenersAttached) return;
  listenersAttached = true;

  const conn = mongoose.connection;
  conn.on('connected', () => logger.info(`MongoDB connected (${conn.host}:${conn.port}/${conn.name})`));
  conn.on('disconnected', () => logger.warn('MongoDB disconnected'));
  conn.on('reconnected', () => logger.info('MongoDB reconnected'));
  conn.on('error', (err) => logger.error('MongoDB connection error', err));
}

export async function connectDatabase() {
  attachConnectionListeners();

  // Throw on queries to undefined schema fields instead of silently ignoring them.
  mongoose.set('strictQuery', true);

  // If MongoDB is not reachable within serverSelectionTimeoutMS, this rejects.
  // The caller (server.js) decides what to do: we stop startup with a clear error.
  await mongoose.connect(config.mongodb.uri, {
    serverSelectionTimeoutMS: config.mongodb.serverSelectionTimeoutMS,
  });
}

export async function disconnectDatabase() {
  if (mongoose.connection.readyState !== 0) {
    await mongoose.disconnect();
    logger.info('MongoDB connection closed');
  }
}

export function getDatabaseStatus() {
  const conn = mongoose.connection;
  const state = READY_STATES[conn.readyState] ?? 'unknown';
  return {
    state,
    isConnected: conn.readyState === 1,
    name: conn.name || null,
    host: conn.host || null,
  };
}

// Sends a lightweight "ping" command to MongoDB and measures the round trip.
// readyState alone can lag behind reality; a ping proves the server answers now.
export async function pingDatabase(timeoutMs = 2000) {
  if (mongoose.connection.readyState !== 1) {
    return { ok: false, latencyMs: null };
  }

  const started = process.hrtime.bigint();
  let timer;
  try {
    await Promise.race([
      mongoose.connection.db.admin().ping(),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('ping timeout')), timeoutMs);
      }),
    ]);
    const latencyMs = Number(process.hrtime.bigint() - started) / 1e6;
    return { ok: true, latencyMs: Math.round(latencyMs * 10) / 10 };
  } catch {
    return { ok: false, latencyMs: null };
  } finally {
    clearTimeout(timer);
  }
}
