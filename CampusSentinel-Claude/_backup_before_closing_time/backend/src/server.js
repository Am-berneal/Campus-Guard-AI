// Entry point: connect to MongoDB, start the HTTP server, and shut both down cleanly.
//
// Startup order:  load config → connect MongoDB → start listening.
// We only accept requests once the database is ready. If MongoDB is down at
// startup, we exit with a clear message instead of running half-broken.
// If MongoDB drops LATER, Mongoose reconnects automatically and
// GET /api/health reports "degraded" (HTTP 503) in the meantime.

import { logger } from './utils/logger.js';

let config;
try {
  ({ config } = await import('./config/env.js'));
} catch (err) {
  logger.error(`Configuration error: ${err.message}`);
  process.exit(1);
}

const { createApp } = await import('./app.js');
const { connectDatabase, disconnectDatabase } = await import('./config/database.js');

const SHUTDOWN_TIMEOUT_MS = 10_000;
let server;
let shuttingDown = false;

async function shutdown(reason, exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  logger.info(`Shutting down (${reason})...`);

  // Safety net: if something hangs (e.g. a stuck connection), force exit.
  const forceTimer = setTimeout(() => {
    logger.error(`Shutdown took longer than ${SHUTDOWN_TIMEOUT_MS} ms; forcing exit`);
    process.exit(1);
  }, SHUTDOWN_TIMEOUT_MS);
  forceTimer.unref();

  try {
    if (server) {
      // Stop accepting new connections and let in-flight requests finish.
      // (Since Node 19, close() also drops idle keep-alive connections.)
      await new Promise((resolve, reject) =>
        server.close((err) => (err ? reject(err) : resolve()))
      );
      logger.info('HTTP server closed');
    }
    await disconnectDatabase();
  } catch (err) {
    logger.error('Error during shutdown', err);
    exitCode = 1;
  }

  process.exit(exitCode);
}

async function start() {
  logger.info(`Starting CampusSentinel backend (NODE_ENV=${config.nodeEnv})`);

  try {
    await connectDatabase();
  } catch (err) {
    logger.error(
      `Could not connect to MongoDB within ${config.mongodb.serverSelectionTimeoutMS} ms. ` +
        'Is the MongoDB service running? Check MONGODB_URI in backend/.env.\n' +
        `  Reason: ${err.message}`
    );
    process.exit(1);
  }

  const app = createApp();
  server = app.listen(config.port);

  server.on('listening', () => {
    logger.info(`Backend listening on http://localhost:${config.port}`);
    logger.info(`Health check:      http://localhost:${config.port}/api/health`);
  });

  server.on('error', async (err) => {
    if (err.code === 'EADDRINUSE') {
      logger.error(`Port ${config.port} is already in use. Stop the other process or change PORT in backend/.env.`);
    } else {
      logger.error('HTTP server error', err);
    }
    await shutdown('server error', 1);
  });
}

// Ctrl+C in PowerShell sends SIGINT. SIGTERM is what process managers send.
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

// Last-resort handlers: log the bug, then exit cleanly rather than keep
// running in an unknown state.
process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled promise rejection', reason);
  shutdown('unhandledRejection', 1);
});
process.on('uncaughtException', (err) => {
  logger.error('Uncaught exception', err);
  shutdown('uncaughtException', 1);
});

start();
