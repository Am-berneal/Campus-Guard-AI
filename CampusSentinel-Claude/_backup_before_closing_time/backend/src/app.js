// Builds and configures the Express application, but does NOT start listening.
// Keeping "build the app" separate from "start the server" makes the app
// easy to test later without opening a real network port.

import express from 'express';
import apiRoutes from './routes/index.js';
import { notFound } from './middleware/notFound.js';
import { errorHandler } from './middleware/errorHandler.js';

export function createApp() {
  const app = express();

  // Don't advertise the framework in response headers.
  app.disable('x-powered-by');

  // Parse JSON request bodies (needed from Phase 2 onwards). Small limit:
  // evidence files will arrive via multipart uploads, not JSON.
  app.use(express.json({ limit: '1mb' }));

  // All API routes live under /api.
  app.use('/api', apiRoutes);

  // Order matters: these must come AFTER all routes.
  app.use(notFound);
  app.use(errorHandler);

  return app;
}
