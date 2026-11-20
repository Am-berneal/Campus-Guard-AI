// Business logic for the health check: gathers facts, knows nothing about HTTP.
// Keeping this out of the controller means the AI service integration or a
// startup self-check can reuse it later.

import { config } from '../config/env.js';
import { getDatabaseStatus, pingDatabase } from '../config/database.js';

export async function getHealthReport() {
  const db = getDatabaseStatus();
  const ping = await pingDatabase();
  const databaseHealthy = db.isConnected && ping.ok;

  return {
    healthy: databaseHealthy,
    report: {
      status: databaseHealthy ? 'ok' : 'degraded',
      service: 'campussentinel-backend',
      environment: config.nodeEnv,
      timestamp: new Date().toISOString(),
      uptimeSeconds: Math.round(process.uptime()),
      database: {
        status: databaseHealthy ? 'connected' : db.state === 'connected' ? 'unresponsive' : db.state,
        name: db.name,
        host: db.host,
        pingMs: ping.latencyMs,
      },
    },
  };
}
