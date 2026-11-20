// Controller: translates between HTTP and the service layer.
// It decides the status code; the service decides what "healthy" means.

import { getHealthReport } from '../services/health.service.js';

export async function getHealth(req, res) {
  const { healthy, report } = await getHealthReport();

  // 503 tells monitoring tools (and later the AI service) that the backend
  // is running but cannot do useful work without its database.
  res.status(healthy ? 200 : 503).json(report);
}
