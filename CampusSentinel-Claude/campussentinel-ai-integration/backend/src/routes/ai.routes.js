import { Router } from 'express';
import { getModelStatus } from '../services/aiService.client.js';

const router = Router();

// GET /api/ai/models/status → which AI models FastAPI has loaded (proxied, read-only)
router.get('/models/status', async (req, res) => {
  res.json(await getModelStatus());
});

export default router;
