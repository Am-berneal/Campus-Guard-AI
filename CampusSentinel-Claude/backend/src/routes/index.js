// Central route registry. Each feature gets its own router file and is mounted here,
// so app.js stays small as features are added (auth, cameras, zones, incidents...).

import { Router } from 'express';
import healthRoutes from './health.routes.js';
import videoRoutes from './video.routes.js';
import incidentRoutes from './incident.routes.js';
import aiRoutes from './ai.routes.js';

const router = Router();

router.use('/health', healthRoutes);
router.use('/videos', videoRoutes);
router.use('/incidents', incidentRoutes);
router.use('/ai', aiRoutes);

export default router;
