import { Router } from 'express';
import {
  getIncidentById,
  getIncidentEvidence,
  getIncidents,
  getIncidentStats,
  patchIncidentReview,
} from '../controllers/incident.controller.js';

const router = Router();

// GET   /api/incidents?status=&type=&cameraId=&videoId=&page=&limit=
router.get('/', getIncidents);
// GET   /api/incidents/stats
router.get('/stats', getIncidentStats);
// GET   /api/incidents/:id
router.get('/:id', getIncidentById);
// PATCH /api/incidents/:id/review   { status: reviewed | dismissed | pending_review, note? }
router.patch('/:id/review', patchIncidentReview);
// GET   /api/incidents/:id/evidence/:index   (image proxied from the AI service)
router.get('/:id/evidence/:index', getIncidentEvidence);

export default router;
