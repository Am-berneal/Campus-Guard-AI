import { Router } from 'express';
import { receiveVideo } from '../middleware/videoUpload.js';
import { analyzeVideo, uploadVideo } from '../controllers/video.controller.js';

const router = Router();

// POST /api/videos/upload  (multipart: camera_id, file)
router.post('/upload', receiveVideo, uploadVideo);

// POST /api/videos/:uploadId/analyze  { camera_id, recorded_at?, force?, original_name? }
router.post('/:uploadId/analyze', analyzeVideo);

export default router;
