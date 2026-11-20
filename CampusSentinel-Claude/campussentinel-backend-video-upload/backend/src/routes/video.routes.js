import { Router } from 'express';
import { receiveVideo } from '../middleware/videoUpload.js';
import { uploadVideo } from '../controllers/video.controller.js';

const router = Router();

// POST /api/videos/upload  (multipart: camera_id, file)
router.post('/upload', receiveVideo, uploadVideo);

export default router;
