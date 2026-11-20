// POST /api/videos/upload
// Multer (middleware/videoUpload.js) has already saved the video to a temp file.
// Here we validate the form, forward the file to FastAPI, and ALWAYS delete the temp file.

import fsp from 'node:fs/promises';
import { forwardVideoToAiService } from '../services/aiService.client.js';
import { invalidCameraError, isValidCameraId } from '../middleware/videoUpload.js';
import { AppError } from '../utils/AppError.js';
import { logger } from '../utils/logger.js';

export async function uploadVideo(req, res) {
  const file = req.file;
  let cameraId;
  let result;
  try {
    if (!file) {
      throw new AppError(400, 'FILE_REQUIRED', 'Attach a video in the "file" field');
    }
    if (file.size === 0) {
      throw new AppError(400, 'EMPTY_FILE', 'The uploaded video is empty');
    }
    cameraId = req.body?.camera_id;
    if (!isValidCameraId(cameraId)) {
      throw invalidCameraError();
    }

    result = await forwardVideoToAiService({
      filePath: file.path,
      originalName: file.originalname,
      mimeType: file.mimetype,
      size: file.size,
      cameraId,
    });
  } finally {
    // Runs on success AND on every error above (Express 5 then passes the error on
    // to the error handler). The temp file is gone before the client gets a reply.
    if (file?.path) {
      await fsp.rm(file.path, { force: true }).catch((err) =>
        logger.warn(`Could not delete temp upload ${file.filename}: ${err.message}`)
      );
    }
  }

  logger.info(`Video forwarded: camera=${cameraId} size=${file.size} B -> AI service ${result.status}`);
  res.status(result.status).json({
    cameraId,
    originalName: file.originalname,
    sizeBytes: file.size,
    aiService: result.data, // FastAPI's response, unchanged
  });
}
