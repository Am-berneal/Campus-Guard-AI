// POST /api/videos/upload
// Multer (middleware/videoUpload.js) has already saved the video to a temp file.
// Here we validate the form, forward the file to FastAPI, and ALWAYS delete the temp file.

import fsp from 'node:fs/promises';
import { analyzeUploadedVideo, forwardVideoToAiService, isValidUploadId } from '../services/aiService.client.js';
import { saveIncidentsFromAnalysis } from '../services/incident.service.js';
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

// POST /api/videos/:uploadId/analyze
// Asks FastAPI to analyse an uploaded video, stores the resulting POTENTIAL incidents in
// MongoDB, and returns both. All inference happens in Python; Node only validates and stores.
const CAMERA_NAMES = { library: 'Outside Library', b_block: 'B Block', cos: 'COS', kravings: 'Kravings' };

export async function analyzeVideo(req, res) {
  const { uploadId } = req.params;
  const { camera_id: cameraId, recorded_at: recordedAt, force, original_name: originalName } = req.body ?? {};
  if (!isValidUploadId(uploadId)) throw new AppError(400, 'INVALID_UPLOAD_ID', 'Invalid upload ID');
  if (!isValidCameraId(cameraId)) throw invalidCameraError();
  if (recordedAt !== undefined && recordedAt !== null && Number.isNaN(Date.parse(recordedAt))) {
    throw new AppError(400, 'VALIDATION_ERROR', 'recorded_at must be an ISO date-time');
  }
  if (force !== undefined && typeof force !== 'boolean') {
    throw new AppError(400, 'VALIDATION_ERROR', 'force must be true or false');
  }

  const started = Date.now();
  const analysis = await analyzeUploadedVideo(uploadId, { cameraId, recordedAt, force: Boolean(force) });
  if (analysis?.upload_id !== uploadId) {
    throw new AppError(502, 'AI_RESULT_INVALID', 'The AI service returned a result for a different upload');
  }

  const incidents = await saveIncidentsFromAnalysis(analysis, {
    cameraId,
    cameraName: CAMERA_NAMES[cameraId],
    originalFilename: typeof originalName === 'string' ? originalName.slice(0, 255) : undefined,
  });
  logger.info(`Video ${uploadId} analysed (${analysis.status}) in ${Date.now() - started} ms; ` +
    `${incidents.length} potential incident(s) stored`);
  res.json({ analysis, incidents });
}
