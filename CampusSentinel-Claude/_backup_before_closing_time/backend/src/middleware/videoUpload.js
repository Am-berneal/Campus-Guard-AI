// Receives ONE video per request with Multer and stores it in a temporary folder on disk
// (never in memory: videos can be up to 500 MB). The controller forwards the file to the
// Python AI service and deletes it afterwards.
//
// Expected multipart fields:   camera_id (text)   file (the video)
// Send camera_id BEFORE file in the form so an invalid camera is rejected before the
// whole video is uploaded. If it arrives later, the controller still validates it.

import crypto from 'node:crypto';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import multer from 'multer';
import { config } from '../config/env.js';
import { AppError } from '../utils/AppError.js';
import { logger } from '../utils/logger.js';

export const CAMERA_IDS = Object.freeze(['library', 'b_block', 'cos', 'kravings']);
export const VIDEO_EXTENSIONS = Object.freeze(['.mp4', '.avi', '.mov', '.mkv']);

const { tmpDir, maxBytes } = config.videoUpload;
const STALE_AFTER_MS = 24 * 60 * 60 * 1000;

// Create the temp folder once, and remove leftovers older than 24 h
// (e.g. files from a backend crash in the middle of a request).
fs.mkdirSync(tmpDir, { recursive: true });
sweepStaleFiles().catch((err) => logger.warn(`Temp upload sweep failed: ${err.message}`));

async function sweepStaleFiles() {
  const now = Date.now();
  for (const name of await fsp.readdir(tmpDir)) {
    const full = path.join(tmpDir, name);
    const stat = await fsp.stat(full).catch(() => null);
    if (stat?.isFile() && now - stat.mtimeMs > STALE_AFTER_MS) {
      await fsp.rm(full, { force: true });
    }
  }
}

export function isValidCameraId(value) {
  return typeof value === 'string' && CAMERA_IDS.includes(value);
}

const storage = multer.diskStorage({
  destination: tmpDir,
  // Never use the client's filename on disk: a random name avoids path tricks and collisions.
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    cb(null, `${Date.now()}-${crypto.randomUUID()}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: maxBytes, // Multer stops the upload as soon as this is exceeded
    files: 1,
    fields: 5,
    fieldSize: 1024, // text fields are tiny (camera_id)
  },
  fileFilter: (req, file, cb) => {
    if (file.fieldname !== 'file') {
      return cb(new AppError(400, 'UNEXPECTED_FIELD', 'The video must be sent in a form field named "file"'));
    }
    const ext = path.extname(file.originalname).toLowerCase();
    if (!VIDEO_EXTENSIONS.includes(ext)) {
      return cb(new AppError(400, 'UNSUPPORTED_FILE_TYPE',
        `Unsupported video type "${ext || '(none)'}". Allowed: ${VIDEO_EXTENSIONS.join(', ')}`));
    }
    // Early camera check (only possible if camera_id was sent before the file).
    if (req.body?.camera_id !== undefined && !isValidCameraId(req.body.camera_id)) {
      return cb(invalidCameraError());
    }
    return cb(null, true);
  },
}).single('file');

export function invalidCameraError() {
  return new AppError(400, 'INVALID_CAMERA_ID', `camera_id must be one of: ${CAMERA_IDS.join(', ')}`);
}

// Translate Multer's own errors into our JSON error format, and make sure no partial
// temp file is left behind when the upload fails.
const MULTER_ERRORS = {
  LIMIT_FILE_SIZE: [413, 'FILE_TOO_LARGE', `Video exceeds the ${Math.round(maxBytes / 1024 / 1024)} MB limit`],
  LIMIT_FILE_COUNT: [400, 'TOO_MANY_FILES', 'Upload exactly one video'],
  LIMIT_UNEXPECTED_FILE: [400, 'UNEXPECTED_FIELD', 'The video must be sent in a form field named "file"'],
  LIMIT_FIELD_COUNT: [400, 'TOO_MANY_FIELDS', 'Too many form fields'],
  LIMIT_FIELD_VALUE: [400, 'FIELD_TOO_LONG', 'A form field value is too long'],
  LIMIT_PART_COUNT: [400, 'TOO_MANY_PARTS', 'Too many form parts'],
};

export function receiveVideo(req, res, next) {
  if (!req.is('multipart/form-data')) {
    return next(new AppError(415, 'UNSUPPORTED_MEDIA_TYPE', 'Send the upload as multipart/form-data'));
  }
  upload(req, res, async (err) => {
    if (!err) return next();
    if (req.file?.path) await fsp.rm(req.file.path, { force: true }).catch(() => {});
    if (err instanceof multer.MulterError) {
      const [status, code, message] = MULTER_ERRORS[err.code] || [400, 'INVALID_UPLOAD', 'Invalid upload'];
      return next(new AppError(status, code, message));
    }
    if (err instanceof AppError) return next(err);
    // Malformed multipart body, client aborted the upload, disk full, ...
    logger.warn(`Video upload failed while receiving: ${err.message}`);
    return next(new AppError(400, 'UPLOAD_FAILED', 'The upload could not be received'));
  });
}
