// HTTP client for the Python FastAPI AI service.
// Only this file knows the AI service's URL and endpoint shapes.

import fs from 'node:fs';
import path from 'node:path';
import axios from 'axios';
import FormData from 'form-data';
import { config } from '../config/env.js';
import { AppError } from '../utils/AppError.js';
import { logger } from '../utils/logger.js';

const UPLOAD_ENDPOINT = '/videos/upload';
const NETWORK_UNAVAILABLE = new Set(['ECONNREFUSED', 'ENOTFOUND', 'EHOSTUNREACH', 'ECONNRESET', 'EAI_AGAIN', 'EPIPE']);
const TIMEOUT_CODES = new Set(['ECONNABORTED', 'ETIMEDOUT']);

// Keep only a short, harmless description from a FastAPI error body.
// FastAPI errors look like {"detail": "text"} or {"detail": [{"loc": [...], "msg": "..."}]}.
function safeDetail(data) {
  const detail = data?.detail;
  if (typeof detail === 'string') return detail.slice(0, 300);
  if (Array.isArray(detail)) {
    return detail.slice(0, 5).map((d) => ({
      field: Array.isArray(d?.loc) ? d.loc.join('.') : undefined,
      message: typeof d?.msg === 'string' ? d.msg.slice(0, 200) : undefined,
    }));
  }
  return undefined;
}

// Clean filename for the forwarded form (no folders, no control characters).
function safeFilename(name) {
  const base = path.basename(String(name || 'video')).replace(/[\x00-\x1f\x7f"\\]/g, '_');
  return base.slice(0, 200) || 'video';
}

/**
 * Stream a temp file to FastAPI as multipart form data.
 * @returns {Promise<{status: number, data: any}>} FastAPI's 2xx status and JSON body
 * @throws {AppError} 503/504/502 or FastAPI's own 4xx, with a safe message
 */
export async function forwardVideoToAiService({ filePath, originalName, mimeType, size, cameraId }) {
  const stream = fs.createReadStream(filePath);
  const form = new FormData();
  form.append('camera_id', cameraId);
  form.append('file', stream, { filename: safeFilename(originalName), contentType: mimeType, knownLength: size });

  const url = `${config.aiService.url}${UPLOAD_ENDPOINT}`;
  try {
    const res = await axios.post(url, form, {
      headers: { ...form.getHeaders(), 'Content-Length': form.getLengthSync() },
      timeout: config.aiService.timeoutMs,
      maxBodyLength: Infinity, // request body = the video (size already limited by Multer)
      maxContentLength: 10 * 1024 * 1024, // response = small JSON
      maxRedirects: 0,
    });
    return { status: res.status, data: res.data };
  } catch (err) {
    throw toAppError(err, UPLOAD_ENDPOINT, 'The AI service rejected the video');
  } finally {
    // Close the file handle BEFORE the controller deletes the file.
    // On Windows an open file cannot be deleted (EBUSY/EPERM), e.g. after ECONNREFUSED.
    await closeStream(stream);
  }
}

function closeStream(stream) {
  if (stream.closed) return Promise.resolve();
  return new Promise((resolve) => {
    stream.once('close', resolve);
    stream.destroy();
  });
}

function toAppError(err, endpoint, rejectedMessage, timeoutMs = config.aiService.timeoutMs) {
  if (err.response) {
    const { status, data } = err.response;
    logger.warn(`AI service answered ${status} for ${endpoint}`);
    if (status === 404) return new AppError(404, 'AI_RESOURCE_NOT_FOUND', 'The AI service could not find this upload or file');
    if (status === 409) return new AppError(409, 'ANALYSIS_IN_PROGRESS', 'This video is already being analysed');
    if (status === 503) {
      return new AppError(503, 'AI_MODELS_UNAVAILABLE', 'No AI model is available in the AI service', safeDetail(data));
    }
    if (status >= 400 && status < 500) {
      // FastAPI rejected the request (e.g. 422 validation, 413 too large): pass the status through.
      return new AppError(status, 'AI_SERVICE_REJECTED', rejectedMessage, safeDetail(data));
    }
    return new AppError(502, 'AI_SERVICE_ERROR', 'The AI service failed to process the request');
  }
  if (TIMEOUT_CODES.has(err.code)) {
    logger.warn(`AI service timed out after ${timeoutMs} ms on ${endpoint}`);
    return new AppError(504, 'AI_SERVICE_TIMEOUT', 'The AI service did not respond in time');
  }
  if (NETWORK_UNAVAILABLE.has(err.code)) {
    logger.warn(`AI service unreachable at ${config.aiService.url} (${err.code})`);
    return new AppError(503, 'AI_SERVICE_UNAVAILABLE', 'The AI service is not reachable. Is it running?');
  }
  logger.error(`Unexpected error while contacting the AI service (${endpoint})`, err);
  return new AppError(502, 'AI_SERVICE_ERROR', 'Could not reach the AI service');
}

// ---------------------------------------------------------------------------- analysis

const UPLOAD_ID_RE = /^[A-Za-z0-9_-]{1,64}$/;
export function isValidUploadId(id) {
  return typeof id === 'string' && UPLOAD_ID_RE.test(id);
}

/**
 * Ask FastAPI to run all available detectors on an already uploaded video.
 * @returns {Promise<object>} FastAPI's structured analysis JSON
 */
export async function analyzeUploadedVideo(uploadId, { cameraId, recordedAt, force = false }) {
  const endpoint = `/videos/${encodeURIComponent(uploadId)}/analyze`;
  try {
    const res = await axios.post(
      `${config.aiService.url}${endpoint}`,
      { camera_id: cameraId, recorded_at: recordedAt ?? null, force },
      { timeout: config.aiService.analysisTimeoutMs, maxContentLength: 20 * 1024 * 1024, maxRedirects: 0 }
    );
    return res.data;
  } catch (err) {
    throw toAppError(err, endpoint, 'The AI service rejected the analysis request', config.aiService.analysisTimeoutMs);
  }
}

/** Which models FastAPI has loaded / unavailable / failed. */
export async function getModelStatus() {
  try {
    const res = await axios.get(`${config.aiService.url}/models/status`, { timeout: 120000, maxRedirects: 0 });
    return res.data;
  } catch (err) {
    throw toAppError(err, '/models/status', 'The AI service rejected the request');
  }
}

/** Stream one evidence image from FastAPI (the caller pipes it to the browser). */
export async function getEvidenceStream(uploadId, fileName) {
  const endpoint = `/evidence/${encodeURIComponent(uploadId)}/${encodeURIComponent(fileName)}`;
  try {
    const res = await axios.get(`${config.aiService.url}${endpoint}`, {
      responseType: 'stream', timeout: 30000, maxRedirects: 0,
    });
    return res;
  } catch (err) {
    throw toAppError(err, endpoint, 'Evidence image not available');
  }
}
