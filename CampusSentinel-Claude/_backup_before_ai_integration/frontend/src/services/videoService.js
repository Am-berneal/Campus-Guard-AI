// Video upload API calls. Uses the shared apiClient, so the request goes to
// `${VITE_API_URL}/videos/upload` → Vite proxy → Express. Never directly to FastAPI.

import { apiClient, toApiError } from './apiClient.js';
import { MAX_VIDEO_BYTES, VIDEO_EXTENSIONS } from '../constants/cameras.js';

// Large videos take longer than apiClient's default 8 s timeout (meant for quick JSON calls).
// Allow up to 30 minutes for uploading + Express forwarding to FastAPI.
const UPLOAD_TIMEOUT_MS = 30 * 60 * 1000;

/** Returns an error message for an invalid file, or null if the file looks acceptable. */
export function validateVideoFile(file) {
  if (!file) return 'Choose a video file.';
  const dot = file.name.lastIndexOf('.');
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : '';
  if (!VIDEO_EXTENSIONS.includes(ext)) {
    return `Unsupported file type "${ext || 'none'}". Use ${VIDEO_EXTENSIONS.join(', ')}.`;
  }
  if (file.size === 0) return 'The selected file is empty.';
  if (file.size > MAX_VIDEO_BYTES) return 'The video is larger than the 500 MB limit.';
  return null;
}

/**
 * Upload one video for a simulated camera.
 * @param {{cameraId: string, file: File, onProgress?: (percent: number) => void, signal?: AbortSignal}} args
 * @returns {Promise<object>} Express response: { cameraId, originalName, sizeBytes, aiService }
 * @throws {object} normalised error from toApiError()
 */
export async function uploadVideo({ cameraId, file, onProgress, signal }) {
  const form = new FormData();
  // Field names must match the backend exactly. camera_id goes FIRST so the backend
  // can reject an invalid camera before receiving the whole video.
  form.append('camera_id', cameraId);
  form.append('file', file);

  try {
    // Do NOT set Content-Type manually: the browser adds the multipart boundary itself.
    const { data } = await apiClient.post('/videos/upload', form, {
      timeout: UPLOAD_TIMEOUT_MS,
      signal,
      onUploadProgress: (event) => {
        if (onProgress && event.total) onProgress(Math.round((event.loaded / event.total) * 100));
      },
    });
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}
