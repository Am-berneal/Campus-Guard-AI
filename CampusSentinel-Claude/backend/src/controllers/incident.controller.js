// Incident review API: list, stats, details, review/dismiss, evidence images.

import { CAMERA_IDS, INCIDENT_STATUSES, INCIDENT_TYPES } from '../models/Incident.js';
import { getEvidenceStream } from '../services/aiService.client.js';
import { getIncident, incidentStats, listIncidents, reviewIncident } from '../services/incident.service.js';
import { AppError } from '../utils/AppError.js';

function optionalEnum(value, allowed, name) {
  if (value === undefined || value === '') return undefined;
  if (!allowed.includes(value)) throw new AppError(400, 'VALIDATION_ERROR', `${name} must be one of: ${allowed.join(', ')}`);
  return value;
}

function positiveInt(value, fallback, max) {
  if (value === undefined) return fallback;
  const n = Number(value);
  if (!Number.isInteger(n) || n < 1 || n > max) throw new AppError(400, 'VALIDATION_ERROR', `must be an integer 1-${max}`);
  return n;
}

export async function getIncidents(req, res) {
  const q = req.query;
  const result = await listIncidents({
    status: optionalEnum(q.status, INCIDENT_STATUSES, 'status'),
    type: optionalEnum(q.type, INCIDENT_TYPES, 'type'),
    cameraId: optionalEnum(q.cameraId, CAMERA_IDS, 'cameraId'),
    videoId: typeof q.videoId === 'string' && /^[A-Za-z0-9_-]{1,64}$/.test(q.videoId) ? q.videoId : undefined,
    page: positiveInt(q.page, 1, 10000),
    limit: positiveInt(q.limit, 50, 200),
  });
  res.json(result);
}

export async function getIncidentStats(req, res) {
  res.json(await incidentStats());
}

export async function getIncidentById(req, res) {
  res.json(await getIncident(req.params.id));
}

export async function patchIncidentReview(req, res) {
  const { status, note } = req.body ?? {};
  if (!INCIDENT_STATUSES.includes(status)) {
    throw new AppError(400, 'VALIDATION_ERROR', `status must be one of: ${INCIDENT_STATUSES.join(', ')}`);
  }
  if (note !== undefined && (typeof note !== 'string' || note.length > 1000)) {
    throw new AppError(400, 'VALIDATION_ERROR', 'note must be text of at most 1000 characters');
  }
  res.json(await reviewIncident(req.params.id, { status, note: note?.trim() }));
}

// GET /api/incidents/:id/evidence/:index → image streamed from the AI service.
// The browser never talks to FastAPI directly.
export async function getIncidentEvidence(req, res) {
  const incident = await getIncident(req.params.id);
  const index = Number(req.params.index);
  const item = Number.isInteger(index) ? incident.evidence?.[index] : undefined;
  if (!item?.imageFile) throw new AppError(404, 'NOT_FOUND', 'No evidence image for this incident');
  const upstream = await getEvidenceStream(incident.videoId, item.imageFile);
  res.set('Content-Type', 'image/jpeg');
  res.set('Cache-Control', 'private, max-age=3600');
  upstream.data.on('error', () => res.destroy()); // AI service dropped mid-stream: end the response
  upstream.data.pipe(res);
}
