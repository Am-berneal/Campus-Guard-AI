// Incident review API (backend /api/incidents) and AI model status (backend /api/ai).

import { apiClient, toApiError } from './apiClient.js';
import { config } from '../config.js';

async function call(promise) {
  try {
    const { data } = await promise;
    return data;
  } catch (err) {
    throw toApiError(err);
  }
}

export const listIncidents = (params = {}) => call(apiClient.get('/incidents', { params }));
export const getIncidentStats = () => call(apiClient.get('/incidents/stats'));
export const reviewIncident = (id, status, note) =>
  call(apiClient.patch(`/incidents/${encodeURIComponent(id)}/review`, { status, note: note || undefined }));
// Loading every model the first time can take a while: allow 2 minutes.
export const getModelStatus = () => call(apiClient.get('/ai/models/status', { timeout: 120000 }));

/** <img> URL for an incident's evidence image (proxied by the backend from the AI service). */
export function evidenceUrl(incidentId, index) {
  return `${config.apiUrl}/incidents/${encodeURIComponent(incidentId)}/evidence/${index}`;
}

export const INCIDENT_TYPE_LABELS = {
  fighting: 'Possible fight',
  smoking: 'Possible smoking',
  restricted_zone: 'Potential restricted-zone entry',
};

export const STATUS_LABELS = {
  pending_review: 'Pending review',
  reviewed: 'Reviewed',
  dismissed: 'Dismissed',
};

export function formatSeconds(s) {
  if (!Number.isFinite(s)) return '—';
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1).padStart(4, '0');
  return `${m}:${sec}`;
}

export const percent = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—');

const CAMPUS_TZ = 'Asia/Kolkata';

/** ISO time → "16 Nov 2026, 8:15:03 pm IST", always shown in campus time whatever the browser's zone. */
export function formatCampusTime(iso) {
  const d = iso ? new Date(iso) : null;
  if (!d || Number.isNaN(d.getTime())) return '—';
  return `${d.toLocaleString('en-IN', { timeZone: CAMPUS_TZ, dateStyle: 'medium', timeStyle: 'medium' })} IST`;
}

/** "20:00" → "8:00 PM" */
export function formatClock12(hhmm) {
  if (typeof hhmm !== 'string' || !/^\d{2}:\d{2}$/.test(hhmm)) return '—';
  const [h, m] = hhmm.split(':').map(Number);
  return `${((h + 11) % 12) + 1}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`;
}
