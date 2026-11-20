// Turns the AI service's structured analysis into stored POTENTIAL incidents.
// The AI service already aggregates frames into events (one event = one incident), so repeated
// frames of the same event never create duplicates. Re-analysing a video upserts by
// (videoId, incidentType, eventKey) and never resets an administrator's review.

import mongoose from 'mongoose';
import { Incident, INCIDENT_STATUSES, INCIDENT_TYPES } from '../models/Incident.js';
import { AppError } from '../utils/AppError.js';

const MODEL_NAMES = {
  fighting: 'CNN-LSTM (ResNet18 + LSTM)',
  smoking: 'SmokingCNN + YOLO26n cigarette detector',
  restricted_zone: 'YOLO26n person detector + ByteTrack + zone/time rules',
};

const clamp01 = (v) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : null);

/** Pure function: analysis JSON → array of incident documents (validated shape, not yet saved). */
export function incidentsFromAnalysis(analysis, { cameraId, cameraName, originalFilename }) {
  if (!analysis || typeof analysis !== 'object' || typeof analysis.analysis !== 'object') {
    throw new AppError(502, 'AI_RESULT_INVALID', 'The AI service returned an unexpected result');
  }
  const docs = [];
  for (const type of INCIDENT_TYPES) {
    const module = analysis.analysis[type];
    if (!module || module.status !== 'completed' || !Array.isArray(module.events)) continue;
    for (const ev of module.events) {
      const confidence = clamp01(Number(ev.confidence));
      if (confidence === null || typeof ev.event_id !== 'string') continue; // skip malformed events
      docs.push({
        cameraId,
        cameraName,
        incidentType: type,
        confidence,
        timestamp: new Date(),
        videoId: analysis.upload_id,
        originalFilename,
        eventKey: ev.event_id.slice(0, 200),
        startSeconds: Number.isFinite(ev.start_s) ? Math.max(0, ev.start_s) : undefined,
        endSeconds: Number.isFinite(ev.end_s) ? Math.max(0, ev.end_s) : undefined,
        evidence: (Array.isArray(ev.evidence) ? ev.evidence : []).slice(0, 5).map((e) => ({
          timeSeconds: Number.isFinite(e.time_s) ? Math.max(0, e.time_s) : undefined,
          frameIndex: Number.isInteger(e.frame_index) ? e.frame_index : undefined,
          confidence: clamp01(Number(e.confidence)) ?? undefined,
          imageFile: typeof e.image === 'string' ? e.image.slice(0, 200) : undefined,
          details: e.details ?? undefined,
        })),
        modelName: MODEL_NAMES[type],
        description: typeof ev.description === 'string' ? ev.description.slice(0, 1000) : undefined,
        zone: type === 'restricted_zone' ? { id: ev.zone_id, name: ev.zone_name } : undefined,
      });
    }
  }
  return docs;
}

const dbUnavailable = () =>
  new AppError(503, 'DATABASE_UNAVAILABLE', 'MongoDB is not reachable; incidents cannot be read or stored right now');

function requireDatabase() {
  if (mongoose.connection.readyState !== 1) throw dbUnavailable();
}

// A dropped MongoDB connection is noticed by the driver only after a few seconds, so queries
// can still start and then fail with a network error. Report those as 503, not as a server bug.
const DB_DOWN_ERRORS = /^(MongoNetworkError|MongoServerSelectionError|MongooseServerSelectionError|MongoNotConnectedError|MongoNetworkTimeoutError|MongoTopologyClosedError)$/;
async function withDb(fn) {
  requireDatabase();
  try {
    return await fn();
  } catch (err) {
    if (DB_DOWN_ERRORS.test(err?.name ?? '')) throw dbUnavailable();
    throw err;
  }
}

/** Upsert incidents for one analysed video. Returns the stored documents. */
export async function saveIncidentsFromAnalysis(analysis, meta) {
  const docs = incidentsFromAnalysis(analysis, meta);
  if (docs.length === 0) return [];
  return withDb(async () => {
  const saved = [];
  for (const doc of docs) {
    const { status, ...fields } = doc; // never overwrite an existing review status
    saved.push(
      await Incident.findOneAndUpdate(
        { videoId: doc.videoId, incidentType: doc.incidentType, eventKey: doc.eventKey },
        { $set: fields, $setOnInsert: { status: 'pending_review' } },
        { upsert: true, returnDocument: 'after', runValidators: true, setDefaultsOnInsert: true }
      ).lean()
    );
  }
  return saved;
  });
}

export async function listIncidents({ status, type, cameraId, videoId, page = 1, limit = 50 }) {
  return withDb(async () => {
  const filter = {};
  if (status) filter.status = status;
  if (type) filter.incidentType = type;
  if (cameraId) filter.cameraId = cameraId;
  if (videoId) filter.videoId = videoId;
  const [items, total] = await Promise.all([
    Incident.find(filter).sort({ createdAt: -1 }).skip((page - 1) * limit).limit(limit).lean(),
    Incident.countDocuments(filter),
  ]);
  return { items, total, page, limit };
  });
}

export async function incidentStats() {
  return withDb(async () => {
  const [byStatus, byType] = await Promise.all([
    Incident.aggregate([{ $group: { _id: '$status', count: { $sum: 1 } } }]),
    Incident.aggregate([{ $group: { _id: '$incidentType', count: { $sum: 1 } } }]),
  ]);
  const toObj = (rows, keys) => Object.fromEntries(keys.map((k) => [k, rows.find((r) => r._id === k)?.count ?? 0]));
  const statusCounts = toObj(byStatus, INCIDENT_STATUSES);
  return {
    total: Object.values(statusCounts).reduce((a, b) => a + b, 0),
    byStatus: statusCounts,
    byType: toObj(byType, INCIDENT_TYPES),
  };
  });
}

export async function getIncident(id) {
  if (!mongoose.isValidObjectId(id)) throw new AppError(400, 'INVALID_ID', 'Invalid incident id');
  const doc = await withDb(() => Incident.findById(id).lean());
  if (!doc) throw new AppError(404, 'NOT_FOUND', 'Incident not found');
  return doc;
}

export async function reviewIncident(id, { status, note }) {
  if (!mongoose.isValidObjectId(id)) throw new AppError(400, 'INVALID_ID', 'Invalid incident id');
  const update = { status, 'review.reviewedAt': status === 'pending_review' ? null : new Date() };
  if (note !== undefined) update['review.note'] = note;
  const doc = await withDb(() =>
    Incident.findByIdAndUpdate(id, { $set: update }, { returnDocument: 'after', runValidators: true }).lean()
  );
  if (!doc) throw new AppError(404, 'NOT_FOUND', 'Incident not found');
  return doc;
}
