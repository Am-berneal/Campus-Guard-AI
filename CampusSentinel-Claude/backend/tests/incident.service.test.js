// Unit tests for turning AI analysis JSON into incident documents (no database needed).
// Run from backend/:  node --test tests/
import { test } from 'node:test';
import assert from 'node:assert/strict';

process.env.MONGODB_URI ??= 'mongodb://127.0.0.1:27017/test-not-used';
const { incidentsFromAnalysis } = await import('../src/services/incident.service.js');
const { Incident } = await import('../src/models/Incident.js');

const meta = { cameraId: 'cos', cameraName: 'COS', originalFilename: 'clip.mp4' };
const analysis = {
  upload_id: 'abc123',
  analysis: {
    fighting: { status: 'completed', events: [{ event_id: 'fighting-0', confidence: 0.81, start_s: 0, end_s: 5,
      description: 'Possible fight', evidence: [{ time_s: 2.5, frame_index: 75, confidence: 0.81, image: 'f.jpg' }] }] },
    smoking: { status: 'completed', events: [] },
    restricted_zone: { status: 'unavailable', events: [{ event_id: 'should-be-ignored', confidence: 0.9 }] },
  },
};

test('maps completed module events to incidents', () => {
  const docs = incidentsFromAnalysis(analysis, meta);
  assert.equal(docs.length, 1);
  const d = docs[0];
  assert.equal(d.incidentType, 'fighting');
  assert.equal(d.videoId, 'abc123');
  assert.equal(d.eventKey, 'fighting-0');
  assert.equal(d.evidence[0].imageFile, 'f.jpg');
  assert.equal(d.status, undefined); // status is set only on insert (pending_review)
  assert.equal(new Incident(d).validateSync(), undefined); // passes the Mongoose schema
});

test('ignores unavailable/failed modules and malformed events', () => {
  const bad = structuredClone(analysis);
  bad.analysis.fighting.events.push({ event_id: 42, confidence: 'x' });
  bad.analysis.smoking = { status: 'failed', events: [{ event_id: 'smoking-1', confidence: 0.9 }] };
  assert.equal(incidentsFromAnalysis(bad, meta).length, 1);
});

test('clamps confidence into 0..1 and rejects non-analysis input', () => {
  const a = structuredClone(analysis);
  a.analysis.fighting.events[0].confidence = 1.7;
  assert.equal(incidentsFromAnalysis(a, meta)[0].confidence, 1);
  assert.throws(() => incidentsFromAnalysis({}, meta), /unexpected result/);
});

test('schema rejects invalid type/status', () => {
  const doc = new Incident({ ...incidentsFromAnalysis(analysis, meta)[0], incidentType: 'theft', status: 'guilty' });
  const err = doc.validateSync();
  assert.ok(err.errors.incidentType && err.errors.status);
});

test('restricted-zone events carry closing time and recording time', () => {
  const a = structuredClone(analysis);
  a.analysis.restricted_zone = { status: 'completed', events: [{
    event_id: 'restricted_zone-cos-monitored-area-1-25', confidence: 0.88, start_s: 0, end_s: 1,
    zone_id: 'cos-monitored-area', zone_name: 'COS monitored area', zone_calibrated: false,
    closing_time: '20:00', recorded_local_time: '2026-11-16T20:15:01+05:30', recorded_utc_time: '2026-11-16T14:45:01+00:00',
    description: 'Potential restricted-zone entry detected in COS at 8:15 PM ...',
    evidence: [{ time_s: 1, frame_index: 25, confidence: 0.88, image: 'z.jpg' }] }] };
  const doc = incidentsFromAnalysis(a, meta).find((d) => d.incidentType === 'restricted_zone');
  assert.equal(doc.closingTime, '20:00');
  assert.equal(doc.recordedLocalTime, '2026-11-16T20:15:01+05:30');
  assert.equal(doc.recordedAt.toISOString(), '2026-11-16T14:45:01.000Z');
  assert.deepEqual(doc.zone, { id: 'cos-monitored-area', name: 'COS monitored area', calibrated: false });
  assert.equal(new Incident(doc).validateSync(), undefined);
});

test('older restricted-zone events without closing-time fields still map', () => {
  const a = structuredClone(analysis);
  a.analysis.restricted_zone = { status: 'completed', events: [{ event_id: 'rz-1', confidence: 0.7, zone_id: 'z', zone_name: 'Z' }] };
  const doc = incidentsFromAnalysis(a, meta).find((d) => d.incidentType === 'restricted_zone');
  assert.equal(doc.closingTime, undefined);
  assert.equal(new Incident(doc).validateSync(), undefined);
});

test('time_unknown restricted-zone module creates no incidents', () => {
  const a = structuredClone(analysis);
  a.analysis.restricted_zone = { status: 'time_unknown', events: [] };
  assert.equal(incidentsFromAnalysis(a, meta).filter((d) => d.incidentType === 'restricted_zone').length, 0);
});
