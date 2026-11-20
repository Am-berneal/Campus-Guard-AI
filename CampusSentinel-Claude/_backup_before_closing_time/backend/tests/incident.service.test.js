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
