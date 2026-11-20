// Potential incidents from AI analysis, for administrator review.
// The administrator decides; the system never takes action. Guards are contacted by walkie-talkie.

import { useCallback, useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import EmptyState from '../components/EmptyState.jsx';
import {
  evidenceUrl,
  formatSeconds,
  INCIDENT_TYPE_LABELS,
  listIncidents,
  percent,
  reviewIncident,
  STATUS_LABELS,
} from '../services/incidentService.js';
import '../styles/incidents.css';

const FILTERS = [
  { value: 'pending_review', label: 'Pending review' },
  { value: 'reviewed', label: 'Reviewed' },
  { value: 'dismissed', label: 'Dismissed' },
  { value: '', label: 'All' },
];

function IncidentCard({ incident, onUpdated }) {
  const [note, setNote] = useState(incident.review?.note ?? '');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  async function setStatus(status) {
    setSaving(true);
    setError(null);
    try {
      onUpdated(await reviewIncident(incident._id, status, note.trim()));
    } catch (err) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <article className={`incident-card type-${incident.incidentType}`}>
      <header className="incident-head">
        <div>
          <h3>{INCIDENT_TYPE_LABELS[incident.incidentType] ?? incident.incidentType}</h3>
          <div className="incident-sub">
            {incident.cameraName} · <span className="badge badge-muted">simulated camera</span> ·{' '}
            {incident.originalFilename ?? incident.videoId}
          </div>
        </div>
        <div className="incident-right">
          <span className={`badge status-badge-${incident.status}`}>{STATUS_LABELS[incident.status]}</span>
          <span className="incident-conf">confidence {percent(incident.confidence)}</span>
        </div>
      </header>

      <p className="incident-desc">{incident.description}</p>
      <dl className="incident-facts">
        <div><dt>In video</dt><dd>{formatSeconds(incident.startSeconds)} – {formatSeconds(incident.endSeconds)}</dd></div>
        <div><dt>Detected</dt><dd>{new Date(incident.timestamp).toLocaleString()}</dd></div>
        <div><dt>Model</dt><dd>{incident.modelName}</dd></div>
        {incident.zone?.name && <div><dt>Zone</dt><dd>{incident.zone.name}</dd></div>}
      </dl>

      {incident.evidence?.length > 0 && (
        <div className="evidence-strip">
          {incident.evidence.map((e, i) => (
            <figure key={i}>
              {e.imageFile ? (
                <a href={evidenceUrl(incident._id, i)} target="_blank" rel="noreferrer">
                  <img src={evidenceUrl(incident._id, i)} alt={`Evidence at ${formatSeconds(e.timeSeconds)}`} loading="lazy" />
                </a>
              ) : (
                <div className="evidence-missing">no image</div>
              )}
              <figcaption>
                {formatSeconds(e.timeSeconds)} · {percent(e.confidence)}
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      <div className="review-row">
        <input
          type="text"
          placeholder="Review note (optional), e.g. 'Guard informed by radio' or 'False alarm'"
          value={note}
          maxLength={1000}
          onChange={(e) => setNote(e.target.value)}
          disabled={saving}
        />
        {incident.status !== 'reviewed' && (
          <button className="button button-review" onClick={() => setStatus('reviewed')} disabled={saving}>
            Mark reviewed
          </button>
        )}
        {incident.status !== 'dismissed' && (
          <button className="button" onClick={() => setStatus('dismissed')} disabled={saving}>Dismiss</button>
        )}
        {incident.status !== 'pending_review' && (
          <button className="button" onClick={() => setStatus('pending_review')} disabled={saving}>Reopen</button>
        )}
      </div>
      {incident.review?.reviewedAt && (
        <div className="incident-sub">Last review: {new Date(incident.review.reviewedAt).toLocaleString()}</div>
      )}
      {error && <p className="field-error">{error}</p>}
    </article>
  );
}

export default function IncidentsPage() {
  const [filter, setFilter] = useState('pending_review');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await listIncidents(filter ? { status: filter, limit: 100 } : { limit: 100 }));
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    load();
  }, [load]);

  function onUpdated(updated) {
    // Remove from the list if it no longer matches the filter; otherwise update in place.
    setData((d) => ({
      ...d,
      items: d.items
        .map((i) => (i._id === updated._id ? updated : i))
        .filter((i) => !filter || i.status === filter),
    }));
  }

  return (
    <>
      <PageHeader
        title="Incidents"
        subtitle="Potential safety events found by AI analysis of uploaded videos. Review each one; the system takes no action itself."
      />

      <div className="incident-toolbar">
        <div className="filter-tabs" role="tablist">
          {FILTERS.map((f) => (
            <button key={f.label} role="tab" aria-selected={filter === f.value}
                    className={`filter-tab ${filter === f.value ? 'active' : ''}`} onClick={() => setFilter(f.value)}>
              {f.label}
            </button>
          ))}
        </div>
        <button className="button" onClick={load} disabled={loading}>{loading ? 'Loading…' : 'Refresh'}</button>
      </div>

      {error && (
        <div className="upload-alert upload-alert-error" role="alert">
          Could not load incidents: {error.message} {error.code && <code>{error.code}</code>}
        </div>
      )}

      {data && data.items.length === 0 && (
        <section className="panel">
          <EmptyState icon="⚑" title="No incidents in this list">
            <p>Upload a video on the dashboard and run AI analysis to create potential incidents.</p>
          </EmptyState>
        </section>
      )}

      <div className="incident-list">
        {data?.items.map((inc) => <IncidentCard key={inc._id} incident={inc} onUpdated={onUpdated} />)}
      </div>
      {data && data.total > data.items.length && (
        <p className="panel-note">Showing {data.items.length} of {data.total}.</p>
      )}
    </>
  );
}
