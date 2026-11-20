// Shows the structured AI analysis returned by the backend for ONE uploaded video.
// Everything displayed comes from the AI service's response; nothing is invented here.

import { Link } from 'react-router';
import { formatCampusTime, formatClock12, formatSeconds, percent } from '../services/incidentService.js';

const MODULE_TITLES = {
  fighting: 'Fighting (CNN-LSTM)',
  smoking: 'Smoking (CNN + cigarette YOLO)',
  restricted_zone: 'Restricted zone / time (person YOLO + rules)',
};

const STATUS_BADGE = {
  completed: ['Analysed', 'badge-ok'],
  unavailable: ['Model unavailable', 'badge-muted'],
  not_configured: ['Not configured', 'badge-muted'],
  failed: ['Failed', 'badge-danger'],
  time_unknown: ['Recording time needed', 'badge-warn'],
};

function ModuleStatus({ module }) {
  const [text, cls] = STATUS_BADGE[module.status] ?? [module.status, 'badge-muted'];
  return <span className={`badge ${cls}`}>{text}</span>;
}

function EventList({ events }) {
  if (!events?.length) return <p className="result-none">No potential incident.</p>;
  return (
    <ul className="event-list">
      {events.map((ev) => (
        <li key={ev.event_id}>
          <div className="event-head">
            <strong>Potential incident</strong>
            <span>
              {formatSeconds(ev.start_s)} – {formatSeconds(ev.end_s)} · confidence {percent(ev.confidence)}
            </span>
          </div>
          <div className="event-desc">{ev.description}</div>
          {ev.evidence?.length > 0 && (
            <div className="event-evidence">
              Evidence at {ev.evidence.map((e) => formatSeconds(e.time_s)).join(', ')}
            </div>
          )}
        </li>
      ))}
    </ul>
  );
}

function ModuleBody({ name, m }) {
  if (m.status === 'time_unknown') {
    return (
      <>
        <p className="result-reason">{m.reason}</p>
        {m.rule?.closing_time && (
          <p className="result-note">Closing time for this camera: {formatClock12(m.rule.closing_time)} ({m.rule.campus_timezone}).</p>
        )}
      </>
    );
  }
  if (m.status === 'unavailable' || m.status === 'not_configured') {
    return <p className="result-reason">{m.reason}</p>;
  }
  if (m.status === 'failed') {
    return <p className="result-reason">The module failed on this video: {m.error}</p>;
  }
  if (name === 'fighting') {
    return (
      <>
        <dl className="result-metrics">
          <div><dt>Prediction</dt><dd>{m.predicted_class}</dd></div>
          <div><dt>Fight probability (max window)</dt><dd>{percent(m.fight_probability)}</dd></div>
          <div><dt>Non-fight probability</dt><dd>{percent(m.nonfight_probability)}</dd></div>
          <div><dt>Threshold</dt><dd>{m.threshold}</dd></div>
          <div><dt>Windows analysed</dt><dd>{m.sampling?.windows_analysed} × {m.sampling?.window_seconds}s, 16 frames each</dd></div>
        </dl>
        {m.notes?.length > 0 && <p className="result-reason">{m.notes.join(' ')}</p>}
        <EventList events={m.events} />
      </>
    );
  }
  if (name === 'smoking') {
    const s = m.summary ?? {};
    return (
      <>
        <dl className="result-metrics">
          <div><dt>Frames sampled</dt><dd>{s.frames_sampled} (every {s.sample_interval_s}s)</dd></div>
          <div><dt>CNN "smoking" frames</dt><dd>{s.frames_cnn_positive} · max {percent(s.max_cnn_smoking_probability)}</dd></div>
          <div><dt>Frames with a cigarette-like object</dt><dd>{s.frames_with_cigarette} · max {percent(s.max_cigarette_confidence)}</dd></div>
          <div><dt>Event rule</dt><dd>{m.rule} (CNN ≥ {m.thresholds?.cnn}, cigarette ≥ {m.thresholds?.cigarette})</dd></div>
        </dl>
        <p className="result-note">A detected cigarette alone does not prove that someone is smoking.</p>
        {m.notes?.length > 0 && <p className="result-reason">{m.notes.join(' ')}</p>}
        <EventList events={m.events} />
      </>
    );
  }
  const s = m.summary ?? {};
  return (
    <>
      <dl className="result-metrics">
        <div><dt>Frames sampled</dt><dd>{s.frames_sampled} ({s.sample_fps} per second)</dd></div>
        <div><dt>Max people in a frame</dt><dd>{s.max_people_in_frame}</dd></div>
        <div><dt>Closing time</dt><dd>{formatClock12(m.rule?.closing_time)} (no entry from then on)</dd></div>
        <div><dt>Recording</dt><dd>{formatCampusTime(s.recorded_at_local)} – {formatCampusTime(s.video_end_local)}</dd></div>
        <div><dt>Frames in restricted period</dt><dd>{s.frames_in_restricted_period} of {s.frames_sampled}</dd></div>
        <div>
          <dt>Zones</dt>
          <dd>{m.zones?.map((z) => `${z.name}${z.calibrated ? '' : ' [placeholder boundary]'}`).join('; ')}</dd>
        </div>
      </dl>
      {s.recorded_at_note && <p className="result-note">{s.recorded_at_note}</p>}
      <EventList events={m.events} />
    </>
  );
}

export default function AnalysisResults({ result }) {
  const { analysis, incidents } = result;
  const modules = analysis.analysis ?? {};
  return (
    <div className="analysis-results">
      <div className={`upload-alert ${analysis.potential_incident_count ? 'upload-alert-warn' : 'upload-alert-success'}`}>
        <strong>
          AI analysis {analysis.status === 'completed' ? 'completed' : `finished (${analysis.status})`}:
        </strong>{' '}
        {analysis.potential_incident_count} potential incident(s) found, {incidents.length} stored for review.
        {incidents.length > 0 && (
          <> <Link to="/incidents">Review them on the Incidents page →</Link></>
        )}
      </div>

      <div className="result-meta">
        {analysis.camera_name} · simulated camera (uploaded recording) · {analysis.video?.duration_seconds}s video ·
        analysed on {analysis.processing?.device?.toUpperCase()} in {analysis.processing?.duration_seconds}s
        {analysis.cached && ' · cached result'}
      </div>

      <div className="module-results">
        {['fighting', 'smoking', 'restricted_zone'].map((name) =>
          modules[name] ? (
            <section key={name} className="module-result">
              <div className="module-result-head">
                <h3>{MODULE_TITLES[name]}</h3>
                <ModuleStatus module={modules[name]} />
              </div>
              <ModuleBody name={name} m={modules[name]} />
            </section>
          ) : null
        )}
      </div>
      <p className="panel-note">{analysis.disclaimer}</p>
    </div>
  );
}
