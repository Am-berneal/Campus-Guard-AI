import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import SystemStatusPanel from '../components/SystemStatusPanel.jsx';
import VideoUploadPanel from '../components/VideoUploadPanel.jsx';

// Statistic cards are PLACEHOLDERS until GET /api/incidents/stats exists (Phase 2).
const INCIDENT_STATS = [
  { label: 'Total incidents', hint: 'All time' },
  { label: 'Awaiting review', hint: 'Status: new' },
  { label: 'Reviewed', hint: 'Confirmed by an administrator' },
  { label: 'Dismissed', hint: 'Marked as not actionable' },
];

const DETECTION_MODULES = [
  { name: 'Smoking detection', phase: 'Phase 3' },
  { name: 'Fighting detection', phase: 'Phase 4' },
  { name: 'Restricted zone / time', phase: 'Phase 5' },
];

export default function DashboardPage() {
  return (
    <>
      <PageHeader title="Dashboard" subtitle="Overview of campus incidents and system health." />

      {/* Video upload from simulated campus cameras (stored by the AI service; analysis not connected yet). */}
      <VideoUploadPanel />

      <div className="notice">
        Incident statistics are <strong>placeholders</strong>. They will show real counts once incident
        management is connected in Phase 2.
      </div>

      <div className="stat-grid">
        {INCIDENT_STATS.map((s) => (
          <StatCard key={s.label} label={s.label} hint={s.hint} placeholder />
        ))}
      </div>

      <div className="two-column">
        <SystemStatusPanel />

        <section className="panel">
          <div className="panel-header">
            <h2>Detection modules</h2>
          </div>
          <ul className="module-list">
            {DETECTION_MODULES.map((m) => (
              <li key={m.name}>
                <span>{m.name}</span>
                <span className="badge badge-muted">Not integrated · {m.phase}</span>
              </li>
            ))}
          </ul>
          <p className="panel-note">No AI models are running yet.</p>
        </section>
      </div>
    </>
  );
}
