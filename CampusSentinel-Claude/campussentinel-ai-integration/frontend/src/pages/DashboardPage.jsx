import { useEffect, useState } from 'react';
import PageHeader from '../components/PageHeader.jsx';
import StatCard from '../components/StatCard.jsx';
import SystemStatusPanel from '../components/SystemStatusPanel.jsx';
import VideoUploadPanel from '../components/VideoUploadPanel.jsx';
import ModelStatusPanel from '../components/ModelStatusPanel.jsx';
import { getIncidentStats } from '../services/incidentService.js';

// Statistic cards show REAL counts from GET /api/incidents/stats.
// If they cannot be loaded (e.g. MongoDB down), they fall back to clearly marked placeholders.
const INCIDENT_STATS = [
  { label: 'Total potential incidents', hint: 'All time', value: (s) => s.total },
  { label: 'Awaiting review', hint: 'Status: pending review', value: (s) => s.byStatus.pending_review },
  { label: 'Reviewed', hint: 'Checked by an administrator', value: (s) => s.byStatus.reviewed },
  { label: 'Dismissed', hint: 'Marked as not actionable', value: (s) => s.byStatus.dismissed },
];

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [statsError, setStatsError] = useState(null);

  useEffect(() => {
    let active = true;
    getIncidentStats()
      .then((s) => active && setStats(s))
      .catch((err) => active && setStatsError(err));
    return () => {
      active = false;
    };
  }, []);

  return (
    <>
      <PageHeader title="Dashboard" subtitle="Overview of campus incidents and system health." />

      {/* Upload from simulated campus cameras → AI analysis → potential incidents for review. */}
      <VideoUploadPanel />

      {statsError && (
        <div className="notice">
          Incident statistics could not be loaded ({statsError.message}). Showing placeholders.
        </div>
      )}

      <div className="stat-grid">
        {INCIDENT_STATS.map((s) => (
          <StatCard key={s.label} label={s.label} hint={s.hint} value={stats ? s.value(stats) : undefined}
                    placeholder={!stats} />
        ))}
      </div>

      <div className="two-column">
        <SystemStatusPanel />
        <ModelStatusPanel />
      </div>
    </>
  );
}
