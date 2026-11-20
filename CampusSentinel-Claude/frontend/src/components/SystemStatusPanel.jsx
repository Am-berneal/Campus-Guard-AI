// Detailed view of the health report, shown on the dashboard.
// Every value comes from the backend response; nothing is invented.

import { useHealth } from '../hooks/BackendHealthContext.jsx';
import { STATUS_TEXT } from './BackendStatus.jsx';

const BACKEND_TEXT = {
  checking: 'Checking…',
  online: 'Running',
  degraded: 'Running',
  offline: 'Unreachable',
};

function formatUptime(seconds) {
  if (seconds == null) return '—';
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}

export default function SystemStatusPanel() {
  const { state, report, error, checkedAt, isChecking, refresh } = useHealth();
  const db = report?.database;

  const rows = [
    ['Backend', BACKEND_TEXT[state]],
    ['Environment', report?.environment ?? '—'],
    ['Backend uptime', formatUptime(report?.uptimeSeconds)],
    ['Database', db ? `${db.status}${db.name ? ` (${db.name})` : ''}` : '—'],
    ['Database ping', db?.pingMs != null ? `${db.pingMs} ms` : '—'],
    ['Last checked', checkedAt ? checkedAt.toLocaleTimeString() : '—'],
  ];

  return (
    <section className="panel">
      <div className="panel-header">
        <h2>System status</h2>
        <button className="button" onClick={refresh} disabled={isChecking}>
          {isChecking ? 'Checking…' : 'Check now'}
        </button>
      </div>

      <div className={`status-banner status-${state}`}>
        <span className="status-dot" aria-hidden="true" />
        {STATUS_TEXT[state].detail}
        {error && state === 'offline' && <span className="status-error"> — {error}</span>}
      </div>

      <dl className="detail-list">
        {rows.map(([label, value]) => (
          <div key={label} className="detail-row">
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <p className="panel-note">Source: live response from <code>GET /api/health</code>.</p>
    </section>
  );
}
