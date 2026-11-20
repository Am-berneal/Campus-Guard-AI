// Compact indicator for the top bar. Shows the REAL result of GET /api/health.

import { useHealth } from '../hooks/BackendHealthContext.jsx';

export const STATUS_TEXT = {
  checking: { label: 'Checking…', detail: 'Contacting backend' },
  online: { label: 'Backend online', detail: 'API and database reachable' },
  degraded: { label: 'Database unavailable', detail: 'Backend running, MongoDB not responding' },
  offline: { label: 'Backend offline', detail: 'Cannot reach the backend' },
};

export default function BackendStatus() {
  const { state, checkedAt } = useHealth();
  const text = STATUS_TEXT[state];
  const tooltip = checkedAt
    ? `${text.detail} — last checked ${checkedAt.toLocaleTimeString()}`
    : text.detail;

  return (
    <div className={`status-pill status-${state}`} role="status" aria-live="polite" title={tooltip}>
      <span className="status-dot" aria-hidden="true" />
      <span className="status-label">{text.label}</span>
    </div>
  );
}
