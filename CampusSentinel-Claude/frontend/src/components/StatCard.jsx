// A single statistic tile. When `placeholder` is true it shows "—" and a
// visible "Placeholder" badge, so it can never be mistaken for real data.

export default function StatCard({ label, value, hint, placeholder = false }) {
  return (
    <div className={`stat-card ${placeholder ? 'is-placeholder' : ''}`}>
      <div className="stat-card-top">
        <span className="stat-label">{label}</span>
        {placeholder && <span className="badge badge-muted">Placeholder</span>}
      </div>
      <div className="stat-value">{placeholder ? '—' : value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}
