import { NavLink } from 'react-router';

// Implemented pages are links. Planned pages are shown (so the layout is
// realistic) but disabled and clearly marked — nothing pretends to work.
const NAV_ITEMS = [
  { to: '/dashboard', label: 'Dashboard', icon: '▦' },
  { to: '/incidents', label: 'Incidents', icon: '⚑' },
];

const PLANNED_ITEMS = [
  { label: 'Cameras', icon: '◉' },
  { label: 'Restricted zones', icon: '⬡' },
];

export default function Sidebar() {
  return (
    <aside className="sidebar" aria-label="Main navigation">
      <div className="sidebar-brand">
        <img src="/favicon.svg" alt="" width="28" height="28" />
        <div>
          <div className="sidebar-brand-name">CampusSentinel</div>
          <div className="sidebar-brand-sub">Administrator</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
          >
            <span className="sidebar-icon" aria-hidden="true">{item.icon}</span>
            {item.label}
          </NavLink>
        ))}

        <div className="sidebar-section-label">Planned</div>
        {PLANNED_ITEMS.map((item) => (
          <span key={item.label} className="sidebar-link disabled" aria-disabled="true" title="Not implemented yet">
            <span className="sidebar-icon" aria-hidden="true">{item.icon}</span>
            {item.label}
            <span className="badge badge-muted">Soon</span>
          </span>
        ))}
      </nav>

      <div className="sidebar-footer">
        AI detections are suggestions.
        <br />
        Every incident requires human review.
      </div>
    </aside>
  );
}
