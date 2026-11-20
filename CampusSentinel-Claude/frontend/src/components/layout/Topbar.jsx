
import BackendStatus from '../BackendStatus.jsx';

export default function Topbar({ onMenuClick, onLogout }) {
  return (
    <header className="topbar">
      <button
        className="menu-button"
        onClick={onMenuClick}
        aria-label="Toggle navigation menu"
      >
        ☰
      </button>

      <div className="topbar-title">Campus Safety Monitoring</div>

      <BackendStatus />

      <button
        className="logout-button"
        onClick={onLogout}
        title="Logout from CampusSentinel"
      >
        Logout
      </button>
    </header>
  );
}
