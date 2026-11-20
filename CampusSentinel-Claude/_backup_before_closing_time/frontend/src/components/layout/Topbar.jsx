import BackendStatus from '../BackendStatus.jsx';

export default function Topbar({ onMenuClick }) {
  return (
    <header className="topbar">
      <button className="menu-button" onClick={onMenuClick} aria-label="Toggle navigation menu">
        ☰
      </button>
      <div className="topbar-title">Campus Safety Monitoring</div>
      <BackendStatus />
    </header>
  );
}
