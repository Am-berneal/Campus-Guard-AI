
import { useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router';
import Sidebar from './Sidebar.jsx';
import Topbar from './Topbar.jsx';

export default function AppLayout({ onLogout }) {
  // Controls the mobile sidebar.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  // Close the mobile sidebar after navigating.
  useEffect(() => setSidebarOpen(false), [location.pathname]);

  return (
    <div className={`app-shell ${sidebarOpen ? 'sidebar-open' : ''}`}>
      <Sidebar />

      {sidebarOpen && (
        <button
          className="sidebar-backdrop"
          aria-label="Close menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="app-main">
        <Topbar
          onMenuClick={() => setSidebarOpen((open) => !open)}
          onLogout={onLogout}
        />

        <main className="app-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
