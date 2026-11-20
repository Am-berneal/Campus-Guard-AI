
import { useState } from 'react';
import { Routes, Route, Navigate } from 'react-router';

import AppLayout from './components/layout/AppLayout.jsx';
import DashboardPage from './pages/DashboardPage.jsx';
import IncidentsPage from './pages/IncidentsPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';
import LoginPage from './pages/LoginPage.jsx';

// Key used to remember login in the browser.
const AUTH_KEY = 'campussentinel_admin_logged_in';

export default function App() {
  // Restore login state when the application loads or refreshes.
  const [isAuthenticated, setIsAuthenticated] = useState(
    () => localStorage.getItem(AUTH_KEY) === 'true'
  );

  function handleLogin() {
    localStorage.setItem(AUTH_KEY, 'true');
    setIsAuthenticated(true);
  }

  function handleLogout() {
    localStorage.removeItem(AUTH_KEY);
    setIsAuthenticated(false);
  }

  return (
    <Routes>
      {/* Public login route */}
      <Route
        path="/login"
        element={
          isAuthenticated
            ? <Navigate to="/dashboard" replace />
            : <LoginPage onLogin={handleLogin} />
        }
      />

      {/* Protected application routes */}
      <Route
        element={
          isAuthenticated
            ? <AppLayout onLogout={handleLogout} />
            : <Navigate to="/login" replace />
        }
      >
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="incidents" element={<IncidentsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
