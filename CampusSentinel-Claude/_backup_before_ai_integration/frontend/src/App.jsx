// Route table. AppLayout renders the sidebar + top bar once; the matched
// page is rendered inside it through <Outlet />.

import { Routes, Route, Navigate } from 'react-router';
import AppLayout from './components/layout/AppLayout.jsx';
import DashboardPage from './pages/DashboardPage.jsx';
import IncidentsPage from './pages/IncidentsPage.jsx';
import NotFoundPage from './pages/NotFoundPage.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="incidents" element={<IncidentsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
