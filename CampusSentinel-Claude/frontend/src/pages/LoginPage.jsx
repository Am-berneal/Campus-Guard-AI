
import { useState } from 'react';
import '../styles/auth.css';

export default function LoginPage({ onLogin }) {
  const [adminId, setAdminId] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  function handleSubmit(event) {
    event.preventDefault();

    // Simple demonstration-level credential check.
    if (adminId.trim() === 'admin' && password === 'admin123') {
      setError('');
      onLogin();
    } else {
      setError('Invalid Admin ID or Password.');
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <div className="login-brand">
          <div className="login-brand-icon">CS</div>
          <h1>CampusSentinel</h1>
          <p>Campus Safety Monitoring System</p>
        </div>

        <div className="login-heading">
          <h2>Administrator Login</h2>
          <p>Sign in to access the monitoring dashboard.</p>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="login-field">
            <label htmlFor="adminId">Admin ID</label>
            <input
              id="adminId"
              type="text"
              placeholder="Enter Admin ID"
              value={adminId}
              onChange={(event) => setAdminId(event.target.value)}
              autoComplete="username"
              required
            />
          </div>

          <div className="login-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              placeholder="Enter password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </div>

          {error && <div className="login-error">{error}</div>}

          <button type="submit" className="login-button">
            Login
          </button>
        </form>

        <p className="login-footer">
          Authorized administrator access only
        </p>
      </div>
    </div>
  );
}
