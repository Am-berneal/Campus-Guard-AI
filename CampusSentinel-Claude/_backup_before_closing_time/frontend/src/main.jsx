import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router';
import App from './App.jsx';
import { BackendHealthProvider } from './hooks/BackendHealthContext.jsx';
import './styles/global.css';
import './styles/layout.css';
import './styles/components.css';

// StrictMode (development only) mounts components twice to expose bugs such as
// missing effect cleanup. That is why you may see two /api/health requests at startup.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <BackendHealthProvider>
        <App />
      </BackendHealthProvider>
    </BrowserRouter>
  </StrictMode>
);
