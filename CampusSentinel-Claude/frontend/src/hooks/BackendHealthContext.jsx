// Shares one health-polling loop with every component that needs it.

import { createContext, useContext } from 'react';
import { useBackendHealth } from './useBackendHealth.js';
import { config } from '../config.js';

const BackendHealthContext = createContext(null);

export function BackendHealthProvider({ children }) {
  const health = useBackendHealth(config.healthPollIntervalMs);
  return <BackendHealthContext.Provider value={health}>{children}</BackendHealthContext.Provider>;
}

export function useHealth() {
  const value = useContext(BackendHealthContext);
  if (!value) throw new Error('useHealth must be used inside <BackendHealthProvider>');
  return value;
}
