// React hook: checks backend health immediately, then every N ms.
// Used by both the top-bar indicator and the dashboard status panel, via
// BackendHealthContext, so only ONE polling loop runs for the whole app.

import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchBackendHealth } from '../services/healthService.js';

const INITIAL = { state: 'checking', report: null, error: null, checkedAt: null };

export function useBackendHealth(intervalMs) {
  const [health, setHealth] = useState(INITIAL);
  const [isChecking, setIsChecking] = useState(false);
  const controllerRef = useRef(null);

  const check = useCallback(async () => {
    // Cancel a previous request that is still running, so results never arrive out of order.
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setIsChecking(true);
    try {
      const result = await fetchBackendHealth({ signal: controller.signal });
      setHealth(result);
    } catch {
      // Only cancellations reach here; a newer request is already on its way.
    } finally {
      if (controllerRef.current === controller) setIsChecking(false);
    }
  }, []);

  useEffect(() => {
    check();
    const id = setInterval(() => {
      // Don't poll while the browser tab is hidden.
      if (document.visibilityState === 'visible') check();
    }, intervalMs);

    const onVisible = () => document.visibilityState === 'visible' && check();
    document.addEventListener('visibilitychange', onVisible);

    // Cleanup when the component unmounts (and React StrictMode's dev-only re-mount).
    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVisible);
      controllerRef.current?.abort();
    };
  }, [check, intervalMs]);

  return { ...health, isChecking, refresh: check };
}
