// Calls GET /api/health and classifies the result into one of:
//   'online'      backend answered 200 and MongoDB responded to a ping
//   'degraded'    backend answered 503: it is running, but MongoDB is not available
//   'offline'     backend could not be reached (not started, crashed, wrong URL, timeout)
//
// Note: when the backend is down, the Vite proxy itself answers with an error
// (often 500/502/504) that is NOT our backend's JSON. We only trust a response
// that identifies itself as our service.

import { apiClient, toApiError } from './apiClient.js';

const SERVICE_NAME = 'campussentinel-backend';

function isOurHealthReport(data) {
  return data && typeof data === 'object' && data.service === SERVICE_NAME;
}

export async function fetchBackendHealth({ signal } = {}) {
  const checkedAt = new Date();
  try {
    const { data } = await apiClient.get('/health', { signal });
    if (!isOurHealthReport(data)) {
      return { state: 'offline', report: null, error: 'Unexpected response from /api/health', checkedAt };
    }
    return { state: data.status === 'ok' ? 'online' : 'degraded', report: data, error: null, checkedAt };
  } catch (err) {
    const apiError = toApiError(err);
    if (apiError.kind === 'cancelled') throw err;

    if (apiError.status === 503 && isOurHealthReport(apiError.data)) {
      return { state: 'degraded', report: apiError.data, error: 'Database unavailable', checkedAt };
    }
    return { state: 'offline', report: null, error: apiError.message, checkedAt };
  }
}
