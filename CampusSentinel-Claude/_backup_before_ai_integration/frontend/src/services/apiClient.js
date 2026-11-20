// One shared Axios instance for every backend call.
// Later phases add things here once (e.g. attaching the JWT), and every
// service file benefits automatically.

import axios from 'axios';
import { config } from '../config.js';

export const apiClient = axios.create({
  baseURL: config.apiUrl,
  timeout: 8000,
  headers: { Accept: 'application/json' },
});

// Converts any Axios error into a small, predictable object the UI can display.
// Our backend sends errors as { error: { code, message } } (docs/api-contract.md).
export function toApiError(err) {
  if (axios.isCancel(err)) {
    return { kind: 'cancelled', status: null, code: 'CANCELLED', message: 'Request cancelled' };
  }
  if (err.response) {
    const body = err.response.data;
    return {
      kind: 'http',
      status: err.response.status,
      code: body?.error?.code ?? `HTTP_${err.response.status}`,
      message: body?.error?.message ?? `Request failed with status ${err.response.status}`,
      data: body,
    };
  }
  if (err.code === 'ECONNABORTED') {
    return { kind: 'timeout', status: null, code: 'TIMEOUT', message: 'The backend did not respond in time' };
  }
  return { kind: 'network', status: null, code: 'NETWORK_ERROR', message: 'Cannot reach the backend' };
}
