// Express recognises an error handler by its FOUR parameters (err, req, res, next).
// Every error — thrown in a controller, a rejected promise (Express 5 forwards
// these automatically), or next(err) — ends up here, and is returned in the
// format agreed in docs/api-contract.md:
//   { "error": { "code": "...", "message": "...", "details": ... } }

import { config } from '../config/env.js';
import { logger } from '../utils/logger.js';
import { AppError } from '../utils/AppError.js';

// eslint-disable-next-line no-unused-vars
export function errorHandler(err, req, res, next) {
  // Malformed JSON body sent by a client.
  if (err.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: { code: 'INVALID_JSON', message: 'Request body is not valid JSON' },
    });
  }

  // Request body exceeded express.json() size limit.
  if (err.type === 'entity.too.large') {
    return res.status(413).json({
      error: { code: 'PAYLOAD_TOO_LARGE', message: 'Request body is too large' },
    });
  }

  if (err instanceof AppError) {
    const body = { error: { code: err.code, message: err.message } };
    if (err.details !== undefined) body.error.details = err.details;
    return res.status(err.statusCode).json(body);
  }

  // Anything else is a bug or an unexpected failure: log it fully,
  // but never leak internals (stack traces, DB messages) to clients in production.
  logger.error(`Unhandled error on ${req.method} ${req.originalUrl}`, err);

  const body = {
    error: { code: 'INTERNAL_ERROR', message: 'An unexpected error occurred' },
  };
  if (!config.isProduction) {
    body.error.details = { message: err.message, stack: err.stack };
  }
  return res.status(500).json(body);
}
