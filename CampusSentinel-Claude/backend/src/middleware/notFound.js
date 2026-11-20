// Runs only when no route matched. Turns it into a consistent JSON 404.

import { AppError } from '../utils/AppError.js';

export function notFound(req, res, next) {
  next(new AppError(404, 'NOT_FOUND', `Route not found: ${req.method} ${req.originalUrl}`));
}
