// An error that carries an HTTP status code and a stable, machine-readable code.
// Throw it anywhere in a route/controller/service; the error handler formats it.
//   throw new AppError(404, 'NOT_FOUND', 'Camera not found');

export class AppError extends Error {
  constructor(statusCode, code, message, details) {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
  }
}
