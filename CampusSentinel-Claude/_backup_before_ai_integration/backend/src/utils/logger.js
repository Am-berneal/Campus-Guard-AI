// Minimal timestamped logger. Enough for now; can be swapped for a
// logging library later without changing the rest of the code.

function format(level, message) {
  return `[${new Date().toISOString()}] ${level.padEnd(5)} ${message}`;
}

export const logger = {
  info: (message) => console.log(format('INFO', message)),
  warn: (message) => console.warn(format('WARN', message)),
  error: (message, err) => {
    console.error(format('ERROR', message));
    if (err) console.error(err);
  },
};
