# CampusSentinel — Backend

Express.js REST API for CampusSentinel, using MongoDB through Mongoose.

**Current scope (Phase 1):** configuration, MongoDB connection, health check, error handling, graceful shutdown.
Not yet implemented: authentication, cameras, zones, incidents, Socket.IO.

## Requirements

- Node.js ≥ 20.19 (developed with v26.2.0)
- MongoDB running locally on `127.0.0.1:27017` (Windows service)

## Folder structure

```
backend/
├── package.json            dependencies + npm scripts
├── .env.example            configuration template (copy to .env)
├── README.md
└── src/
    ├── server.js           entry point: connect DB → start HTTP → graceful shutdown
    ├── app.js              builds the Express app (middleware, routes, error handling)
    ├── config/
    │   ├── env.js          loads + validates environment variables
    │   └── database.js     Mongoose connect / disconnect / status / ping
    ├── routes/
    │   ├── index.js        mounts every feature router under /api
    │   └── health.routes.js
    ├── controllers/
    │   └── health.controller.js   HTTP layer: status code + JSON response
    ├── services/
    │   └── health.service.js      logic: builds the health report
    ├── middleware/
    │   ├── notFound.js     JSON 404 for unknown routes
    │   └── errorHandler.js consistent JSON errors, no leaked internals in production
    └── utils/
        ├── AppError.js     error class carrying an HTTP status + code
        └── logger.js       timestamped console logging
```

Request flow: `route → controller → service → config/database`.

## Setup (PowerShell)

```powershell
cd "C:\Users\hp\Projects\CampusSentinel-Claude\backend"
Copy-Item .env.example .env
npm install
```

Edit `.env` if needed. The defaults match a standard local MongoDB install.

> `.env` holds your real configuration. Do not share it. Later phases will add secrets (JWT, service key) to it.

## Run

| Command | Use |
|---|---|
| `npm run dev` | Development. Restarts automatically when you save a file (`node --watch`) |
| `npm start` | Production-style run, no file watching |

Stop with **Ctrl+C**. The server closes HTTP connections and the MongoDB connection before exiting.

## Health check

```powershell
Invoke-RestMethod http://localhost:5000/api/health | ConvertTo-Json -Depth 5
```

| HTTP status | `status` | Meaning |
|---|---|---|
| 200 | `ok` | Server running and MongoDB answered a ping |
| 503 | `degraded` | Server running but MongoDB is unreachable (it reconnects automatically) |

Example shape (values will differ):

```json
{
  "status": "ok",
  "service": "campussentinel-backend",
  "environment": "development",
  "timestamp": "…",
  "uptimeSeconds": 12,
  "database": { "status": "connected", "name": "campussentinel", "host": "127.0.0.1", "pingMs": 1.2 }
}
```

## Startup behaviour

- **Missing/invalid configuration** → exits with a message naming the variable.
- **MongoDB unreachable at startup** → waits `MONGODB_SERVER_SELECTION_TIMEOUT_MS` (default 5 s), then exits with a clear error. The server never accepts requests without a working database.
- **MongoDB drops while running** → Mongoose reconnects automatically. Health returns 503 until it is back.
- **Port already in use** → exits with a message telling you to change `PORT` or stop the other process.

## Why `127.0.0.1` instead of `localhost`?

Recent Node.js versions may resolve `localhost` to the IPv6 address `::1` first, while MongoDB on Windows listens on IPv4 `127.0.0.1` by default. Using the IP avoids a confusing `ECONNREFUSED ::1:27017` error.
