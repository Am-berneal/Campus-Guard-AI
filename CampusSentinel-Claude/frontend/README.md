# CampusSentinel — Frontend

React + Vite administrator dashboard for CampusSentinel.

**Current scope (Phase 1):** layout with sidebar, dashboard with clearly labelled placeholder statistics, incidents page (empty state), and a live backend status indicator that calls the real `GET /api/health`.
Not yet implemented: login, incidents from the backend, real-time notifications, cameras, zones.

## Requirements

- Node.js ≥ 22.22 (developed with v26.2.0)
- The backend running on `http://127.0.0.1:5000` (see `../backend/README.md`) for the status indicator to show "online"

## Folder structure

```
frontend/
├── package.json         dependencies + npm scripts
├── vite.config.js       dev server (port 5173) + proxy /api → backend
├── index.html           HTML shell; React mounts into <div id="root">
├── .env.example         configuration template (copy to .env)
├── public/favicon.svg
└── src/
    ├── main.jsx         entry: StrictMode, router, health provider, CSS
    ├── App.jsx          route table
    ├── config.js        reads VITE_* environment variables
    ├── services/
    │   ├── apiClient.js     shared Axios instance + error normaliser
    │   └── healthService.js GET /api/health → online | degraded | offline
    ├── hooks/
    │   ├── useBackendHealth.js     polling logic (interval, cancel, pause when tab hidden)
    │   └── BackendHealthContext.jsx shares ONE polling loop with the whole app
    ├── components/
    │   ├── layout/AppLayout.jsx    sidebar + top bar + page area
    │   ├── layout/Sidebar.jsx
    │   ├── layout/Topbar.jsx
    │   ├── BackendStatus.jsx       status pill in the top bar
    │   ├── SystemStatusPanel.jsx   detailed health on the dashboard
    │   ├── StatCard.jsx            statistic tile (supports "placeholder")
    │   ├── EmptyState.jsx
    │   └── PageHeader.jsx
    ├── pages/
    │   ├── DashboardPage.jsx
    │   ├── IncidentsPage.jsx
    │   └── NotFoundPage.jsx
    └── styles/
        ├── global.css       colours, fonts, base elements
        ├── layout.css       sidebar/top bar, responsive breakpoint (900px)
        └── components.css   cards, panels, badges, status colours
```

## Setup (Command Prompt)

```cmd
cd /d C:\Users\hp\Projects\CampusSentinel-Claude\frontend
copy .env.example .env
npm install
```

## Run

Start the backend first (separate window: `cd backend` then `npm run dev`), then:

```cmd
npm run dev
```

Open http://localhost:5173

| Command | Use |
|---|---|
| `npm run dev` | Development server with instant reload |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the `dist/` build locally to check it |

## How API calls reach the backend

The browser calls `/api/...` on the **same origin** as the page (`localhost:5173`).
The Vite dev server forwards those requests to `BACKEND_PROXY_TARGET` (`http://127.0.0.1:5000`).
Because the browser only talks to one origin, no CORS configuration is needed during development.

## Backend status indicator

| Indicator | Meaning |
|---|---|
| Backend online (green) | `/api/health` returned 200: API and MongoDB reachable |
| Database unavailable (amber) | `/api/health` returned 503: backend running, MongoDB not responding |
| Backend offline (red) | No valid answer: backend not started, crashed, or wrong proxy target |

Re-checked every `VITE_HEALTH_POLL_INTERVAL_MS` (default 15 s) while the tab is visible, or immediately with **Check now**.

## Environment variables

| Variable | Sent to browser? | Purpose |
|---|---|---|
| `VITE_API_URL` | Yes | Base URL for API calls (default `/api`) |
| `VITE_HEALTH_POLL_INTERVAL_MS` | Yes | Health polling interval |
| `BACKEND_PROXY_TARGET` | No (dev server only) | Where Vite forwards `/api` |

Anything prefixed `VITE_` is embedded in the JavaScript bundle and visible to anyone using the site. **Never put secrets in frontend `.env`.**
After changing `.env`, stop and restart `npm run dev`.
