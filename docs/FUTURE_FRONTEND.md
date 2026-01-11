# Future Frontend Migration: Next.js + FastAPI

## Phase 1: API Layer (FastAPI)
- **Location**: `api/` directory.
- **Endpoints**:
  - `GET /performance`: Returns equity curve and benchmark comparison.
  - `GET /decisions`: Returns signals and target weights for a given date.
  - `GET /holdings`: Returns current positions and value.
  - `POST /rebalance/trigger`: Manual trigger for rebalance (requires auth).
- **Auth**: JWT-based authentication for administrative actions.

## Phase 2: Frontend (Next.js)
- **Tech Stack**: Next.js (App Router), Tailwind CSS, Shadcn UI, Recharts/Tremor for charting.
- **Features**:
  - Real-time updates via WebSocket (optional).
  - Advanced filtering for decisions history.
  - Mobile-responsive design for on-the-go monitoring.

## Why this transition?
While Streamlit is excellent for internal MVPs and data science transparency, a Next.js + API architecture provides:
- Better scalability for concurrent users.
- More granular control over UI/UX and styling.
- Secure separation between the "read-only" dashboard and "write-capable" execution controls.



