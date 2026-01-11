# Deployment Runbook: GCP (Google Cloud Platform)

This guide explains how to deploy your investment bot to GCP using **Cloud Run Jobs** for the rebalance/scoring tasks and **Cloud Run Service** for the dashboard.

## Prerequisites
- A GCP Account and a Project created.
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed locally.
- Docker installed locally.

---

## 1. Database Setup (Cloud SQL)
Do not use a container for your production database.
1.  Go to **Cloud SQL** in the GCP Console.
2.  Create a **PostgreSQL** instance (choose the "Shared Core" or "Micro" tier for low cost, ~$10/mo).
3.  Create a database named `investment_bot`.
4.  Create a user and password.
5.  **Important**: Enable "Public IP" but restrict access to your home IP, OR (better) use the [Cloud SQL Auth Proxy](https://cloud.google.com/sql/docs/postgres/sql-proxy).

---

## 2. Container Registry (Artifact Registry)
1.  Create a repository in **Artifact Registry** named `bot-repo`.
2.  Configure Docker to authenticate with GCP:
    ```bash
    gcloud auth configure-docker us-central1-docker.pkg.dev
    ```
3.  Build and push your image:
    ```bash
    docker build -t us-central1-docker.pkg.dev/[PROJECT_ID]/bot-repo/investment-bot:latest .
    docker push us-central1-docker.pkg.dev/[PROJECT_ID]/bot-repo/investment-bot:latest
    ```

---

## 3. Deployment: The Dashboard (Cloud Run Service)
The dashboard needs to be "always on" or "on-demand."
1.  Go to **Cloud Run** -> **Create Service**.
2.  Select your image from Artifact Registry.
3.  Set **Container Port** to `8501` (Streamlit default).
4.  Add **Environment Variables** (Alpaca keys, DB URL).
5.  Set "Authentication" to "Allow unauthenticated" (if you want it public) or "Require authentication" (if it's just for you).
6.  Deploy.

---

## 4. Deployment: The Bot (Cloud Run Jobs)
Jobs are for tasks that run and then exit.
1.  Go to **Cloud Run** -> **Jobs** -> **Create Job**.
2.  Create a job named `rebalance-job`.
    - **Container Command**: `python`
    - **Arguments**: `src/bot.py`, `--mode`, `rebalance`
3.  Create a second job named `scoring-job`.
    - **Container Command**: `python`
    - **Arguments**: `scripts/run_qlib_score.py`
4.  Add your `.env` variables to both jobs.

---

## 5. Automation: Cloud Scheduler
1.  **Monday Scoring**:
    - Create a **Cloud Scheduler** trigger.
    - Frequency: `0 17 * * 1` (Monday 5 PM ET).
    - Target: **Cloud Run Job** -> `scoring-job`.
2.  **Tuesday Trading**:
    - Create a trigger.
    - Frequency: `35 9 * * 2` (Tuesday 9:35 AM ET).
    - Target: **Cloud Run Job** -> `rebalance-job`.
3.  **Daily Heartbeat**:
    - Create a trigger.
    - Frequency: `0 16 * * *` (Daily 4 PM ET).
    - Target: **Cloud Run Job** -> `rebalance-job` with arg `--mode heartbeat`.

---

## 6. Security (Secrets Manager)
Instead of plain environment variables, use **GCP Secrets Manager** for:
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `DATABASE_URL`
Mount these secrets into your Cloud Run services/jobs for production-grade security.

