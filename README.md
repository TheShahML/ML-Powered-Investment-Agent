# Investment Bot - Production-Grade Automated Investing

## Setup & Local Development

1. **Prerequisites**

   - Docker & Docker Compose
   - Alpaca API Keys (Paper or Live)
   - Python 3.10+

2. **Configuration**

   - Copy `env_example` to `.env` and fill in your keys.
   - Adjust strategy parameters in `config/*.yaml`.

3. **External Frameworks Setup**

   - **Qlib**: Installed via `requirements.txt`.
   - **LEAN**: Run `bash scripts/setup_lean.sh` to clone the engine.

4. **Start Database**

   ```bash
   docker-compose up -d db
   ```

5. **Train & Score (Qlib)**

   - Train model: `python scripts/run_qlib_train.py` (Implementation placeholder)
   - Generate weekly scores: `python scripts/run_qlib_score.py`
     - This writes Monday close signals to Postgres.

6. **Run Weekly Rebalance (Tuesday)**

   - Dry run: `python src/bot.py --mode rebalance --dry-run`
   - Live/Paper: `python src/bot.py --mode rebalance`

7. **Launch Dashboard**
   ```bash
   streamlit run streamlit_app/app.py
   ```

## Architecture & Integration

Detailed documentation on framework integration is available in `docs/`:

- `docs/ARCHITECTURE.md`: How LEAN, Qlib, and the Bot interact.
- `docs/DATA_FLOW.md`: Step-by-step weekly lifecycle.
- `docs/REVISED_PLAN_V2.md`: Migration status and goals.
- `docs/FUTURE_FRONTEND.md`: Roadmap for Next.js + FastAPI.
- `docs/DEPLOYMENT_RUNBOOK.md`: Step-by-step GCP deployment guide.

## Interview & Portfolio Showcasing

If you are using this repository for interviews, focus on these talking points:

1. **System Engineering**: Explain the Dockerized architecture and how the Postgres ledger ensures auditability.
2. **Framework Integration**: Discuss why you chose **Microsoft Qlib** for research and **QuantConnect LEAN** for backtesting.
3. **Safety First**: Point out the `BROKER_MODE` safety switches and the dry-run capabilities.
4. **Data Integrity**: Explain the Monday-Signal/Tuesday-Trade convention to prevent lookahead bias.

_Note: For security, never commit your `.env` file or actual API keys to GitHub._

## Deployment

### Cloud Run (GCP)

1. Build and push the Docker image to Artifact Registry.
2. Create a Cloud Run job for the rebalance task.
3. Schedule the job using Cloud Scheduler (e.g., `0 14 * * 2` for Tuesday 9am ET / 2pm UTC).
4. Deploy the Streamlit app as a Cloud Run service for the dashboard.

### Fargate (AWS)

1. Push image to ECR.
2. Create an ECS Task Definition.
3. Use EventBridge Scheduler to trigger the task on Tuesday market open.

## Safety Features

- **BROKER_MODE**: Set to `paper` or `live`. Default is `paper`.
- **I_ACKNOWLEDGE_LIVE_TRADING**: Must be set to `true` in `.env` for live execution.
- **Dry Run**: Use `--dry-run` to simulate orders without execution.
- **Min Notional**: Skips trades below a threshold (default $2) to avoid dust.
- **Kill Switch**: The execution service will abort if live mode is set without explicit acknowledgment.

## Monitoring

Check the Streamlit dashboard for performance curves, decision transparency (ML scores/ranks), and system health logs.
