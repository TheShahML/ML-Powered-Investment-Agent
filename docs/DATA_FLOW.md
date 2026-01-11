# Data Flow & Lifecycle

## 1. Weekly Lifecycle (Live/Paper)

### Monday (Signal Generation)
- **16:00 ET (Market Close)**: Market data is finalized.
- **17:00 ET**: `scripts/run_qlib_score.py` is triggered.
- **Output**: Scores for the entire universe are written to the `signals` table in Postgres, associated with a new `run_id`.

### Tuesday (Execution)
- **09:30 ET (Market Open)**: `src/bot.py --mode rebalance` is triggered.
- **Step 1**: Reads the latest successful `run_id` and its associated `signals` from Postgres.
- **Step 2**: Fetches current holdings from Alpaca.
- **Step 3**: Computes target weights (e.g., Top 15 equal weight).
- **Step 4**: Places orders via Alpaca (market or limit-on-open).
- **Step 5**: Updates the `ledger`, `orders`, and `positions` tables in Postgres.

## 2. Training Lifecycle (Asynchronous)
- **Frequency**: Monthly or Quarterly.
- **Tool**: `scripts/run_qlib_train.py`.
- **Flow**:
  1. Pulls historical data from Alpaca/YFinance into Qlib format.
  2. Runs walk-forward training.
  3. Registers the new model version in Postgres.
  4. Saves the model artifact to `research/models/`.

## 3. Backtesting Lifecycle (On-Demand)
- **Tool**: `integrations/lean_runner/run_backtest.py`.
- **Flow**:
  1. Exports weekly scores from Postgres to a CSV format readable by LEAN.
  2. Triggers the LEAN engine using the exported signals.
  3. Parses LEAN's `results.json` and writes results to `backtest_runs` and `backtest_equity_curve` tables.



