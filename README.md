# ML-Powered Investment Bot

A fully automated, production-grade investment system that uses machine learning to select and trade stocks from the S&P 500 and NASDAQ-100. The system runs autonomously on GitHub Actions, making data-driven investment decisions every trading day and rebalancing the portfolio monthly.

## What This System Does

This investment bot implements a **momentum-based trading strategy** enhanced by machine learning. Every trading day, it:

1. **Analyzes 700-800 stocks** from the S&P 500 and NASDAQ-100 indexes
2. **Predicts future performance** using three separate ML models trained on different time horizons
3. **Selects the top 25 stocks** with the strongest expected returns
4. **Rebalances the portfolio** monthly (every 20 trading days) to maintain optimal positions
5. **Tracks performance** against pure momentum and market benchmarks to validate that ML adds value

The system trades real money through Alpaca Markets and sends daily updates via Discord, making it a complete hands-off investment solution.

## The Investment Strategy

### Core Philosophy: Momentum Investing

The strategy is built on **momentum investing**, one of the most well-documented market anomalies in academic finance. Momentum refers to the tendency of stocks that have performed well over the past 6-12 months to continue outperforming in the near future.

**Academic Foundation:**
- Based on research by Jegadeesh & Titman (1993) showing that momentum strategies generate significant returns
- Uses the classic **12-1 momentum indicator**: 12-month returns minus the most recent month (to avoid short-term reversals)
- Augmented with volatility, volume, and mean reversion signals

### Machine Learning Enhancement

While momentum works, machine learning can potentially improve it by:
- **Learning complex patterns** between multiple momentum signals, volatility, and volume trends
- **Adapting to market conditions** through periodic retraining
- **Ranking stocks more accurately** than simple sorting by momentum

The system uses **XGBoost** (gradient boosted decision trees), a proven algorithm for structured data that won't overfit if properly constrained.

## Multi-Horizon ML Implementation

### Three Models, Three Time Horizons

The system trains **three separate XGBoost models** to predict stock performance over different timeframes:

1. **1-Day Model**: Predicts next-day returns (informational, shows short-term sentiment)
2. **5-Day Model**: Predicts 5-day forward returns (captures weekly trends)
3. **20-Day Model**: Predicts 20-day forward returns (PRIMARY - used for actual trading)

**Why three models?**
- Different horizons capture different market dynamics
- Comparing all three provides confidence in the signals
- The 20-day model matches the rebalancing frequency (no horizon mismatch)

### Features: Simple and Academically-Backed

Each model uses only **5 features** to avoid overfitting:

1. **12-1 Momentum**: 12-month return minus 1-month return (classic factor)
2. **6-Month Momentum**: Medium-term trend
3. **3-Month Volatility**: Risk adjustment (prefer stable winners)
4. **Volume Trend**: Liquidity signal (rising volume confirms moves)
5. **Mean Reversion**: Short-term reversal (contrarian signal)

These features are well-documented in academic research and work across different time periods and market conditions.

### Portfolio Construction

From 700-800 stocks analyzed daily:
- **Select top 25 stocks** with highest ML predicted returns
- **Weight by inverse volatility** (give more capital to stable stocks, less to volatile ones)
- **Cap each position at 10%** to avoid concentration risk
- **Rebalance every 20 trading days** (roughly monthly)
- **Ignore trades < 1% of position size** to reduce turnover costs

### Risk Management

**Market Regime Filter:**
- Monitors the **SPY 200-day moving average** to detect bear markets
- If market is below 200-day MA → Reduce equity exposure to 50%
- This protects capital during major downturns while staying invested

**Trading Constraints:**
- Maximum 100 orders per rebalance
- Daily notional cap: $1M
- Minimum trade size: $10 (avoid dust)
- Transaction costs modeled at 15-20 basis points

## Production-Grade ML Engineering

This isn't a backtested strategy with hindsight - it's built to survive real-world trading. Here's how:

### 1. Eliminating Lookahead Bias

**The Problem:** Most backtests cheat by using data that wouldn't have been available at decision time.

**Our Solution:**
- All operations use explicit **"as-of date"** control
- Data freshness validation (blocks trading if data is stale)
- Never use `date.today()` - always use `get_last_completed_trading_day()` to account for market hours

### 2. Eliminating Survivorship Bias

**The Problem:** Testing on today's S&P 500 ignores companies that failed and were removed from the index.

**Our Solution:**
- Universe is rebuilt **dynamically** from current constituents at decision time
- No historical cherry-picking of "winners"
- Reflects what an investor could actually buy at that moment

### 3. Leakage-Safe Cross-Validation

**The Problem:** Standard time-series CV can leak future information into the training set.

**Our Solution:**
- **Date-grouped splitting**: Split by unique dates, not by rows (prevents stocks on the same day from appearing in both train and validation)
- **Embargo periods**: Add gaps between train and validation sets matching the prediction horizon (1d model = 1 day gap, 20d model = 20 day gap)
- **Walk-forward validation**: Each fold trains only on past data, validates on future data

### 4. Candidate/Active Model Separation

**The Problem:** Replacing the model every week creates instability and overfitting to recent data.

**Our Solution:**
- **Weekly training** creates a **candidate model** with fresh data
- Candidate must **pass promotion gates** before replacing the active model:
  - Beat pure momentum baseline by **Sharpe ratio margin ≥ 0.2**
  - Maximum drawdown within **5% tolerance**
  - Turnover cap at **80%** (avoid excessive churn)
- Only promote candidate at monthly rebalance (stable deployment schedule)
- Active model stays in production until a better candidate emerges

### 5. Canary Baseline (Shadow Portfolio)

**The Problem:** How do you know if ML is actually adding value or just riding momentum?

**Our Solution:**
- Maintain a **pure momentum baseline** (no ML, just rank by 12-1 momentum)
- Track its performance daily alongside the ML portfolio
- If ML consistently underperforms momentum → the models aren't working
- Provides objective accountability for ML value-add

## Operational Architecture

### GitHub Actions Workflows

The system runs autonomously on GitHub Actions with four scheduled jobs:

1. **Weekly Training** (Sunday 10 PM UTC)
   - Trains all 3 models (1d, 5d, 20d) on latest data
   - Runs walk-forward backtest against momentum baseline
   - Checks promotion gate
   - Approves candidate for next rebalance if gates pass

2. **Daily Signal Generation** (Weekdays, after market close)
   - Generates predictions from all 3 models
   - Computes canary baseline signals
   - Verifies data freshness
   - Sends Discord summary with top picks from each horizon

3. **Monthly Rebalance** (Weekdays, 30 min after market open)
   - Checks if rebalance is due (20-day counter)
   - Validates market is open
   - Checks kill switch
   - Executes portfolio rebalancing with approved model
   - Promotes new model if candidate passed gate
   - Updates state and sends execution report

4. **Daily Health Check** (11 PM UTC)
   - Monitors Alpaca connectivity
   - Checks for anomalies
   - Alerts only if issues detected (silent otherwise)

### State Management

System state (rebalance schedule, active/candidate models, performance) is stored on a separate **`state` branch** in git:
- Prevents polluting main branch with automated commits
- Provides audit trail of all model promotions and rebalances
- Enables rollback if needed

### Safety Controls

**Kill Switch:**
- Repository variable `KILL_SWITCH_ENABLED` blocks all trading instantly
- Useful for market emergencies or system maintenance
- Doesn't stop signal generation (observation only)

**Idempotency:**
- Tracks last rebalance date to prevent double-execution
- Safe to retry failed workflow runs

**Market Guards:**
- Only trades when market is confirmed open
- Blocks stale data (must be current day)

**Order Caps:**
- Maximum 100 orders per rebalance (prevent runaway execution)
- Daily notional cap at $1M
- Minimum $10 per trade (avoid dust orders)

## Daily Operations

### What You See in Discord

**Daily Signals Update** (after market close):
```
📊 MULTI-HORIZON SIGNALS | 2026-01-26
3 XGBoost models (1d/5d/20d horizons)

🔵 1-Day Horizon Top 5:
1. NVDA (+0.0234)
2. MSFT (+0.0198)
...

🟢 5-Day Horizon Top 5:
1. AAPL (+0.0312)
2. GOOGL (+0.0276)
...

🎯 20-Day Horizon Top 5 (PRIMARY):
1. META (+0.0445)
2. AMZN (+0.0389)
...

📈 Canary Top 5 (Pure Momentum):
1. TSLA (+18.3%)
2. NVDA (+15.7%)
...

📊 Since Last Rebalance (11d)
Actual: +2.3% | Canary: +1.8% | SPY: +1.2%
```

**Weekly Training Report** (Sunday):
```
🧠 WEEKLY TRAINING COMPLETE
Candidate Model: multi_horizon_20260126

📊 Cross-Validation (Leakage-Safe)
1d Model IC: 0.0523 ± 0.0312
5d Model IC: 0.0387 ± 0.0245
20d Model IC: 0.0234 ± 0.0198
(20d model used for rebalancing)

📈 Walk-Forward Backtest
CAGR: 23.4%
Sharpe: 1.85
MaxDD: -12.3%
Turnover: 45.2%

🚪 Promotion Gate: ✅ PASSED
Sharpe margin: +0.28 (req: +0.20)
MaxDD diff: +3.2% (tol: 5.0%)
```

**Monthly Rebalance** (every ~20 trading days):
```
✅ MONTHLY REBALANCE EXECUTED
Date: 2026-02-07
Portfolio Value: $123,456

🔄 Model Promotion
multi_horizon_20260126 → ACTIVE

📊 Orders: 18 buys, 15 sells

🟢 Top Buys:
• META: $6,789
• AMZN: $6,234
...

🔴 Top Sells:
• OLD_STOCK_1: $5,432
• OLD_STOCK_2: $4,987
...
```

## Performance Metrics

The system tracks several metrics to ensure it's working:

**Information Coefficient (IC):**
- Spearman rank correlation between predictions and actual returns
- Target: Mean IC > 0.02 for 20-day model (statistically significant edge)
- Measures predictive power across stocks each day

**Sharpe Ratio:**
- Risk-adjusted returns (return per unit of volatility)
- Target: Sharpe > 1.5 (excellent for equity strategies)
- Must beat momentum baseline by 0.2 margin to promote

**Maximum Drawdown:**
- Largest peak-to-trough decline
- Target: MaxDD < 20% (tolerable for retail investors)
- Must be within 5% of baseline to promote

**Turnover:**
- Percentage of portfolio replaced each rebalance
- Target: < 80% (avoid excessive trading costs)
- 1% buffer prevents small rebalancing trades

## Technology Stack

**Data & Execution:**
- Alpaca Markets API (commission-free trading + historical data)
- IEX feed for real-time pricing

**Machine Learning:**
- XGBoost (gradient boosted trees)
- Scikit-learn (preprocessing & validation)
- Pandas/NumPy (data manipulation)

**Infrastructure:**
- Docker (containerized execution for reproducibility)
- GitHub Actions (compute & orchestration with Docker layer caching)
- Git branch (`state`) for state persistence
- Discord webhooks (notifications)

**Languages:**
- Python 3.10+ (entire system)

## Local Development

The system is fully containerized for consistent execution across environments. All scripts run inside Docker containers both locally and in GitHub Actions.

**Quick Start:**
```bash
# Build the Docker image
make build

# Train models locally
make train

# Generate signals
make signals

# Run health check
make health

# Dry-run rebalance (no actual orders)
make rebalance-dry-run
```

**Requirements:**
- Docker installed and running
- `.env` file with API credentials (copy from `.env.example`)

All development workflows use the same Docker image as production, ensuring reproducibility and eliminating "works on my machine" issues.

## Why This Project Matters

This project demonstrates several important engineering and ML capabilities:

1. **Production ML System Design**: Not just a model - a complete system with training, deployment, monitoring, and fallback strategies
2. **Financial Domain Knowledge**: Understanding of market microstructure, trading costs, and academic factor research
3. **Bias Elimination**: Rigorous attention to lookahead bias, survivorship bias, and data leakage
4. **Risk Management**: Multiple safety controls, kill switches, and promotion gates
5. **Automated Operations**: Zero-touch weekly retraining and monthly rebalancing
6. **Accountability**: Canary baseline provides objective measure of ML value
7. **DevOps Best Practices**: Containerized workflows, reproducible builds, infrastructure as code

## Project Status

**✅ Complete and Operational:**
- Multi-horizon model training (1d, 5d, 20d)
- Leakage-safe cross-validation with embargo periods
- Walk-forward backtesting with realistic costs
- Candidate/active model separation with promotion gates
- Daily signal generation from all horizons
- Monthly rebalancing with safety controls
- Canary momentum baseline tracking
- Discord operational notifications
- GitHub Actions automation (4 containerized workflows)
- Dockerized execution environment with layer caching

**Ready for Deployment:**
- Create `state` branch for persistence
- Configure GitHub Secrets (API keys, webhook URL)
- Enable workflows
- Run initial training
- Monitor on paper trading before going live

---

*This is a personal quantitative investment system built for educational purposes and portfolio demonstration. Past performance does not guarantee future results. Trading involves risk of loss.*
