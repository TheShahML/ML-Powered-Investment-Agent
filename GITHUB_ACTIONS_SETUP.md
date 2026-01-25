# GitHub Actions Setup Guide

This guide explains how to set up the automated trading bot on GitHub Actions for **free**.

## Overview

The bot runs on GitHub Actions with the following schedule:
- **Daily (9:00 AM ET)**: Generate ML signals before market opens + Discord notification
- **Daily (9:35 AM ET)**: Check if it's a rebalance day (every 20 trading days)
- **Weekly (Saturday)**: Portfolio update summary to Discord
- **Weekly (Sunday)**: Retrain the ML model with latest data

## Quick Start

### 1. Fork/Clone This Repository

```bash
git clone https://github.com/YOUR_USERNAME/investment-bot.git
cd investment-bot
```

### 2. Set Up GitHub Secrets

Go to your repo → Settings → Secrets and variables → Actions → New repository secret

Add these secrets:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `ALPACA_API_KEY` | Your Alpaca API key | `PKXXXXXXXX` |
| `ALPACA_SECRET_KEY` | Your Alpaca secret key | `XXXXXXXXXXXXXXXX` |
| `ALPACA_BASE_URL` | Paper or live URL | `https://paper-api.alpaca.markets` |
| `BROKER_MODE` | `paper` or `live` | `paper` |
| `I_ACKNOWLEDGE_LIVE_TRADING` | Set to `true` for live | `false` |
| `DISCORD_WEBHOOK_URL` | Discord webhook for notifications | `https://discord.com/api/webhooks/...` |

### 3. Get Alpaca API Keys

1. Sign up at [Alpaca](https://alpaca.markets/)
2. Go to Paper Trading → API Keys
3. Generate new keys
4. For paper trading, use: `https://paper-api.alpaca.markets`
5. For live trading, use: `https://api.alpaca.markets`

### 4. Set Up Discord Notifications

Get notified about daily signals, rebalances, and portfolio updates:

1. Open Discord and go to the server where you want notifications
2. Right-click on a channel → Edit Channel → Integrations → Webhooks
3. Click "New Webhook"
4. Name it "Investment Bot" and copy the webhook URL
5. Add it as `DISCORD_WEBHOOK_URL` in your GitHub secrets

**What you'll receive:**
- 📊 **Daily signals** - Top 15 stock picks after market close
- ✅ **Rebalance notifications** - Buy/sell orders when executed
- 📋 **Weekly portfolio updates** - Performance summary every Saturday
- 🧠 **Model training updates** - When weekly retraining completes

### 5. Train the Model (First Time)

Before the bot can generate signals, you need to train the model:

**Option A: Run manually via GitHub Actions**
1. Go to Actions → Weekly Model Training
2. Click "Run workflow"
3. Wait for completion (~10-15 minutes)

**Option B: Run locally**
```bash
pip install -r requirements.txt
python scripts/train_stacking_model.py
git add models/
git commit -m "Initial model training"
git push
```

### 5. Enable GitHub Actions

1. Go to your repo → Actions
2. Click "I understand my workflows, go ahead and enable them"
3. The workflows will now run on schedule

## Workflows

### Daily Signal Generation
**File**: `.github/workflows/daily-signals.yml`
**Schedule**: 5:00 PM ET (after market close)
**What it does**:
- Fetches latest market data from Alpaca
- Computes ML features (momentum, RSI, MACD, etc.)
- Generates stock rankings using the stacking ensemble
- Saves signals to `data/signals/`

### Monthly Rebalance
**File**: `.github/workflows/monthly-rebalance.yml`
**Schedule**: 9:35 AM ET (5 min after market open)
**What it does**:
- Checks if it's a rebalance day (every 20 trading days)
- If yes: executes trades via Alpaca API
- If no: increments day counter
- Logs all orders to `data/logs/`

### Weekly Model Training
**File**: `.github/workflows/weekly-train.yml`
**Schedule**: Sunday 6:00 PM ET
**What it does**:
- Fetches 2 years of historical data
- Trains the stacking ensemble (XGBoost, LightGBM, RF, etc.)
- Saves models to `models/`

## Configuration

### Strategy Settings
Edit `config/strategy.yaml`:
```yaml
strategy:
  name: "ml_momentum_ranking"
  n_holdings: 15          # Number of stocks to hold
  rebalance_day: 1        # Not used with 20-day cycle
  benchmark_tickers: ["SPY", "QQQ", "VTI"]
```

### Risk Settings
Edit `config/risk.yaml`:
```yaml
risk:
  max_position_weight: 0.12    # Max 12% per stock
  min_trade_notional: 10       # Min $10 per trade
  max_drawdown: 0.20           # 20% max drawdown stop
```

### Rebalancing Frequency
The bot rebalances every 20 trading days (~monthly). To change this:

Edit `scripts/execute_rebalance.py`:
```python
should_rebalance = days >= 20  # Change 20 to your preferred frequency
```

## Manual Operations

### Force a Rebalance
1. Go to Actions → Monthly Rebalance
2. Click "Run workflow"
3. Check "Force rebalance regardless of day count"
4. Run

### Dry Run (Test Without Trading)
1. Go to Actions → Monthly Rebalance
2. Click "Run workflow"
3. Check "Run without placing actual orders"
4. Run

### Check Current State
Look at `data/state/rebalance_state.json`:
```json
{
  "days_since_rebalance": 15,
  "last_rebalance": "2024-01-15T09:35:00",
  "total_rebalances": 3
}
```

## Safety Features

1. **Paper Trading Default**: `BROKER_MODE=paper` by default
2. **Live Trading Acknowledgment**: Must set `I_ACKNOWLEDGE_LIVE_TRADING=true`
3. **Dry Run Mode**: Test without placing orders
4. **Day Counter**: Prevents accidental over-trading
5. **Market Hours Check**: Only trades when market is open

## Switching to Live Trading

⚠️ **WARNING**: Only do this after thorough testing with paper trading!

1. Get live API keys from Alpaca (requires approval)
2. Update secrets:
   - `ALPACA_BASE_URL` → `https://api.alpaca.markets`
   - `BROKER_MODE` → `live`
   - `I_ACKNOWLEDGE_LIVE_TRADING` → `true`

## Monitoring

### View Logs
- Go to Actions → Select a workflow run → Click on job → View logs

### Check Performance
Look at `data/state/portfolio_history.json` for equity curve data.

### View Recent Signals
Check `data/signals/latest_signals.csv` for current stock rankings.

## Troubleshooting

### "No signals found"
- Run the Daily Signal Generation workflow manually
- Check if the market was open that day

### "Model not trained"
- Run the Weekly Model Training workflow manually

### Orders not executing
- Check if market is open (9:30 AM - 4:00 PM ET)
- Verify Alpaca API keys are correct
- Check if `BROKER_MODE` matches your intent

### Rate limits
- Alpaca has rate limits on API calls
- The bot includes delays to avoid hitting limits

## Cost

**GitHub Actions**: FREE for public repos (2,000 minutes/month for private)
**Alpaca**: FREE for paper and live trading (commission-free)

## Comparison: Your Friend's Setup

| Feature | Your Friend | This Bot |
|---------|-------------|----------|
| ML Model | XGBoost | Stacking Ensemble (XGBoost + LightGBM + RF + GB + ET) |
| Features | Basic | Enhanced (RSI, MACD, Bollinger, OBV, etc.) |
| Rebalancing | ~Monthly | Every 20 trading days |
| Execution | Manual/Email | Automated via Alpaca API |
| Infrastructure | GitHub Actions | GitHub Actions |
| Cost | Free | Free |
