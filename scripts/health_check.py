#!/usr/bin/env python3
"""Daily health check - alerts on anomalies only."""
import os
import sys
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.trading_calendar import TradingCalendar
from src.state_manager import StateManager
from src.discord_prod import DiscordProductionNotifier
from src.execution_safe import check_kill_switch
import alpaca_trade_api as tradeapi


def main():
    logger.info("=" * 60)
    logger.info("DAILY HEALTH CHECK")
    logger.info("=" * 60)

    config = load_config()
    issues = []

    # Check Alpaca connectivity
    try:
        api = tradeapi.REST(
            config['ALPACA_API_KEY'],
            config['ALPACA_SECRET_KEY'],
            config['ALPACA_BASE_URL']
        )
        api.get_clock()
        logger.info("✅ Alpaca connectivity OK")
    except Exception as e:
        issues.append(f"Alpaca connectivity FAILED: {e}")
        logger.error(issues[-1])

    # Check kill switch
    kill_enabled, kill_reason = check_kill_switch()
    if kill_enabled:
        issues.append(f"Kill switch ENABLED: {kill_reason}")
        logger.warning(issues[-1])

    # Check state
    try:
        state_manager = StateManager(state_file_path="state-repo/latest_state.json")
        state = state_manager.load_state()

        # Check if active model exists
        if not state.get('models', {}).get('active_model'):
            issues.append("No active model in state")

        logger.info("✅ State loaded OK")
    except Exception as e:
        issues.append(f"State loading FAILED: {e}")
        logger.error(issues[-1])

    # Send alert if issues found
    if issues:
        logger.warning(f"Found {len(issues)} issues")
        discord = DiscordProductionNotifier()
        discord.send_health_alert(issues, severity="warning")
    else:
        logger.info("✅ All checks passed")

    logger.info("=" * 60)
    logger.info("HEALTH CHECK COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
