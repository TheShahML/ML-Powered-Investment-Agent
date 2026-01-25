"""
File-based storage for GitHub Actions deployment.
Replaces PostgreSQL with JSON/CSV files that can be committed to the repo.
"""

import json
import os
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from loguru import logger


class FileStorage:
    """
    Simple file-based storage for running on GitHub Actions.
    Data is stored in the data/ directory and committed back to the repo.
    """

    def __init__(self, base_dir: str = 'data'):
        self.base_dir = base_dir
        self.signals_dir = os.path.join(base_dir, 'signals')
        self.state_dir = os.path.join(base_dir, 'state')
        self.positions_dir = os.path.join(base_dir, 'positions')
        self.logs_dir = os.path.join(base_dir, 'logs')

        # Create directories
        for d in [self.signals_dir, self.state_dir, self.positions_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

    # ===== STATE MANAGEMENT =====

    def get_rebalance_state(self) -> Dict:
        """Get the current rebalance state."""
        state_file = os.path.join(self.state_dir, 'rebalance_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                return json.load(f)
        return {
            'days_since_rebalance': 20,  # Start ready to rebalance
            'last_rebalance': None,
            'last_signal_date': None,
            'total_rebalances': 0
        }

    def save_rebalance_state(self, state: Dict):
        """Save rebalance state."""
        state_file = os.path.join(self.state_dir, 'rebalance_state.json')
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def increment_day_counter(self):
        """Increment the days since last rebalance counter."""
        state = self.get_rebalance_state()
        state['days_since_rebalance'] = state.get('days_since_rebalance', 0) + 1
        self.save_rebalance_state(state)
        logger.info(f"Days since rebalance: {state['days_since_rebalance']}")

    def reset_day_counter(self):
        """Reset the day counter after a rebalance."""
        state = self.get_rebalance_state()
        state['days_since_rebalance'] = 0
        state['last_rebalance'] = datetime.now().isoformat()
        state['total_rebalances'] = state.get('total_rebalances', 0) + 1
        self.save_rebalance_state(state)

    # ===== SIGNALS MANAGEMENT =====

    def save_signals(self, signals: pd.DataFrame, signal_date: date):
        """Save daily signals to CSV."""
        date_str = signal_date.strftime('%Y-%m-%d')
        filepath = os.path.join(self.signals_dir, f'signals_{date_str}.csv')
        signals.to_csv(filepath)
        logger.info(f"Signals saved to {filepath}")

        # Update state
        state = self.get_rebalance_state()
        state['last_signal_date'] = date_str
        self.save_rebalance_state(state)

        # Also save as latest
        latest_path = os.path.join(self.signals_dir, 'latest_signals.csv')
        signals.to_csv(latest_path)

    def get_latest_signals(self) -> Optional[pd.DataFrame]:
        """Get the most recent signals."""
        latest_path = os.path.join(self.signals_dir, 'latest_signals.csv')
        if os.path.exists(latest_path):
            return pd.read_csv(latest_path, index_col=0)
        return None

    def get_signals_by_date(self, signal_date: date) -> Optional[pd.DataFrame]:
        """Get signals for a specific date."""
        date_str = signal_date.strftime('%Y-%m-%d')
        filepath = os.path.join(self.signals_dir, f'signals_{date_str}.csv')
        if os.path.exists(filepath):
            return pd.read_csv(filepath, index_col=0)
        return None

    # ===== POSITIONS MANAGEMENT =====

    def save_positions(self, positions: List[Dict], timestamp: datetime = None):
        """Save current positions snapshot."""
        if timestamp is None:
            timestamp = datetime.now()

        date_str = timestamp.strftime('%Y-%m-%d')
        filepath = os.path.join(self.positions_dir, f'positions_{date_str}.json')

        with open(filepath, 'w') as f:
            json.dump({
                'timestamp': timestamp.isoformat(),
                'positions': positions
            }, f, indent=2)

        # Also save as latest
        latest_path = os.path.join(self.positions_dir, 'latest_positions.json')
        with open(latest_path, 'w') as f:
            json.dump({
                'timestamp': timestamp.isoformat(),
                'positions': positions
            }, f, indent=2)

        logger.info(f"Positions saved: {len(positions)} holdings")

    def get_latest_positions(self) -> List[Dict]:
        """Get the most recent positions."""
        latest_path = os.path.join(self.positions_dir, 'latest_positions.json')
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                data = json.load(f)
                return data.get('positions', [])
        return []

    # ===== PORTFOLIO VALUE TRACKING =====

    def save_portfolio_value(self, equity: float, cash: float, timestamp: datetime = None):
        """Append portfolio value to history."""
        if timestamp is None:
            timestamp = datetime.now()

        history_file = os.path.join(self.state_dir, 'portfolio_history.json')

        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []

        history.append({
            'timestamp': timestamp.isoformat(),
            'equity': equity,
            'cash': cash
        })

        # Keep last 365 days
        if len(history) > 365:
            history = history[-365:]

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)

    def get_portfolio_history(self) -> List[Dict]:
        """Get portfolio value history."""
        history_file = os.path.join(self.state_dir, 'portfolio_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                return json.load(f)
        return []

    # ===== LOGGING =====

    def log_execution(self, log_entry: Dict):
        """Log an execution event."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = os.path.join(self.logs_dir, f'execution_{date_str}.json')

        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        log_entry['timestamp'] = datetime.now().isoformat()
        logs.append(log_entry)

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def log_orders(self, orders: List[Dict]):
        """Log orders placed."""
        self.log_execution({
            'type': 'orders',
            'count': len(orders),
            'orders': orders
        })

    # ===== METRICS =====

    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics from history."""
        history = self.get_portfolio_history()
        if len(history) < 2:
            return {}

        equity_values = [h['equity'] for h in history]
        initial = equity_values[0]
        current = equity_values[-1]

        total_return = (current / initial - 1) * 100

        # Calculate max drawdown
        peak = equity_values[0]
        max_drawdown = 0
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return {
            'total_return_pct': round(total_return, 2),
            'max_drawdown_pct': round(max_drawdown * 100, 2),
            'initial_equity': initial,
            'current_equity': current,
            'days_tracked': len(history)
        }
