"""State persistence on separate git branch."""
import json
import os
import subprocess
from datetime import date, datetime
from loguru import logger
from typing import Dict, Optional


class StateManager:
    """Manage state/latest_state.json on state branch."""

    def __init__(self, state_file_path: str = "state/latest_state.json"):
        self.state_file = state_file_path

    def load_state(self) -> Dict:
        """Load state from file."""
        if not os.path.exists(self.state_file):
            logger.warning(f"State file not found: {self.state_file}, returning empty state")
            return self._empty_state()

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded state as-of {state.get('as_of_date')}")
            return state
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return self._empty_state()

    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_state(self, state: Dict):
        """Save state to file."""
        try:
            # Handle case where state_file is just a filename (no directory)
            state_dir = os.path.dirname(self.state_file)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            state['last_updated_utc'] = datetime.utcnow().isoformat() + 'Z'

            # Convert NumPy types to Python native types
            state = self._convert_numpy_types(state)

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Saved state: {self.state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def _empty_state(self) -> Dict:
        """Return empty state template."""
        return {
            'as_of_date': date.today().isoformat(),
            'last_updated_utc': datetime.utcnow().isoformat() + 'Z',
            'rebalance': {
                'last_rebalance_date': None,
                'next_rebalance_date': None,
                'days_since_rebalance': 0,
                'days_until_rebalance': 20
            },
            'models': {
                'active_model': None,
                'candidate_model': None
            },
            'holdings': {'equities': [], 'crypto': [], 'cash': 0, 'total_equity': 0},
            'performance': {},
            'data_freshness': {'is_fresh': True},
            'kill_switch': {'enabled': False},
            'last_workflow_runs': {},
            'execution': {
                'last_successful_trade_date': None,
                'last_run': {}
            }
        }

    def promote_candidate_to_active(self):
        """Promote approved candidate to active model."""
        state = self.load_state()

        if not state.get('models', {}).get('candidate_model', {}).get('approved_for_next_rebalance'):
            logger.warning("No approved candidate to promote")
            return

        state['models']['active_model'] = state['models']['candidate_model'].copy()
        state['models']['candidate_model'] = None

        logger.info(f"Promoted candidate to active: {state['models']['active_model']['version']}")
        self.save_state(state)

    def update_rebalance_schedule(self, last_rebalance: date, rebalance_freq: int = 20):
        """Update rebalance tracking after execution."""
        state = self.load_state()

        state['rebalance']['last_rebalance_date'] = last_rebalance.isoformat()
        state['rebalance']['days_since_rebalance'] = 0

        # Calculate next (approximate, ignoring weekends)
        from datetime import timedelta
        next_date = last_rebalance + timedelta(days=rebalance_freq)
        state['rebalance']['next_rebalance_date'] = next_date.isoformat()
        state['rebalance']['days_until_rebalance'] = rebalance_freq

        logger.info(f"Updated rebalance schedule: next on {next_date}")
        self.save_state(state)

    def increment_day_counter(self):
        """Increment days since last rebalance (call daily)."""
        state = self.load_state()
        
        # Only increment if a rebalance has actually occurred
        last_rebalance_date = state.get('rebalance', {}).get('last_rebalance_date')
        if last_rebalance_date is None:
            logger.debug("No previous rebalance - skipping day counter increment")
            return

        state['rebalance']['days_since_rebalance'] += 1
        state['rebalance']['days_until_rebalance'] = max(
            0, 20 - state['rebalance']['days_since_rebalance']
        )

        self.save_state(state)

    def check_rebalance_due(self, threshold: int = 20) -> bool:
        """Check if rebalance is due."""
        state = self.load_state()
        last_rebalance_date = state.get('rebalance', {}).get('last_rebalance_date')
        
        # If never rebalanced, allow first investment immediately
        if last_rebalance_date is None:
            logger.info("No previous rebalance found - allowing initial investment")
            return True
        
        days_since = state.get('rebalance', {}).get('days_since_rebalance', 0)
        return days_since >= threshold

    def check_already_rebalanced(self, target_date: date) -> bool:
        """Check if already rebalanced for this date (idempotency)."""
        state = self.load_state()
        last = state.get('rebalance', {}).get('last_rebalance_date')

        if last and last == target_date.isoformat():
            logger.warning(f"Already rebalanced on {target_date}")
            return True
        return False
