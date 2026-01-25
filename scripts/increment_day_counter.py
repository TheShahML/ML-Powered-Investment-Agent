#!/usr/bin/env python3
"""
Simple script to increment the day counter when no rebalance occurs.
Called by GitHub Actions on trading days between rebalances.
"""

import os
import sys
from loguru import logger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.file_storage import FileStorage


def main():
    storage = FileStorage()
    state = storage.get_rebalance_state()

    old_count = state.get('days_since_rebalance', 0)
    storage.increment_day_counter()

    logger.info(f"Day counter incremented: {old_count} -> {old_count + 1}")
    logger.info(f"Rebalance in {20 - (old_count + 1)} trading days")


if __name__ == "__main__":
    main()
