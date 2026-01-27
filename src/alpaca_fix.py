"""
Workaround for alpaca-trade-api alpha_vantage import issue.
This module patches the import before alpaca_trade_api is imported.
"""
import sys

# Create a mock alpha_vantage.sectorperformance module if it doesn't exist
try:
    from alpha_vantage import sectorperformance
except ImportError:
    # Create mock module
    import types
    sectorperformance_module = types.ModuleType('alpha_vantage.sectorperformance')
    sys.modules['alpha_vantage.sectorperformance'] = sectorperformance_module
    
    # Add a dummy REST class if needed
    class MockREST:
        pass
    sectorperformance_module.REST = MockREST

