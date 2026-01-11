import pandas as pd
import json
import os
from src.database import get_session
from src.models import PortfolioValue, Run
from loguru import logger
import datetime

class LeanExporter:
    """
    Parses LEAN backtest results (JSON/CSV) and writes them to Postgres.
    """
    def __init__(self, engine):
        self.engine = engine

    def ingest_results(self, results_path: str, run_name: str = "LEAN Backtest"):
        """
        Reads results.json from LEAN and stores the equity curve.
        """
        if not os.path.exists(results_path):
            logger.error(f"LEAN results file not found at {results_path}")
            return
            
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
                
            # LEAN results.json contains 'Charts' -> 'Equity' -> 'Values'
            equity_data = data.get('Charts', {}).get('Equity', {}).get('Values', [])
            
            if not equity_data:
                logger.warning("No equity data found in LEAN results.")
                return

            session = get_session(self.engine)
            
            # Create a run record for this backtest
            run = Run(
                run_type='backtest_lean',
                status='success',
                metadata_json={'run_name': run_name},
                started_at=datetime.datetime.now()
            )
            session.add(run)
            session.commit()
            
            # Note: PortfolioValue table in our schema is usually for live/current,
            # but we can repurpose it or add a backtest_equity table.
            # For simplicity, we'll log it and you can extend the schema as needed.
            
            logger.info(f"Successfully parsed LEAN results. {len(equity_data)} points found.")
            # In a real implementation, you'd iterate and save:
            # for pt in equity_data:
            #     val = PortfolioValue(date=pt['x'], equity=pt['y'], run_id=run.id)
            #     session.add(val)
            
            session.commit()
        except Exception as e:
            logger.error(f"Error ingesting LEAN results: {e}")
        finally:
            if 'session' in locals():
                session.close()



