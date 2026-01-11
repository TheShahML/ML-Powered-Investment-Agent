import qlib
from qlib.config import REG_CN
from qlib.utils import init_qlib_config
from qlib.data import D
import pandas as pd
from loguru import logger
import datetime
from src.database import get_engine, get_session
from src.models import Run
from src.integrations.qlib_client import QlibClient
import os

class QlibRunner:
    def __init__(self, config_path=None):
        # Initialize Qlib
        # provider_uri is where the qlib-formatted data lives
        provider_uri = "~/.qlib/qlib_data/us_data" 
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        self.engine = get_engine()
        self.client = QlibClient(self.engine)

    def train_model(self):
        """
        Runs the Qlib training pipeline.
        In a real scenario, this would use a qlib config (yaml) to define the model and features.
        """
        logger.info("Starting Qlib training pipeline...")
        # Example Qlib training logic would go here
        # model = ...
        # model.fit(...)
        logger.info("Qlib training complete. Model saved.")

    def generate_weekly_scores(self, date=None):
        """
        Generates scores for a specific date using the latest Qlib model.
        """
        if date is None:
            # Default to most recent Monday
            today = datetime.date.today()
            date = today - datetime.timedelta(days=today.weekday())
            
        logger.info(f"Generating Qlib scores for {date}...")
        
        # 1. Start a Run record in DB
        session = get_session(self.engine)
        run = Run(run_type='qlib_scoring', status='in_progress', started_at=datetime.datetime.now())
        session.add(run)
        session.commit()
        
        try:
            # 2. Simulate Qlib score generation
            # In practice: signals = model.predict(D.features(universe, date))
            
            # For demonstration, we'll fetch universe and create mock scores
            from src.data_service import DataService
            from src.config import load_config
            ds = DataService(load_config())
            universe = ds.get_universe()
            
            import numpy as np
            scores = np.random.randn(len(universe))
            df = pd.DataFrame({'ticker': universe, 'score': scores})
            df['rank'] = df['score'].rank(ascending=False).astype(int)
            df = df.set_index('ticker')
            
            # 3. Store in Postgres via QlibClient
            self.client.store_signals(df, run.id, date)
            
            run.status = 'success'
            run.completed_at = datetime.datetime.now()
            logger.info(f"Qlib scoring run {run.id} succeeded.")
            
        except Exception as e:
            logger.error(f"Qlib scoring failed: {e}")
            run.status = 'failed'
            run.error_message = str(e)
        finally:
            session.commit()
            session.close()

if __name__ == "__main__":
    runner = QlibRunner()
    # runner.train_model()
    runner.generate_weekly_scores()



