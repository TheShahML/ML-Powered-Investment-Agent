import pandas as pd
from src.config import load_config
from src.data_service import DataService
from src.feature_engineering import FeatureEngineer
from src.strategy_ml import MLStrategy
from loguru import logger
import datetime

def run_training():
    config = load_config()
    data_service = DataService(config)
    feature_engineer = FeatureEngineer(config)
    ml_strategy = MLStrategy(config)
    
    # 1. Fetch historical data for a wide range
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    start_date = (datetime.date.today() - datetime.timedelta(days=365*2)).strftime('%Y-%m-%d') # 2 years
    
    universe = data_service.get_universe()
    df = data_service.get_historical_data(universe, start_date, end_date)
    
    # 2. Compute features
    df_features = feature_engineer.compute_features(df)
    
    # 3. Compute targets (next week's return)
    # We shift the next week's return back to the current date
    # Approx 5 trading days
    df_features['target'] = df_features.groupby(level=1)['close'].shift(-5) / df_features['close'] - 1
    
    # Drop rows with NaN targets (the most recent week)
    df_train = df_features.dropna(subset=['target'])
    
    # 4. Train model
    ml_strategy.train(df_train)
    logger.info("Training complete.")

if __name__ == "__main__":
    run_training()



