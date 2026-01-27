import yaml
import os
from dotenv import load_dotenv

load_dotenv()

def load_config():
    config = {}
    config_dir = "config"
    for filename in os.listdir(config_dir):
        if filename.endswith((".yaml", ".yml")):
            with open(os.path.join(config_dir, filename), "r") as f:
                config.update(yaml.safe_all_load(f) if hasattr(yaml, 'safe_all_load') else yaml.safe_load(f))
    
    # Flatten config if nested under top-level keys
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict):
            flat_config.update(v)
        else:
            flat_config[k] = v
            
    # Add environment variables
    flat_config['ALPACA_API_KEY'] = os.getenv('ALPACA_API_KEY')
    flat_config['ALPACA_SECRET_KEY'] = os.getenv('ALPACA_SECRET_KEY')
    flat_config['ALPACA_BASE_URL'] = os.getenv('ALPACA_BASE_URL')
    flat_config['DATABASE_URL'] = os.getenv('DATABASE_URL')
    flat_config['BROKER_MODE'] = os.getenv('BROKER_MODE', 'paper')
    flat_config['I_ACKNOWLEDGE_LIVE_TRADING'] = os.getenv('I_ACKNOWLEDGE_LIVE_TRADING', 'false').lower() == 'true'
    
    return flat_config



