import pandas as pd
from sqlalchemy import desc
from src.models import Signal, Run
from src.database import get_session
from loguru import logger

class QlibClient:
    """
    Client for fetching Qlib-generated signals from Postgres.
    """
    def __init__(self, engine):
        self.engine = engine

    def get_latest_signals(self, date=None) -> pd.DataFrame:
        """
        Fetches the most recent signals from the signals table.
        """
        session = get_session(self.engine)
        try:
            # Find the latest successful rebalance or scoring run
            query = session.query(Signal)
            if date:
                query = query.filter(Signal.date == date)
            else:
                # Get the latest date available in signals
                latest_date_subquery = session.query(Signal.date).order_by(desc(Signal.date)).limit(1).scalar()
                if not latest_date_subquery:
                    logger.warning("No signals found in database.")
                    return pd.DataFrame()
                query = query.filter(Signal.date == latest_date_subquery)
            
            signals_list = query.all()
            
            if not signals_list:
                return pd.DataFrame()
                
            data = []
            for s in signals_list:
                data.append({
                    'ticker': s.ticker,
                    'score': s.score,
                    'rank': s.rank
                })
            
            df = pd.DataFrame(data).set_index('ticker')
            return df.sort_values('rank')
            
        except Exception as e:
            logger.error(f"Error fetching signals from DB: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def store_signals(self, signals_df: pd.DataFrame, run_id: int, date):
        """
        Stores Qlib-generated signals into Postgres.
        signals_df: Index=ticker, Columns=[score, rank]
        """
        session = get_session(self.engine)
        try:
            for ticker, row in signals_df.iterrows():
                signal = Signal(
                    run_id=run_id,
                    date=date,
                    ticker=ticker,
                    score=float(row['score']),
                    rank=int(row['rank']),
                    model_version="qlib_v1"
                )
                session.add(signal)
            session.commit()
            logger.info(f"Stored {len(signals_df)} signals for date {date} in DB.")
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing signals in DB: {e}")
            raise
        finally:
            session.close()



