from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Base
import os

def get_engine(db_url=None):
    if not db_url:
        db_url = os.getenv('DATABASE_URL')
    return create_engine(db_url)

def init_db(engine):
    Base.metadata.create_all(engine)

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()



