from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Run(Base):
    __tablename__ = 'runs'
    id = Column(Integer, primary_key=True)
    run_type = Column(String) # 'rebalance', 'train', 'heartbeat'
    status = Column(String) # 'success', 'failed', 'in_progress'
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime)
    metadata_json = Column(JSON) # Store config used, model version, etc.
    error_message = Column(String)

class UniverseSnapshot(Base):
    __tablename__ = 'universe_snapshots'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    ticker = Column(String, index=True)
    market_cap = Column(Numeric)
    avg_dollar_volume = Column(Numeric)
    price = Column(Numeric)

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    date = Column(DateTime, index=True)
    ticker = Column(String, index=True)
    score = Column(Float)
    rank = Column(Integer)
    model_version = Column(String)

class TargetWeight(Base):
    __tablename__ = 'target_weights'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    date = Column(DateTime, index=True)
    ticker = Column(String, index=True)
    weight = Column(Float)

class Order(Base):
    __tablename__ = 'orders'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('runs.id'))
    broker_order_id = Column(String, unique=True)
    ticker = Column(String, index=True)
    side = Column(String) # 'buy', 'sell'
    qty = Column(Numeric)
    order_type = Column(String)
    status = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Fill(Base):
    __tablename__ = 'fills'
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'))
    broker_fill_id = Column(String, unique=True)
    timestamp = Column(DateTime)
    qty = Column(Numeric)
    price = Column(Numeric)
    commission = Column(Numeric, default=0)

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    ticker = Column(String, index=True)
    qty = Column(Numeric)
    cost_basis = Column(Numeric)

class PortfolioValue(Base):
    __tablename__ = 'portfolio_values'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    equity = Column(Numeric)
    cash = Column(Numeric)

class BenchmarkValue(Base):
    __tablename__ = 'benchmark_values'
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, index=True)
    ticker = Column(String, index=True)
    price = Column(Numeric)

class Alert(Base):
    __tablename__ = 'alerts'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    level = Column(String) # 'info', 'warning', 'critical'
    message = Column(String)
    run_id = Column(Integer, ForeignKey('runs.id'), nullable=True)



