from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, autoincrement=True)

    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

    signal = Column(Float, nullable=True)
    meta = Column(JSON, nullable=True)
    is_latest = Column(Boolean, default=False)

    ticker = Column(String, nullable=False, index=True)
    interval = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False)
    strategy = Column(String, nullable=True)

    __table_args__ = (
        Index('ix_ticker_interval_time', "ticker", "interval", "timestamp"),
    )
