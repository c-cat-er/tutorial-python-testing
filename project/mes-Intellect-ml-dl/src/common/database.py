import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/gold/yield.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# 補齊 ORM Table Model 定義
class YieldPredictionRecord(Base):
    __tablename__ = "yield_predictions"

    id = Column(DateTime, primary_key=True, default=datetime.now)
    predicted_yield = Column(Float, nullable=False)
    status_alert = Column(String, nullable=False)


# 確保執行時會自動建立資料表
Base.metadata.create_all(bind=engine)
