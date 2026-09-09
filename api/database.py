from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from src.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DecisionLog(Base):
    __tablename__ = "decision_logs"
    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True)
    requested_by = Column(String)
    risk_score = Column(Float)
    causal_effect_on_default = Column(Float)
    predicted_spend_lift_pct = Column(Float)
    decision = Column(String)  # "approved" or "denied"
    top_reason = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()