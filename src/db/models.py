"""
Module 4 (data layer) - SQLAlchemy table definitions.

Two tables, matching what Module 1 (chatbot) and Module 2 (quote predictor)
actually produce. No `users` or `claims` tables - those belonged to the
deferred Module 3/5 scope and have nothing to join against right now.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    response_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Quotation(Base):
    __tablename__ = "quotations"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Inputs the customer provided
    vehicle_type = Column(String(10), nullable=False)      # car / bike
    vehicle_make = Column(String(50), nullable=False)
    segment = Column(String(50), nullable=False)
    fuel_type = Column(String(20), nullable=False)
    policy_type = Column(String(30), nullable=False)
    customer_age = Column(Integer, nullable=False)
    city_tier = Column(Integer, nullable=False)
    city_risk_score = Column(Float, nullable=False)
    manufacturing_year = Column(Integer, nullable=False)
    vehicle_age_years = Column(Float, nullable=False)
    engine_cc = Column(Integer, nullable=False)
    idv = Column(Float, nullable=False)
    ncb_percent = Column(Float, nullable=False)
    claim_history_count = Column(Integer, nullable=False)
    num_addons = Column(Integer, nullable=False)

    # Output
    predicted_premium = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)