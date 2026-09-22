"""
Module 4 (data layer) - DB connection/session setup.

Reads DATABASE_URL from .env (SQLite locally, swap to a Postgres URL for
deployment - nothing else in the code needs to change, that's the point
of going through SQLAlchemy instead of raw file/CSV reads).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{PROJECT_ROOT}/acko_platform.db")

# check_same_thread=False is a SQLite-only requirement, needed because
# Streamlit can touch the DB from more than one thread. Harmless no-op
# for Postgres, so this line doesn't need to change when we deploy.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create tables if they don't exist yet. Safe to call every app startup."""
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()