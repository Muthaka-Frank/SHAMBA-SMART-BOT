"""
Database — SQLAlchemy ORM models + helper functions for Shamba-Smart.
Uses PostgreSQL (configured via DATABASE_URL in .env)
"""
import os
import logging
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/shamba_smart"
)

ENGINE = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
Session = sessionmaker(bind=ENGINE)
Base = declarative_base()


# ─── ORM Models ───────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    phone = Column(String(30), unique=True, nullable=False)
    preferred_language = Column(String(5), default="sw")   # 'sw' or 'ki'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class SessionLog(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    transcript = Column(Text)
    intent = Column(String(30))
    response = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class MarketPrice(Base):
    __tablename__ = "market_prices"
    id = Column(Integer, primary_key=True)
    commodity = Column(String(100))
    market = Column(String(100))
    price_ksh = Column(Float)
    unit = Column(String(20), default="kg")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QueryLog(Base):
    __tablename__ = "query_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    transcript = Column(Text)
    intent = Column(String(30))
    rag_chunks_used = Column(Integer, default=0)
    response = Column(Text)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─── Helper Functions ──────────────────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(ENGINE)
    logger.info(f"✅ PostgreSQL database initialized.")


def get_or_create_user(phone: str, lang: str = "sw") -> User:
    with Session() as session:
        user = session.query(User).filter_by(phone=phone).first()
        if not user:
            user = User(phone=phone, preferred_language=lang)
            session.add(user)
            session.commit()
            logger.info(f"New user registered: {phone}")
        else:
            user.last_seen = datetime.now(timezone.utc)
            user.preferred_language = lang
            session.commit()
        session.refresh(user)
        return user


def log_query(user_id: int, transcript: str, intent: str, response: str, chunks: int = 0):
    with Session() as session:
        entry = QueryLog(
            user_id=user_id,
            transcript=transcript,
            intent=intent,
            rag_chunks_used=chunks,
            response=response,
        )
        session.add(entry)
        session.commit()
