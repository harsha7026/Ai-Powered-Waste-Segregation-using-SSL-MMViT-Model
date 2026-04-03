"""Database connection and session management using SQLAlchemy."""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
try:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,   # Recycle connections after 1 hour
        echo=False           # Set to True for SQL query logging
    )
except Exception as e:
    logger.error(f"Failed to create database engine: {e}")
    engine = None

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None

# Base class for models
Base = declarative_base()


class Prediction(Base):
    """Model for storing waste classification predictions."""
    
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    predicted_class = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Optional fields for future expansion
    image_path = Column(String(255), nullable=True)
    user_id = Column(String(100), nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "image_path": self.image_path,
            "user_id": self.user_id
        }


def init_db():
    """Initialize database tables."""
    if engine is None:
        logger.error("Database engine not initialized. Cannot create tables.")
        return False
    
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        return False


def get_db():
    """Get database session (dependency injection for FastAPI)."""
    if SessionLocal is None:
        raise RuntimeError("Database not configured properly")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_connection():
    """Check if database connection is working."""
    if engine is None:
        return False
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False
