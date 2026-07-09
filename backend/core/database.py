from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from core.config import config
from domain.models import Base

# Create the database engine
engine = create_engine(config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)

# Create a thread-safe session factory
session_factory = sessionmaker(bind=engine)
db_session = scoped_session(session_factory)

def init_db():
    """
    Creates all tables based on the models defined in Base.
    Run this once during application startup.
    """
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")