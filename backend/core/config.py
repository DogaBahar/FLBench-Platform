import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    FLASK_APP = os.getenv("FLASK_APP", "main.py")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    
    # FIX: Explicitly ensure the worker falls back to the exact Docker Compose database service name
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@fl_benchmark_platform-db-1:5432/fl_benchmark" 
        if os.path.exists("/.dockerenv") else 
        "postgresql://postgres:postgres@localhost:5432/fl_benchmark"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://fl_benchmark_platform-redis-1:6379/0" if os.path.exists("/.dockerenv") else "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://fl_benchmark_platform-redis-1:6379/0" if os.path.exists("/.dockerenv") else "redis://localhost:6379/0")
    
    SHARED_RUN_DIR = os.getenv("SHARED_RUN_DIR", "/tmp/fl_benchmark_runs")
    DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "/tmp/fl_benchmark_data")
    USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"

config = Config()