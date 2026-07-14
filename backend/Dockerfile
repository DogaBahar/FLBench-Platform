FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including gcc for some python packages)
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x docker-entrypoint.sh

# Runs Alembic migrations (see scripts/migrate.py) before either process below starts.
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command (will be overridden by docker-compose for celery)
CMD ["python", "main.py"]