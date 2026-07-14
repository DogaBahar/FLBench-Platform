"""
Runs at container startup (see docker-entrypoint.sh), before the actual app
process. Reconciles this platform's move from Base.metadata.create_all() to
Alembic-managed migrations without touching an existing dev database:

- Fresh database (no benchmark_runs table): run migrations normally.
- Pre-Alembic database (benchmark_runs exists, but no alembic_version table
  -- i.e. it was created by the old create_all() path): its schema already
  matches revision head, so it's stamped as head in place rather than
  replaying DDL that would fail on tables that already exist.
- Anything else (alembic_version already present): run migrations normally;
  this is a no-op if already at head.
"""
import os

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from core.database import engine

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    alembic_cfg = Config(os.path.join(BACKEND_DIR, "alembic.ini"))
    tables = inspect(engine).get_table_names()

    if "benchmark_runs" in tables and "alembic_version" not in tables:
        print("[migrate] Adopting pre-Alembic schema: stamping as head without replaying DDL.")
        command.stamp(alembic_cfg, "head")
    else:
        print("[migrate] Running Alembic migrations to head.")
        command.upgrade(alembic_cfg, "head")


if __name__ == "__main__":
    main()
