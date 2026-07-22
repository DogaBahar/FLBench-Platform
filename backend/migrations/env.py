from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

from core.config import config as app_config
from domain.models import Base

# Alembic Config object, provides access to values within alembic.ini.
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Reuse the app's own DB URL logic (core/config.py already handles the
# in-Docker vs. local hostname switch) instead of duplicating it in alembic.ini.
config.set_main_option("sqlalchemy.url", app_config.SQLALCHEMY_DATABASE_URI)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
