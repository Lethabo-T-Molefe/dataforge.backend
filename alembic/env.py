import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from dataforge_backend.core.config import settings
from dataforge_backend.core.database import APP_SCHEMA, Base
from dataforge_backend.infrastructure.db import models  # noqa: F401  (registers the tables)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Use the app's DATABASE_URL (from .env) instead of a hardcoded URL in alembic.ini.
# '%' is escaped because the value goes through configparser interpolation.
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# Import model modules before this point so their tables register on Base.metadata.
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_schemas=True,
        version_table_schema=APP_SCHEMA,
        include_name=_include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def _include_name(name: str | None, type_: str, parent_names: object) -> bool:
    """Only the app schema is managed by migrations; bronze, silver and gold hold member data."""
    if type_ == "schema":
        return name == APP_SCHEMA
    return True


def do_run_migrations(connection: Connection) -> None:
    # The version table lives in the app schema, which must exist before Alembic writes to it.
    connection.exec_driver_sql(f"CREATE SCHEMA IF NOT EXISTS {APP_SCHEMA}")
    connection.commit()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        version_table_schema=APP_SCHEMA,
        include_name=_include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
