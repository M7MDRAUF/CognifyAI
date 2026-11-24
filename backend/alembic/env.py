"""Alembic migration environment (async engine)."""

import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.core.config import read_database_url
from app.domain.models import Base

config = context.config
if config.config_file_name is not None:
    # `disable_existing_loggers` defaults to True, which sets `disabled = True` on every logger
    # that already exists and is not named in `alembic.ini` -- silencing `app.*`, `uvicorn.*` and
    # the vendor SDKs for the rest of the process, with no level or handler change to explain it.
    # Harmless when `alembic` runs as its own short-lived command, but this module is also
    # imported in-process by the integration suite (and would be by any entrypoint that migrates
    # before serving), where it used to take the application's logging down with it.
    # `app.core.configure_logging` also re-enables the loggers it owns; this stops the problem at
    # its source rather than relying on that running afterwards.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    return read_database_url()


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_database_url(), poolclass=pool.NullPool)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
