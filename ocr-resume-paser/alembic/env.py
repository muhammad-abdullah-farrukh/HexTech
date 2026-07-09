"""Alembic environment: wires migrations to the app's metadata and DATABASE_URL.

The URL is read from the environment (via .env) rather than alembic.ini so the
same value drives both the app and migrations.
"""

from __future__ import annotations

import os

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine

from resume_parser.db.models import Base

# Register Ontogen's tables (Part 2) on the shared Base.metadata so autogenerate
# sees the whole schema. Guarded: the parser's own migrations must still run if
# the sibling Ontogen repo isn't importable. Hand-written migrations don't need
# this, but keeping metadata complete avoids a future autogenerate dropping them.
try:
    import sys
    from pathlib import Path

    _ontogen_root = Path(__file__).resolve().parents[2] / "ontogen"
    if _ontogen_root.is_dir() and str(_ontogen_root) not in sys.path:
        sys.path.insert(0, str(_ontogen_root))
    import db.models as _ontogen_models  # noqa: F401  (registers tables on Base)
except Exception:  # pragma: no cover - ontogen is optional for parser migrations
    pass

load_dotenv(override=False)

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env (or export it) so "
            "Alembic knows which database to migrate."
        )
    return url


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    engine = create_engine(_database_url())
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
