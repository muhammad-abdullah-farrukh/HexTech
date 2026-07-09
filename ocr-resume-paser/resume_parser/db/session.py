"""Engine + session-factory construction for the resume database.

`make_session_factory` returns a `sessionmaker`, not a live session — each ingest
call opens and closes its own session. When batch/parallel processing is added
later, the factory must be constructed *fresh inside each worker process*, after
the fork: SQLAlchemy engines are not fork-safe, so a factory built in the parent
and inherited by forked workers would share one TCP socket to Postgres and
corrupt concurrent queries. The `pool_size` cap keeps `workers × pool_size` under
Postgres's `max_connections` once that day comes.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def make_session_factory(database_url: str) -> sessionmaker:
    """Build an engine for `database_url` and return a `sessionmaker` bound to it."""
    engine = create_engine(database_url, pool_pre_ping=True, pool_size=5)
    return sessionmaker(bind=engine)
