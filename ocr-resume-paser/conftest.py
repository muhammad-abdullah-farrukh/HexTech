"""Ensure the repo root is importable so `import resume_parser` works under pytest.

Also provides the Postgres fixtures for tests/test_db_ingest.py. The psycopg2 /
SQLAlchemy imports live inside the fixtures so the rest of the suite — which needs
no external services — is unaffected when the DB packages or a server are absent.
"""

import os
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

TEST_DB_ENV = "TEST_DATABASE_URL"


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """The URL of the test database, from TEST_DATABASE_URL (skips if unset)."""
    from dotenv import load_dotenv

    load_dotenv(override=False)
    url = os.environ.get(TEST_DB_ENV)
    if not url:
        pytest.skip(f"{TEST_DB_ENV} not set; skipping database tests.")
    return url


@pytest.fixture(scope="session")
def _ensure_test_db(test_db_url: str) -> str:
    """Create the test database if it does not exist; skip-with-warning if no server.

    Connects to the default `postgres` database with autocommit and issues an
    idempotent CREATE DATABASE, so repeated runs (and a non-freshly-initialized
    data volume) both work without a compose init script.
    """
    import psycopg2
    from sqlalchemy.engine import make_url

    url = make_url(test_db_url)
    try:
        conn = psycopg2.connect(
            host=url.host,
            port=url.port,
            user=url.username,
            password=url.password,
            dbname="postgres",
        )
    except psycopg2.OperationalError as exc:
        warnings.warn(
            f"Postgres not reachable at {url.host}:{url.port} ({exc}); "
            "skipping database tests. Is `docker compose up` running?"
        )
        pytest.skip("Postgres not reachable; skipping database tests.")

    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (url.database,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{url.database}"')
    finally:
        conn.close()
    return test_db_url


@pytest.fixture
def session_factory(_ensure_test_db: str, test_db_url: str):
    """A session factory bound to a freshly-created, empty test schema per test."""
    from sqlalchemy import text

    from resume_parser.db.models import Base
    from resume_parser.db.session import make_session_factory

    factory = make_session_factory(test_db_url)
    engine = factory.kw["bind"]
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE resumes, skills, work_history, education, projects "
                "RESTART IDENTITY CASCADE"
            )
        )
    yield factory
    engine.dispose()
