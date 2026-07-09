"""Pytest fixtures for Ontogen's real-Postgres DB tests.

Mirrors ocr_resume_parser's conftest pattern (a real test database, created on
demand, skipped-with-warning when Postgres is down) but layers on Ontogen's
needs: the pgvector extension and the shared Base's Ontogen tables. DB imports
live inside the fixtures so the rest of any suite runs without a server.
"""
import os
import sys
import warnings
from pathlib import Path

import pytest

_ONTOGEN_ROOT = Path(__file__).resolve().parent
_PARSER_ROOT = _ONTOGEN_ROOT.parent / "ocr-resume-paser"
for _p in (str(_ONTOGEN_ROOT), str(_PARSER_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TEST_DB_ENV = "TEST_DATABASE_URL"

# Every table managed on the shared Base — truncated between tests.
_ALL_TABLES = (
    "graph_relationships, graph_entities, provenance, pipeline_runs, "
    "canon_store, gazetteers, wikidata_properties, "
    "projects, skills, work_history, education, resumes"
)


@pytest.fixture(scope="session")
def test_db_url() -> str:
    from dotenv import load_dotenv

    load_dotenv(_PARSER_ROOT / ".env", override=False)
    load_dotenv(override=False)
    url = os.environ.get(TEST_DB_ENV)
    if not url:
        pytest.skip(f"{TEST_DB_ENV} not set; skipping database tests.")
    return url


@pytest.fixture(scope="session")
def _ensure_test_db(test_db_url: str) -> str:
    """Create the test database if absent; skip-with-warning if no server."""
    import psycopg2
    from sqlalchemy.engine import make_url

    url = make_url(test_db_url)
    try:
        conn = psycopg2.connect(
            host=url.host, port=url.port, user=url.username,
            password=url.password, dbname="postgres",
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
def ontogen_session_factory(_ensure_test_db: str, test_db_url: str):
    """A session factory on a fresh, empty schema (parser + Ontogen tables) with
    the pgvector extension available."""
    from sqlalchemy import text

    from db.models import Base  # shared Base: registers parser + ontogen tables
    from db.session import make_session_factory

    factory = make_session_factory(test_db_url)
    engine = factory.kw["bind"]
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))  # gen_random_uuid()
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {_ALL_TABLES} RESTART IDENTITY CASCADE"))
    yield factory
    engine.dispose()
