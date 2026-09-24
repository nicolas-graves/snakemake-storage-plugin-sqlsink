import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "workflow" / "scripts"))

import pytest
from sqlalchemy import text

from fixtures.make_fixtures import make_default_fixtures  # noqa: E402
from sql_incremental.engine import make_engine  # noqa: E402
from sql_incremental.metadata import create_all  # noqa: E402


@pytest.fixture
def engine(tmp_path):
    """A DuckDB file per test. Set SNAKEMAKE_SQL_TEST_PG_DSN to a *disposable*
    PostgreSQL database to run the same tests there: its `public` and
    `analytics_storage` schemas are dropped and recreated before each test."""
    dsn = os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN")
    eng = make_engine(dsn or f"duckdb:///{tmp_path / 'sink.duckdb'}")
    if dsn:
        with eng.begin() as conn:
            for schema in ("public", "analytics_storage"):
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            conn.execute(text('CREATE SCHEMA "public"'))
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def parquet_dir(tmp_path):
    return make_default_fixtures(tmp_path / "parquet")
