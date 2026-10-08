"""Concurrent first use of an empty database must not race on the marker tables."""
import os
import threading

import pytest
from sqlalchemy import text

from sqlsink.engine import make_engine
from sqlsink.metadata import create_all


@pytest.mark.skipif(not os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN"),
                    reason="needs a disposable PostgreSQL (SNAKEMAKE_SQL_TEST_PG_DSN)")
def test_concurrent_create_all_on_an_empty_database():
    dsn = os.environ["SNAKEMAKE_SQL_TEST_PG_DSN"]
    engines = [make_engine(dsn) for _ in range(8)]
    with engines[0].begin() as conn:
        conn.execute(text('DROP SCHEMA IF EXISTS "public" CASCADE'))
        conn.execute(text('CREATE SCHEMA "public"'))
    barrier, errors = threading.Barrier(len(engines)), []

    def bootstrap(engine):
        barrier.wait()
        try:
            create_all(engine)
        except Exception as error:  # collected: a thread's exception would be lost
            errors.append(error)

    threads = [threading.Thread(target=bootstrap, args=(e,)) for e in engines]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for engine in engines:
        engine.dispose()
    assert errors == []
