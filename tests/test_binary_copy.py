"""PostgreSQL binary COPY integration checks (disposable test database only)."""

import os

import duckdb
import pytest
from sqlalchemy import Boolean, Column, Date, DateTime, Integer, MetaData, Numeric, Table, Text, select

from sqlsink.engine import bulk_load_query
from sqlsink.sink_postgres import pg_attach


pytestmark = pytest.mark.skipif(
    not os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN"), reason="needs a disposable PostgreSQL database"
)


def test_binary_query_copy_preserves_values_and_order(engine):
    table = Table(
        "binary_values", MetaData(),
        Column("id", Integer, primary_key=True),
        Column("txt", Text),
        Column("flag", Boolean),
        Column("amount", Numeric),
        Column("day", Date),
        Column("moment", DateTime),
    )
    source = duckdb.connect()
    sql = """
        SELECT 1::INTEGER AS id, 'été "a"\nline'::VARCHAR AS txt,
               true::BOOLEAN AS flag, 123.45::DECIMAL(10,2) AS amount,
               DATE '2024-02-29' AS day, TIMESTAMP '2024-02-29 12:34:56' AS moment
        UNION ALL
        SELECT 2::INTEGER, ''::VARCHAR, false::BOOLEAN, -0.01::DECIMAL(10,2),
               DATE '2020-01-01', TIMESTAMP '2020-01-01 00:00:00'
        UNION ALL
        SELECT 3::INTEGER, NULL::VARCHAR, NULL::BOOLEAN, NULL::DECIMAL(10,2),
               NULL::DATE, NULL::TIMESTAMP
    """
    with engine.begin() as conn:
        table.create(conn)
        assert bulk_load_query(conn, table, source, sql, [c.name for c in table.columns]) == 3
    with engine.connect() as conn:
        rows = conn.execute(select(table).order_by(table.c.id)).all()
    assert rows[0].txt == 'été "a"\nline'
    assert rows[0].flag is True and str(rows[0].amount) == "123.45"
    assert str(rows[0].day) == "2024-02-29"
    assert str(rows[0].moment) == "2024-02-29 12:34:56"
    assert rows[1].txt == "" and rows[1].flag is False
    assert rows[2].txt is None and rows[2].flag is None and rows[2].amount is None


@pytest.mark.parametrize("bad_id", ["1::INTEGER", "NULL::INTEGER"])
def test_binary_copy_failure_rolls_back_table_and_marker(engine, bad_id):
    table = Table("binary_failure", MetaData(), Column("id", Integer, primary_key=True))
    marker = Table("binary_marker", MetaData(), Column("id", Integer, primary_key=True))
    source = duckdb.connect()
    with pytest.raises(Exception):
        with engine.begin() as conn:
            table.create(conn)
            marker.create(conn)
            bulk_load_query(conn, table, source, "SELECT 1::INTEGER AS id", ["id"])
            bulk_load_query(conn, table, source, f"SELECT {bad_id} AS id", ["id"])
            conn.execute(marker.insert().values(id=1))
    from sqlalchemy import inspect

    assert not inspect(engine).has_table("binary_failure")
    assert not inspect(engine).has_table("binary_marker")


def test_direct_attachment_cannot_load_a_table_created_in_the_stage_transaction(engine, tmp_path):
    source = duckdb.connect()
    path = tmp_path / "source.parquet"
    source.execute("COPY (SELECT 1::INTEGER AS id) TO ? (FORMAT parquet)", [str(path)])
    pg_attach(source, engine.url, alias="separate", read_only=False)
    with engine.begin() as conn:
        conn.exec_driver_sql('CREATE TABLE "public"."binary_uncommitted" (id integer PRIMARY KEY)')
        with pytest.raises(duckdb.Error):
            source.execute('COPY separate.public.binary_uncommitted FROM ? (FORMAT parquet)', [str(path)])
        assert conn.exec_driver_sql('SELECT count(*) FROM "public"."binary_uncommitted"').scalar_one() == 0
