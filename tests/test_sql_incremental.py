from __future__ import annotations

from sqlalchemy import inspect, select, text

from sql_incremental.metadata import analytics_table_updates
from sql_incremental.publish import PublishConflict, publish_tables
from sql_incremental.stage import stage_table


def _fetch_all(engine, table_name):
    with engine.connect() as conn:
        rows = conn.execute(text(f'SELECT * FROM "{table_name}" ORDER BY id')).fetchall()
    return rows


def _stage_all(engine, parquet_dir):
    return [stage_table(engine, name, str(path)).to_dict() for name, path in parquet_dir.items()]


def test_cold_start_stages_and_publishes_everything(engine, parquet_dir):
    receipts = _stage_all(engine, parquet_dir)
    assert all(r["status"] == "staged" for r in receipts)

    published = publish_tables(engine, receipts)
    assert sorted(published) == sorted(parquet_dir.keys())

    for name in parquet_dir:
        rows = _fetch_all(engine, name)
        assert len(rows) > 0


def test_rerun_with_no_changes_is_a_noop(engine, parquet_dir):
    receipts = _stage_all(engine, parquet_dir)
    publish_tables(engine, receipts)

    with engine.connect() as conn:
        before = {
            row.table_name: row.published_at
            for row in conn.execute(select(analytics_table_updates))
        }

    receipts_2 = _stage_all(engine, parquet_dir)
    assert all(r["status"] == "current" for r in receipts_2)

    published_2 = publish_tables(engine, receipts_2)
    assert published_2 == []

    with engine.connect() as conn:
        after = {
            row.table_name: row.published_at
            for row in conn.execute(select(analytics_table_updates))
        }
    assert before == after


def test_changing_one_table_only_touches_that_table(engine, parquet_dir):
    receipts = _stage_all(engine, parquet_dir)
    publish_tables(engine, receipts)

    with engine.connect() as conn:
        before = {
            row.table_name: row.update_id
            for row in conn.execute(select(analytics_table_updates))
        }
    before_rows = {name: _fetch_all(engine, name) for name in parquet_dir}

    # Change only fake_b's Parquet.
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 'dan', 4.0), (2, 'zoe', 99.0), (3, 'new', 1.0)")
    con.execute(f"COPY t TO '{parquet_dir['fake_b']}' (FORMAT PARQUET)")
    con.close()

    receipts_2 = _stage_all(engine, parquet_dir)
    statuses = {r["table"]: r["status"] for r in receipts_2}
    assert statuses["fake_b"] == "staged"
    assert statuses["fake_a"] == "current"
    assert statuses["fake_c"] == "current"

    published = publish_tables(engine, receipts_2)
    assert published == ["fake_b"]

    with engine.connect() as conn:
        after = {
            row.table_name: row.update_id
            for row in conn.execute(select(analytics_table_updates))
        }
    assert after["fake_a"] == before["fake_a"]
    assert after["fake_c"] == before["fake_c"]
    assert after["fake_b"] != before["fake_b"]

    assert _fetch_all(engine, "fake_a") == before_rows["fake_a"]
    assert _fetch_all(engine, "fake_c") == before_rows["fake_c"]
    assert len(_fetch_all(engine, "fake_b")) == 3


def test_interrupted_stage_is_repaired_on_rerun(engine, parquet_dir, monkeypatch):
    import sql_incremental.stage as stage_mod

    real_bulk_load = stage_mod.bulk_load
    calls = {"n": 0}

    def flaky_bulk_load(conn, table, rows):
        calls["n"] += 1
        if table.name.endswith("fake_b") and calls["n"] == 1:
            raise RuntimeError("simulated crash mid-stage")
        return real_bulk_load(conn, table, rows)

    monkeypatch.setattr(stage_mod, "bulk_load", flaky_bulk_load)

    try:
        stage_table(engine, "fake_b", str(parquet_dir["fake_b"]))
    except RuntimeError:
        pass

    inspector = inspect(engine)
    assert not inspector.has_table("fake_b")  # public table never existed/touched

    # Rerun cleanly repairs it.
    receipt = stage_table(engine, "fake_b", str(parquet_dir["fake_b"]))
    assert receipt.to_dict()["status"] == "staged"
    publish_tables(engine, [receipt.to_dict()])
    assert inspect(engine).has_table("fake_b")


def test_interrupted_publish_rolls_back_all_renames(engine, parquet_dir):
    receipts = _stage_all(engine, parquet_dir)
    publish_tables(engine, receipts)

    with engine.connect() as conn:
        before_a = _fetch_all(engine, "fake_a")

    # Change fake_a and fake_b's Parquet so both get staged...
    import duckdb

    for name in ("fake_a", "fake_b"):
        con = duckdb.connect()
        con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
        con.execute("INSERT INTO t VALUES (1, 'changed', 1.0)")
        con.execute(f"COPY t TO '{parquet_dir[name]}' (FORMAT PARQUET)")
        con.close()

    receipts_2 = _stage_all(engine, parquet_dir)
    assert {r["table"]: r["status"] for r in receipts_2}["fake_a"] == "staged"
    assert {r["table"]: r["status"] for r in receipts_2}["fake_b"] == "staged"

    # ...then simulate another process publishing a newer version of
    # fake_b between this run's staging and its (still pending) publish.
    with engine.begin() as conn:
        conn.execute(
            analytics_table_updates.update()
            .where(analytics_table_updates.c.table_name == "fake_b")
            .values(update_id="published-by-another-process")
        )

    import pytest

    with pytest.raises(PublishConflict):
        publish_tables(engine, receipts_2)

    # fake_a's rename must have been rolled back along with fake_b's failure,
    # since both were in the same transaction.
    inspector = inspect(engine)
    assert inspector.has_table("fake_a")
    assert not any(n.startswith("fake_a__old__") for n in inspector.get_table_names())
    assert _fetch_all(engine, "fake_a") == before_a
