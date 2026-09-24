"""Unit tests for the private storage plugin, exercised directly against a
StorageObject instance (no full Snakemake DAG needed for this level).
"""

from __future__ import annotations

import logging

import pytest

from sqlsink.publish import publish_tables
from sqlsink.stage import stage_table

pytest.importorskip("snakemake_storage_plugin_sqlsink")

from snakemake_storage_plugin_sqlsink import (  # noqa: E402
    StorageObject,
    StorageProvider,
    StorageProviderSettings,
)


@pytest.fixture
def provider(engine, parquet_dir, tmp_path):
    p = StorageProvider(
        local_prefix=tmp_path / "local",
        logger=logging.getLogger("test"),
        settings=StorageProviderSettings(dsn="duckdb:///:memory:", parquet_dir=str(next(iter(parquet_dir.values())).parent)),
    )
    p.engine = engine  # reuse the test's in-memory engine/connection pool
    return p


def _obj(provider, table_name):
    return StorageObject(query=table_name, keep_local=True, retrieve=True, provider=provider)


def test_exists_false_before_any_publish(provider):
    assert _obj(provider, "fake_a").exists() is False


def test_exists_true_after_publish_and_mtime_reflects_marker(provider, engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])

    obj = _obj(provider, "fake_a")
    assert obj.exists() is True
    assert obj.mtime() > 0


def test_exists_becomes_false_again_after_parquet_changes(provider, engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])
    assert _obj(provider, "fake_a").exists() is True

    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 'changed', 1.0)")
    con.execute(f"COPY t TO '{parquet_dir['fake_a']}' (FORMAT PARQUET)")
    con.close()

    assert _obj(provider, "fake_a").exists() is False


def test_retrieve_object_writes_manifest_for_current_table(provider, engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])

    obj = _obj(provider, "fake_a")
    obj.retrieve_object()

    import json

    manifest = json.loads(obj.local_path().read_text())
    assert manifest["table"] == "fake_a"
    assert manifest["status"] == "current"


def test_exists_false_when_the_published_table_was_dropped_but_its_marker_remains(provider, engine, parquet_dir):
    from sqlalchemy import text

    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE "fake_a"'))

    assert _obj(provider, "fake_a").exists() is False
    assert stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).status == "staged"


def test_exists_false_when_the_staging_table_was_dropped_but_its_marker_remains(provider, engine, parquet_dir):
    from sqlalchemy import text

    from sqlsink.metadata import staging_name

    stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    assert _obj(provider, "fake_a").exists() is True  # staged, awaiting publish
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE "{staging_name("fake_a")}"'))

    assert _obj(provider, "fake_a").exists() is False
