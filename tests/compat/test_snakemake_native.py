"""What we keep by staying inside Snakemake: freshness comes from the
dependency graph through a storage plugin, with no second orchestrator."""

from __future__ import annotations

import logging

import duckdb
import pytest

from compat_support import capability
from sqlsink.publish import publish_tables
from sqlsink.stage import stage_table

pytest.importorskip("snakemake_storage_plugin_sqlsink")
from snakemake_storage_plugin_sqlsink import (  # noqa: E402
    StorageObject,
    StorageProvider,
    StorageProviderSettings,
)


@capability("snakemake", "freshness_via_storage_plugin", "beyond", note="dbt/Dagster need their own scheduler for this")
def test_a_rule_input_is_stale_exactly_when_its_source_changes(engine, parquet_dir, tmp_path):
    provider = StorageProvider(
        local_prefix=tmp_path / "local",
        logger=logging.getLogger("test"),
        settings=StorageProviderSettings(dsn="duckdb:///:memory:", parquet_dir=str(next(iter(parquet_dir.values())).parent)),
    )
    provider.engine = engine
    obj = StorageObject(query="fake_a", keep_local=True, retrieve=True, provider=provider)

    assert obj.exists() is False
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    assert obj.exists() is True

    con = duckdb.connect()
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 'changed', 1.0)")
    con.execute(f"COPY t TO '{parquet_dir['fake_a']}' (FORMAT PARQUET)")
    con.close()
    assert obj.exists() is False
