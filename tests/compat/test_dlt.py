"""dlt's advantages: nested data -> child tables, load-state tracking,
schema contracts. Ours goes further on enforcement."""

from __future__ import annotations

import dataclasses

import duckdb
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from compat_support import MANIFEST, capability
from fixtures.dialects import declared_keys, qualified
from sqlsink import metadata as meta_mod
from sqlsink.sink import materialize, normalize, stage
from sqlsink.sink_postgres import SqlSink, fetch_staged_dataset_marker

KEYED = dataclasses.replace(MANIFEST, keyed=True)


def _phys(engine, bare):
    return meta_mod.physical_name_and_schema(engine.dialect.name, bare, KEYED.compact_schema)


@capability("dlt", "nested_to_child_rows", note="polygon parts become zone_part rows keyed by (zone, part_no)")
def test_polygon_parts_become_ordered_child_rows(engine, paths):
    materialize(KEYED, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    parts, ps = _phys(engine, "contours")
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT zone_id, part_no FROM {qualified(parts, ps)} ORDER BY 1, 2")).fetchall()
    assert [tuple(r) for r in rows] == [("Z1", 1), ("Z1", 2), ("Z2", 1)]


@capability("dlt", "load_state_marker", "beyond", note="dlt _dlt_loads has no staged-vs-published distinction")
def test_a_staged_load_is_recorded_separately_from_the_published_one(sink, engine, paths):
    result = stage(normalize(MANIFEST, str(paths["facts"]), str(paths["contours"])), sink)
    assert sink.current_update_id("zones") is None
    assert fetch_staged_dataset_marker(engine, "zones")["update_id"] == result.dataset.update_id


@capability("dlt", "references_enforced", "beyond", note="dlt references are annotations, not verified")
def test_a_declared_reference_is_enforced_by_the_database(engine, paths):
    materialize(KEYED, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    facts, fs = _phys(engine, meta_mod.compact_table_name("zones"))
    with declared_keys(engine), pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(text(f"INSERT INTO {qualified(facts, fs)} (zone_id, metric) VALUES ('NOPE', 1)"))


@capability("dlt", "schema_contract", note="dlt 'freeze' mode: an unexpected schema change stops the load")
def test_a_source_that_lost_a_declared_column_fails_and_keeps_the_previous_version(sink, paths):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    first = sink.current_update_id("zones")

    con = duckdb.connect()
    con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")  # `metric` dropped
    con.execute("INSERT INTO t VALUES ('Z1','part-Z1-a'), ('Z2','part-Z2-a')")
    con.execute(f"COPY t TO '{paths['facts']}' (FORMAT PARQUET)")
    con.close()

    with pytest.raises(Exception):
        materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    assert sink.current_update_id("zones") == first
