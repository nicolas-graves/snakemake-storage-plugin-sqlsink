"""dbt's advantages: build in staging, constrain before insert, swap, skip
what is unchanged, contract the output schema."""

from __future__ import annotations

import dataclasses

import duckdb
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from fixtures.dialects import declared_keys, pk_columns, qualified
from compat_support import MANIFEST, PARTS_V2, capability, has_feature, joined_rows, rewrite_contours
from sqlsink import metadata as meta_mod
from sqlsink import relations
from sqlsink.sink import OrphanFactsError, duckdb_session, materialize, normalize, publish, stage
from sqlsink.sink_postgres import SqlSink

KEYED = dataclasses.replace(MANIFEST, keyed=True)


def _phys(engine, bare):
    return meta_mod.physical_name_and_schema(engine.dialect.name, bare, KEYED.compact_schema)


@capability("dbt", "staging_then_swap")
def test_nothing_is_visible_until_publish(sink, paths):
    dataset = normalize(MANIFEST, str(paths["facts"]), str(paths["contours"]))
    result = stage(dataset, sink)

    assert sink.current_update_id("zones") is None  # staged, not visible
    assert publish(sink, [MANIFEST], [result]) == ["zones"]
    assert sink.current_update_id("zones") == dataset.update_id


@capability("dbt", "relationships_test")
def test_a_fact_without_a_dimension_row_is_refused_before_any_write(sink, paths):
    rewrite_contours(paths, "('Z1','part-Z1-a')")  # Z2 has no contour
    with pytest.raises(OrphanFactsError):
        materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    assert sink.current_update_id("zones") is None


@capability("dbt", "constraints_before_insert", note="PK/FK declared in staging DDL")
def test_keys_are_declared_and_enforced_on_the_published_tables(engine, paths):
    materialize(KEYED, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    zone, zs = _phys(engine, "contours_zone")
    parts, ps = _phys(engine, "contours")
    facts, fs = _phys(engine, meta_mod.compact_table_name("zones"))
    insp = inspect(engine)

    assert pk_columns(engine, zone, zs) == ["zone_id"]
    with declared_keys(engine):
        assert insp.get_foreign_keys(facts, schema=fs)  # facts -> zone
        assert insp.get_foreign_keys(parts, schema=ps)  # parts -> zone
    with pytest.raises(IntegrityError), engine.begin() as conn:  # unique / primary_key
        conn.execute(text(f"INSERT INTO {qualified(zone, zs)} (zone_id) VALUES ('Z1')"))


@capability("dbt", "fk_survives_swap", note="reduces the FK-across-swap risk")
def test_foreign_keys_follow_the_republished_tables(engine, paths):
    sink = SqlSink(engine)
    materialize(KEYED, str(paths["facts"]), str(paths["contours"]), sink)
    rewrite_contours(paths, PARTS_V2)
    assert materialize(KEYED, str(paths["facts"]), str(paths["contours"]), sink)[0] == ["zones"]

    zone, _ = _phys(engine, "contours_zone")
    facts, fs = _phys(engine, meta_mod.compact_table_name("zones"))
    with declared_keys(engine):
        (fk,) = inspect(engine).get_foreign_keys(facts, schema=fs)
        assert fk["referred_table"] == zone


@capability("dbt", "constraints_not_on_views")
def test_the_public_name_is_a_view_without_constraints(engine, paths):
    materialize(KEYED, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    insp = inspect(engine)

    assert "zones" in insp.get_view_names()
    assert insp.get_pk_constraint("zones")["constrained_columns"] == []
    assert insp.get_foreign_keys("zones") == []


@capability("dbt", "incremental_skip", note="state:modified analogue")
def test_unchanged_inputs_skip_and_a_manifest_change_alone_invalidates(sink, paths):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    published, result = materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    assert published == [] and result.dataset.status == "current"

    reordered = dataclasses.replace(MANIFEST, output_columns=("zone_id", "polygon_coords", "metric"))
    published, _ = materialize(reordered, str(paths["facts"]), str(paths["contours"]), sink)
    assert published == ["zones"]  # same files, new manifest


@capability("dbt", "model_contract", note="output columns and order are fixed by the manifest")
def test_published_columns_follow_the_declared_order(sink, paths):
    reordered = dataclasses.replace(MANIFEST, output_columns=("polygon_coords", "zone_id", "metric"))
    materialize(reordered, str(paths["facts"]), str(paths["contours"]), sink)
    with duckdb_session(None, None, sink.spill_dir()) as con:
        relation = sink.joined_relation(reordered, con)
        columns = [row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]
    assert columns == list(reordered.output_columns)


@capability(
    "dbt",
    "accepted_values_test",
    "gap",
    note="close: a manifest `checks` field run in stage; adopt: none needed (dbt tests need dbt runtime)",
)
def test_value_checks_can_be_declared_on_a_dataset():
    assert has_feature(MANIFEST, "checks") or has_feature(MANIFEST, "accepted_values")


@capability(
    "dbt",
    "docs_export",
    "gap",
    note="close: inverse of graph_from_spec (Frictionless descriptor); adopt: frictionless-py, optional",
)
def test_the_relation_graph_exports_a_standard_descriptor():
    assert has_feature(relations, "graph_to_spec")
