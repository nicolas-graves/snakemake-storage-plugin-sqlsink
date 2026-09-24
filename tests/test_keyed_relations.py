"""The keyed (relational) form: zone + zone_part + facts with declared keys,
same joined output as the flat form."""

from __future__ import annotations

import dataclasses

import duckdb
import pytest
from sqlalchemy import inspect, text

from fixtures.make_fixtures import make_multipart_geometry_fixtures
from sqlsink import metadata as meta_mod
from sqlsink.manifest import DatasetMaterialization
from sqlsink.publish import PublishConflict
from sqlsink.queries import read_parquet_sql
from sqlsink.sink import materialize, normalize, publish, stage
from fixtures.dialects import declared_keys, pk_columns, qualified
from sqlsink.sink_postgres import SqlSink

MANIFEST = DatasetMaterialization(
    name="zones",
    geometry_column="polygon_coords",
    contour_table="contours",
    fact_join_columns=("zone_id",),
    contour_join_columns=("zone_id",),
    output_columns=("zone_id", "metric", "polygon_coords"),
    fact_source="facts",
    keyed=True,
)
OTHER = dataclasses.replace(MANIFEST, name="others")


@pytest.fixture
def paths(tmp_path):
    return make_multipart_geometry_fixtures(tmp_path / "parquet")


def _phys(engine, bare):
    return meta_mod.physical_name_and_schema(engine.dialect.name, bare, MANIFEST.compact_schema)


def _joined(engine):
    with engine.connect() as conn:
        return sorted(tuple(r) for r in conn.execute(text('SELECT * FROM "zones"')).fetchall())


def test_keyed_publish_creates_declared_keys(engine, paths):
    sink = SqlSink(engine)
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    insp = inspect(engine)

    zone, zs = _phys(engine, "contours_zone")
    parts, ps = _phys(engine, "contours")
    facts, fs = _phys(engine, meta_mod.compact_table_name("zones"))

    assert pk_columns(engine, zone, zs) == ["zone_id"]
    assert pk_columns(engine, parts, ps) == ["zone_id", "part_no"]
    with declared_keys(engine):
        (parts_fk,) = insp.get_foreign_keys(parts, schema=ps)
        (facts_fk,) = insp.get_foreign_keys(facts, schema=fs)
        for fk in (parts_fk, facts_fk):
            assert fk["constrained_columns"] == ["zone_id"]
            assert fk["referred_columns"] == ["zone_id"]
            assert fk["referred_table"] == zone  # follows the zone through the rename-swap
        assert any(ix["column_names"] == ["zone_id"] for ix in insp.get_indexes(facts, schema=fs))


def test_keyed_view_output_equals_original_rows(engine, paths):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    con = duckdb.connect()
    original = sorted(
        tuple(r) for r in con.execute(f"SELECT * FROM {read_parquet_sql(str(paths['facts']))}").fetchall()
    )
    assert _joined(engine) == original


def test_part_numbers_are_deterministic(engine, paths):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    parts, ps = _phys(engine, "contours")
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT zone_id, part_no, polygon_coords FROM {qualified(parts, ps)} ORDER BY 1, 2")).fetchall()
    assert [tuple(r) for r in rows] == [
        ("Z1", 1, "part-Z1-a"),
        ("Z1", 2, "part-Z1-b"),
        ("Z2", 1, "part-Z2-a"),
    ]


def test_keyed_flag_changes_the_manifest_hash_only_when_set():
    flat = dataclasses.replace(MANIFEST, keyed=False)
    assert "keyed" not in flat.canonical_dict()
    assert flat.manifest_hash() != MANIFEST.manifest_hash()


def test_relation_graph_from_manifest_is_valid_and_transitive():
    graph = MANIFEST.relation_graph()
    assert graph.path("zones", "contours_zone").cardinality == "many_to_one"
    # The parts sit behind a fan-out: facts do not reach them without opting in.
    with pytest.raises(Exception, match="multiplies rows"):
        graph.path("zones", "contours")


def test_replacing_a_shared_zone_requires_restaging_its_dependents(engine, paths):
    sink = SqlSink(engine)
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    materialize(OTHER, str(paths["facts"]), str(paths["contours"]), sink)

    con = duckdb.connect()
    con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")
    con.execute("INSERT INTO t VALUES ('Z1','new-a'), ('Z2','new-b')")
    con.execute(f"COPY t TO '{paths['contours']}' (FORMAT PARQUET)")
    con.close()

    result = stage(normalize(MANIFEST, str(paths["facts"]), str(paths["contours"])), sink)
    with pytest.raises(PublishConflict, match="others"):
        publish(sink, [MANIFEST, OTHER], [result])

    # Restaging both is accepted and leaves both consistent.
    other = stage(normalize(OTHER, str(paths["facts"]), str(paths["contours"])), sink)
    assert publish(sink, [MANIFEST, OTHER], [result, other])
    assert len(_joined(engine)) == 2
