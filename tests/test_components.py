"""Parquet components: joined Parquet -> normalized components -> rebuilt joined
Parquet, proven equal as a multiset; plus the dialect-independent join."""

from __future__ import annotations

import duckdb
import pytest

from fixtures.make_fixtures import make_multipart_geometry_fixtures
from sql_incremental.components import (
    OrphanFactsError,
    export_joined_parquet,
    write_components,
)
from sql_incremental.join import Relation, join_spec, render_join_sql
from sql_incremental.manifest import DatasetMaterialization
from sql_incremental.materialize import compact_select_sql, read_parquet_sql, view_select_sql
from sql_incremental.verify import verify_parquet_roundtrip

MANIFEST = DatasetMaterialization(
    name="zones",
    geometry_column="polygon_coords",
    contour_table="contours",
    fact_join_columns=("zone_id",),
    contour_join_columns=("zone_id",),
    output_columns=("zone_id", "metric", "polygon_coords"),
    fact_source="facts",
)


def _rows(path):
    con = duckdb.connect()
    try:
        return sorted(con.execute(f"SELECT * FROM {read_parquet_sql(str(path))}").fetchall())
    finally:
        con.close()


def _roundtrip(tmp_path):
    paths = make_multipart_geometry_fixtures(tmp_path / "parquet")
    out = tmp_path / "components"
    receipt = write_components(MANIFEST, str(paths["facts"]), str(paths["contours"]), str(out))
    rebuilt = tmp_path / "rebuilt.parquet"
    rows = export_joined_parquet(MANIFEST, receipt.compact_path, receipt.contour_path, str(rebuilt))
    return paths, receipt, rebuilt, rows


def test_components_do_not_repeat_geometry(tmp_path):
    _, receipt, _, _ = _roundtrip(tmp_path)

    assert receipt.compact_rows == 2  # two logical facts, not three expanded rows
    assert receipt.contour_rows == 3
    assert _rows(receipt.compact_path) == [("Z1", 10), ("Z2", 20)]


def test_rebuilt_joined_parquet_equals_original_multiset(tmp_path):
    paths, _, rebuilt, rows = _roundtrip(tmp_path)

    assert rows == 3
    assert _rows(rebuilt) == _rows(paths["facts"])
    result = verify_parquet_roundtrip(MANIFEST, str(paths["facts"]), str(rebuilt))
    assert result.equivalent
    assert result.original == result.view and result.original[0] == 3


def test_components_are_reproducible_byte_for_byte(tmp_path):
    paths = make_multipart_geometry_fixtures(tmp_path / "parquet")
    first = write_components(MANIFEST, str(paths["facts"]), str(paths["contours"]), str(tmp_path / "a"))
    second = write_components(MANIFEST, str(paths["facts"]), str(paths["contours"]), str(tmp_path / "b"))

    assert first.compact_sha256 == second.compact_sha256
    assert first.contour_sha256 == second.contour_sha256


def test_orphan_facts_fail_loudly_and_write_nothing(tmp_path):
    paths = make_multipart_geometry_fixtures(tmp_path / "parquet")
    con = duckdb.connect()
    con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")
    con.execute("INSERT INTO t VALUES ('Z1','part-Z1-a')")  # Z2 has no contour any more
    con.execute(f"COPY t TO '{paths['contours']}' (FORMAT PARQUET)")
    con.close()

    out = tmp_path / "components"
    with pytest.raises(OrphanFactsError):
        write_components(MANIFEST, str(paths["facts"]), str(paths["contours"]), str(out))
    assert not out.exists() or not list(out.iterdir())


def test_verify_detects_a_dropped_part(tmp_path):
    paths, _, rebuilt, _ = _roundtrip(tmp_path)
    broken = tmp_path / "broken.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM {read_parquet_sql(str(rebuilt))} WHERE polygon_coords <> 'part-Z1-b') "
        f"TO '{broken}' (FORMAT PARQUET)"
    )
    con.close()

    result = verify_parquet_roundtrip(MANIFEST, str(paths["facts"]), str(broken))
    assert not result.equivalent


def test_verify_detects_schema_drift(tmp_path):
    paths, _, rebuilt, _ = _roundtrip(tmp_path)
    drifted = tmp_path / "drifted.parquet"
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT zone_id, CAST(metric AS BIGINT) AS metric, polygon_coords "
        f"FROM {read_parquet_sql(str(rebuilt))}) TO '{drifted}' (FORMAT PARQUET)"
    )
    con.close()

    result = verify_parquet_roundtrip(MANIFEST, str(paths["facts"]), str(drifted))
    assert result.schema_mismatches
    assert not result.equivalent


def test_pipeline_can_stage_directly_from_compact_parquet(tmp_path, engine):
    """PostgreSQL loads the components with no new code path: the compact
    projection is idempotent on an already-compact Parquet."""
    from sql_incremental.materialize import stage_contours, stage_dataset

    _, receipt, _, _ = _roundtrip(tmp_path)
    con = duckdb.connect()
    try:
        assert (
            sorted(con.execute(compact_select_sql(MANIFEST, receipt.compact_path)).fetchall())
            == _rows(receipt.compact_path)
        )
    finally:
        con.close()

    # A compact Parquet carries no geometry column, so it goes straight in.
    # The fact hash is of the compact file; the contour is the component.
    contour = stage_contours(engine, MANIFEST, receipt.contour_path)
    dataset = stage_dataset(engine, MANIFEST, receipt.compact_path, receipt.contour_path)
    assert contour.status == "staged" and dataset.status == "staged"
    assert dataset.row_count == 2


def test_join_renders_identically_for_physical_tables_and_parquet():
    pg = view_select_sql(MANIFEST, "postgresql")
    assert pg == (
        'SELECT f."zone_id", f."metric", c."polygon_coords" '
        'FROM "analytics_storage"."compact__zones" f '
        'JOIN "analytics_storage"."contours" c ON f."zone_id" = c."zone_id"'
    )

    spec = join_spec(
        MANIFEST,
        Relation(sql="read_parquet('a.parquet')"),
        Relation(sql="read_parquet('b.parquet')"),
    )
    assert render_join_sql(spec, "duckdb") == (
        'SELECT f."zone_id", f."metric", c."polygon_coords" '
        "FROM read_parquet('a.parquet') f "
        "JOIN read_parquet('b.parquet') c ON f.\"zone_id\" = c.\"zone_id\""
    )


def test_relation_requires_exactly_one_form():
    with pytest.raises(ValueError):
        Relation()
    with pytest.raises(ValueError):
        Relation(name="t", sql="x")
