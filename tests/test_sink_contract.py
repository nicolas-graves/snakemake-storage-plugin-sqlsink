"""The sink contract: every behavior below is asserted through the
`sink.materialize` API against the SQL sink (a DuckDB file stands in for
PostgreSQL)."""

from __future__ import annotations

import duckdb
import pytest

from fixtures.make_fixtures import make_multipart_geometry_fixtures
from sql_incremental import publish as publish_mod
from sql_incremental.manifest import DatasetMaterialization
from sql_incremental.queries import read_parquet_sql
from sql_incremental.sink import (
    OrphanFactsError,
    duckdb_session,
    make_sink,
    materialize,
    normalize,
    publish,
    stage,
)
from sql_incremental.sink_postgres import SqlSink, fetch_staged_dataset_marker

MANIFEST = DatasetMaterialization(
    name="zones",
    geometry_column="polygon_coords",
    contour_table="contours",
    fact_join_columns=("zone_id",),
    contour_join_columns=("zone_id",),
    output_columns=("zone_id", "metric", "polygon_coords"),
    fact_source="facts",
)


class _Boom(RuntimeError):
    pass


def _fail_sql_marker(monkeypatch):
    def boom(*args, **kwargs):
        raise _Boom()

    monkeypatch.setattr(publish_mod, "_upsert_dataset_marker", boom)


@pytest.fixture
def kit(engine):
    """(sink, nothing_written)"""
    return SqlSink(engine), lambda: fetch_staged_dataset_marker(engine, "zones") is None


@pytest.fixture
def paths(tmp_path):
    return make_multipart_geometry_fixtures(tmp_path / "parquet")


def _original_rows(paths):
    con = duckdb.connect()
    try:
        return _canonical(con.execute(f"SELECT * FROM {read_parquet_sql(str(paths['facts']))}").fetchall())
    finally:
        con.close()


def _canonical(rows):
    return sorted(tuple(None if v is None else str(v) for v in row) for row in rows)


def _joined_rows(sink, manifest=MANIFEST):
    with duckdb_session(None, None, sink.spill_dir()) as con:
        relation = sink.joined_relation(manifest, con)
        return _canonical(con.execute(f"SELECT * FROM {relation}").fetchall())


def _rewrite_contours(paths, values_sql):
    con = duckdb.connect()
    con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")
    con.execute(f"INSERT INTO t VALUES {values_sql}")
    con.execute(f"COPY t TO '{paths['contours']}' (FORMAT PARQUET)")
    con.close()


def test_published_joined_dataset_equals_the_original_multiset(kit, paths):
    sink, _ = kit

    published, result = materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)

    assert published == ["zones"]
    assert result.dataset.status == "staged"
    assert result.dataset.row_count == 2  # two logical facts, not three expanded rows
    assert sink.current_update_id("zones") == normalize(
        MANIFEST, str(paths["facts"]), str(paths["contours"])
    ).update_id
    assert _joined_rows(sink) == _original_rows(paths)


def test_rerun_with_unchanged_inputs_is_current_and_publishes_nothing(kit, paths):
    sink, _ = kit
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    before = _joined_rows(sink)

    published, result = materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)

    assert published == []
    assert result.dataset.status == "current"
    assert result.contour.status == "current"
    assert _joined_rows(sink) == before


def test_contour_change_republishes_even_without_fact_change(kit, paths):
    sink, _ = kit
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    first = sink.current_update_id("zones")

    _rewrite_contours(
        paths,
        "('Z1','part-Z1-a'), ('Z1','part-Z1-b'), ('Z1','part-Z1-c'), ('Z2','part-Z2-a')",
    )
    published, result = materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)

    assert published == ["zones"]
    assert result.contour.status == "staged" and result.dataset.status == "staged"
    assert sink.current_update_id("zones") != first
    rows = _joined_rows(sink)
    assert ("Z1", "10", "part-Z1-c") in rows and len(rows) == 4


def test_orphan_facts_fail_before_the_sink_writes_anything(kit, paths):
    sink, nothing_written = kit
    _rewrite_contours(paths, "('Z1','part-Z1-a')")  # Z2 has no contour

    with pytest.raises(OrphanFactsError):
        materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)

    assert nothing_written()
    assert sink.current_update_id("zones") is None


def test_crash_before_the_commit_point_keeps_the_previous_version(kit, paths):
    sink, _ = kit
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    first_id = sink.current_update_id("zones")
    first_rows = _joined_rows(sink)

    _rewrite_contours(
        paths,
        "('Z1','part-Z1-a'), ('Z1','part-Z1-b'), ('Z1','part-Z1-c'), ('Z2','part-Z2-a')",
    )
    dataset = normalize(MANIFEST, str(paths["facts"]), str(paths["contours"]))
    result = stage(dataset, sink)
    # A publish that dies at its commit point must not change what is current.
    with pytest.MonkeyPatch.context() as patch:
        _install = _fail_sql_marker
        _install(patch)
        with pytest.raises(_Boom):
            publish(sink, [MANIFEST], [result])
    assert sink.current_update_id("zones") == first_id

    # The staged work is reused and the retry commits.
    assert publish(sink, [MANIFEST], [result]) == ["zones"]
    assert sink.current_update_id("zones") == dataset.update_id
    assert len(_joined_rows(sink)) == 4
    assert first_rows != _joined_rows(sink)


def test_make_sink_builds_each_kind_and_rejects_unknown(tmp_path):
    assert isinstance(make_sink({"type": "duckdb", "path": str(tmp_path / "s.duckdb")}), SqlSink)
    with pytest.raises(ValueError):
        make_sink({"type": "csv"})


def test_a_republished_marker_advances_its_timestamp(kit, paths):
    sink, _ = kit
    from sqlalchemy import select

    from sql_incremental import metadata as meta_mod

    def published_at():
        with sink.engine.connect() as conn:
            return conn.execute(select(meta_mod.analytics_dataset_updates.c.published_at)).scalar_one()

    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    first = published_at()
    _rewrite_contours(paths, "('Z1','part-Z1-a'), ('Z1','part-Z1-b'), ('Z2','part-Z2-a'), ('Z2','part-Z2-b')")
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    assert published_at() > first
