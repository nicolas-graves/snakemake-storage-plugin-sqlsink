"""Multipart-geometry normalization: compact fact table + shared contour
table + public compatibility view, published from an already-expanded
Parquet without repeating the geometry column in PostgreSQL.
"""

from __future__ import annotations

from sqlalchemy import text

from fixtures.make_fixtures import make_multipart_geometry_fixtures
from sqlsink import metadata as meta_mod
from sqlsink.manifest import DatasetMaterialization
from sqlsink.materialize import stage_contours, stage_dataset
from sqlsink.publish import publish_datasets

MANIFEST = DatasetMaterialization(
    name="zones",
    geometry_column="polygon_coords",
    contour_table="contours",
    fact_join_columns=("zone_id",),
    contour_join_columns=("zone_id",),
    output_columns=("zone_id", "metric", "polygon_coords"),
    fact_source="facts",
)


def _fixtures(tmp_path):
    return make_multipart_geometry_fixtures(tmp_path / "parquet")


def _stage_and_publish(engine, paths):
    contour_receipt = stage_contours(engine, MANIFEST, str(paths["contours"])).to_dict()
    dataset_receipt = stage_dataset(engine, MANIFEST, str(paths["facts"]), str(paths["contours"])).to_dict()
    published = publish_datasets(engine, [MANIFEST], [dataset_receipt], [contour_receipt])
    return contour_receipt, dataset_receipt, published


def _view_rows(engine):
    with engine.connect() as conn:
        rows = conn.execute(text('SELECT zone_id, metric, polygon_coords FROM "zones" ORDER BY polygon_coords')).fetchall()
    return [tuple(r) for r in rows]


def test_compact_table_does_not_repeat_geometry(tmp_path, engine):
    paths = _fixtures(tmp_path)
    _, dataset_receipt, published = _stage_and_publish(engine, paths)

    assert published == ["zones"]
    assert dataset_receipt["status"] == "staged"
    # Two logical facts (Z1/10, Z2/20), not three expanded rows.
    assert dataset_receipt["row_count"] == 2

    compact_name, compact_schema = meta_mod.physical_name_and_schema(
        engine.dialect.name, meta_mod.compact_table_name("zones"), MANIFEST.compact_schema
    )
    compact_ref = f'"{compact_schema}"."{compact_name}"' if compact_schema else f'"{compact_name}"'
    with engine.connect() as conn:
        compact_rows = conn.execute(text(f"SELECT zone_id, metric FROM {compact_ref} ORDER BY zone_id")).fetchall()
    assert [tuple(r) for r in compact_rows] == [("Z1", 10), ("Z2", 20)]


def test_view_reconstructs_original_expanded_facts_as_multiset(tmp_path, engine):
    paths = _fixtures(tmp_path)
    _stage_and_publish(engine, paths)

    assert sorted(_view_rows(engine)) == sorted(
        [
            ("Z1", 10, "part-Z1-a"),
            ("Z1", 10, "part-Z1-b"),
            ("Z2", 20, "part-Z2-a"),
        ]
    )


def test_rerun_with_no_changes_is_a_noop(tmp_path, engine):
    paths = _fixtures(tmp_path)
    _stage_and_publish(engine, paths)
    rows_before = sorted(_view_rows(engine))

    contour_receipt = stage_contours(engine, MANIFEST, str(paths["contours"])).to_dict()
    dataset_receipt = stage_dataset(engine, MANIFEST, str(paths["facts"]), str(paths["contours"])).to_dict()
    assert contour_receipt["status"] == "current"
    assert dataset_receipt["status"] == "current"

    published = publish_datasets(engine, [MANIFEST], [dataset_receipt], [contour_receipt])
    assert published == []
    assert sorted(_view_rows(engine)) == rows_before


def test_contour_change_invalidates_dataset_even_without_fact_change(tmp_path, engine):
    import duckdb

    paths = _fixtures(tmp_path)
    _stage_and_publish(engine, paths)

    # Z1 gains a third polygon part; the fact Parquet is untouched.
    con = duckdb.connect()
    con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")
    con.execute(
        "INSERT INTO t VALUES ('Z1','part-Z1-a'), ('Z1','part-Z1-b'), "
        "('Z1','part-Z1-c'), ('Z2','part-Z2-a')"
    )
    con.execute(f"COPY t TO '{paths['contours']}' (FORMAT PARQUET)")
    con.close()

    contour_receipt = stage_contours(engine, MANIFEST, str(paths["contours"])).to_dict()
    dataset_receipt = stage_dataset(engine, MANIFEST, str(paths["facts"]), str(paths["contours"])).to_dict()
    assert contour_receipt["status"] == "staged"
    assert dataset_receipt["status"] == "staged"  # invalidated via contour_sha256, not fact_sha256

    published = publish_datasets(engine, [MANIFEST], [dataset_receipt], [contour_receipt])
    assert published == ["zones"]

    rows = sorted(_view_rows(engine))
    assert ("Z1", 10, "part-Z1-c") in rows
    assert len(rows) == 4
