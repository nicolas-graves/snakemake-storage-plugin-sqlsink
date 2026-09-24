"""Parquet is an export of the published SQL dataset, not a sink of its own."""

from __future__ import annotations

import duckdb
import pytest
from compat_support import MANIFEST, capability, original_rows

from sqlsink.export import DatasetNotPublished, export_parquet
from sqlsink.queries import read_parquet_sql
from sqlsink.sink import materialize
from sqlsink.verify import verify_parquet_roundtrip


def _rows(path):
    con = duckdb.connect()
    try:
        return sorted(
            tuple(None if v is None else str(v) for v in r)
            for r in con.execute(f"SELECT * FROM {read_parquet_sql(str(path))}").fetchall()
        )
    finally:
        con.close()


def _describe(path):
    con = duckdb.connect()
    try:
        return [(c[0], c[1]) for c in con.execute(f"DESCRIBE SELECT * FROM {read_parquet_sql(str(path))}").fetchall()]
    finally:
        con.close()


@capability("parquet_export", "export_equals_original", "beyond", note="the export is taken from the published version")
def test_the_export_equals_the_original_multiset(engine, sink, paths, tmp_path):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    out = tmp_path / "out" / "zones.parquet"

    assert export_parquet(sink, MANIFEST, str(out)) == 3
    assert _rows(out) == original_rows(paths)
    result = verify_parquet_roundtrip(MANIFEST, str(paths["facts"]), str(out))
    assert result.equivalent and result.original == result.view


@capability("parquet_export", "column_names_and_order", "beyond")
def test_the_export_keeps_the_declared_columns(sink, paths, tmp_path):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    out = tmp_path / "zones.parquet"
    export_parquet(sink, MANIFEST, str(out))
    assert [name for name, _ in _describe(out)] == list(MANIFEST.output_columns)


@capability("parquet_export", "column_types_preserved", "beyond", note="types survive the SQL round trip")
def test_the_export_keeps_the_source_column_types(sink, paths, tmp_path):
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    out = tmp_path / "zones.parquet"
    export_parquet(sink, MANIFEST, str(out))
    assert _describe(out) == _describe(paths["facts"])


@capability("parquet_export", "unpublished_fails", "beyond")
def test_exporting_an_unpublished_dataset_fails_and_writes_nothing(sink, tmp_path):
    out = tmp_path / "zones.parquet"
    with pytest.raises(DatasetNotPublished):
        export_parquet(sink, MANIFEST, str(out))
    assert not out.exists()
