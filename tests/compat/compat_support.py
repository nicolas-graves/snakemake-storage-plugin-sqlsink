"""Shared pieces for the capability suite: one manifest, canonical row
helpers, and the `capability` marker shorthand."""

from __future__ import annotations

import dataclasses

import duckdb
import pytest

from sql_incremental.manifest import DatasetMaterialization
from sql_incremental.queries import read_parquet_sql
from sql_incremental.sink import duckdb_session

MANIFEST = DatasetMaterialization(
    name="zones",
    geometry_column="polygon_coords",
    contour_table="contours",
    fact_join_columns=("zone_id",),
    contour_join_columns=("zone_id",),
    output_columns=("zone_id", "metric", "polygon_coords"),
    fact_source="facts",
)
OTHER = dataclasses.replace(MANIFEST, name="others")

PARTS_V2 = "('Z1','part-Z1-a'), ('Z1','part-Z1-b'), ('Z1','part-Z1-c'), ('Z2','part-Z2-a')"


def capability(tool: str, id: str, kind: str = "parity", note: str = ""):
    """Tie a test to one advantage of a reference tool.

    kind: parity -- we reproduce what the tool does;
          beyond -- we do something the tool does not;
          gap    -- the tool does it, we do not yet (the test is a strict
                    xfail that starts failing loudly once the gap closes).
    """
    assert kind in ("parity", "beyond", "gap"), kind
    mark = pytest.mark.capability(tool=tool, id=id, kind=kind, note=note)
    if kind != "gap":
        return mark

    def apply(fn):
        return pytest.mark.xfail(strict=True, reason=f"gap: {tool}/{id}")(mark(fn))

    return apply


def canonical(rows):
    return sorted(tuple(None if v is None else str(v) for v in row) for row in rows)


def original_rows(paths):
    con = duckdb.connect()
    try:
        return canonical(con.execute(f"SELECT * FROM {read_parquet_sql(str(paths['facts']))}").fetchall())
    finally:
        con.close()


def joined_rows(sink, manifest=MANIFEST):
    with duckdb_session(None, None, sink.spill_dir()) as con:
        relation = sink.joined_relation(manifest, con)
        return canonical(con.execute(f"SELECT * FROM {relation}").fetchall())


def rewrite_contours(paths, values_sql):
    con = duckdb.connect()
    con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")
    con.execute(f"INSERT INTO t VALUES {values_sql}")
    con.execute(f"COPY t TO '{paths['contours']}' (FORMAT PARQUET)")
    con.close()


def has_feature(obj, name: str) -> bool:
    """Probe used by `gap` tests: does the API expose the capability yet?"""
    return hasattr(obj, name)
