"""Build small fake Parquet tables for tests, using DuckDB (already a
pipeline dependency) rather than pyarrow directly.
"""

from __future__ import annotations

from pathlib import Path

import duckdb


def write_fixture(path: str | Path, rows: list[tuple]) -> None:
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
        con.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()


def make_default_fixtures(directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    specs = {
        "fake_a": [(1, "alice", 1.5), (2, "bob", None), (3, "carl", 3.5)],
        "fake_b": [(1, "dan", 4.0), (2, None, 5.0)],
        "fake_c": [(1, "erin", 6.0)],
    }
    paths = {}
    for name, rows in specs.items():
        path = directory / f"{name}.parquet"
        write_fixture(path, rows)
        paths[name] = path
    return paths


def make_multipart_geometry_fixtures(directory: str | Path) -> dict[str, Path]:
    """A minimal multipart-geometry scenario: a `contours` dimension where
    zone "Z1" has two polygon parts and "Z2" has one, and an already-joined
    `facts` Parquet (one logical fact per zone, expanded to one row per
    polygon part -- the shape `stage.py`'s naive `SELECT *` loader would
    otherwise reproduce unchanged in PostgreSQL).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    contour_path = directory / "contours.parquet"
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (zone_id VARCHAR, polygon_coords VARCHAR)")
        con.executemany(
            "INSERT INTO t VALUES (?, ?)",
            [("Z1", "part-Z1-a"), ("Z1", "part-Z1-b"), ("Z2", "part-Z2-a")],
        )
        con.execute(f"COPY t TO '{contour_path}' (FORMAT PARQUET)")
    finally:
        con.close()

    fact_path = directory / "facts.parquet"
    con = duckdb.connect()
    try:
        con.execute("CREATE TABLE t (zone_id VARCHAR, metric INTEGER, polygon_coords VARCHAR)")
        con.executemany(
            "INSERT INTO t VALUES (?, ?, ?)",
            [
                ("Z1", 10, "part-Z1-a"),
                ("Z1", 10, "part-Z1-b"),
                ("Z2", 20, "part-Z2-a"),
            ],
        )
        con.execute(f"COPY t TO '{fact_path}' (FORMAT PARQUET)")
    finally:
        con.close()

    return {"facts": fact_path, "contours": contour_path}
