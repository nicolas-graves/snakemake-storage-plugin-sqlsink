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


def write_table(path: str | Path, ddl: str, rows: list[tuple]) -> Path:
    """A Parquet file from `CREATE TABLE t (<ddl>)` and rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"CREATE TABLE t ({ddl})")
        if rows:
            marks = ", ".join("?" for _ in rows[0])
            con.executemany(f"INSERT INTO t VALUES ({marks})", rows)
        con.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")
    finally:
        con.close()
    return path


def make_transitions_fixtures(directory: str | Path) -> dict[str, Path]:
    """A toy of the transitions dataset: native-grain `fact`, a `bridge`
    (anchor -> f21) and `sector_paths` (region x f21 -> path + a metric).

    Anchor A reaches both F1 and F2, and both lead to path P in region R1, with
    different metrics. The bridge `priority` ranks F2 before F1, against the
    alphabetical order of the f21 codes: a view that kept the smallest f21
    instead of the first bridge row would show metric 1.0, not 2.0."""
    directory = Path(directory)
    return {
        "fact": write_table(
            directory / "fact.parquet",
            "obs_id BIGINT, region VARCHAR, anchor VARCHAR, flow DOUBLE",
            [(1, "R1", "A", 10.5), (2, "R1", "B", 20.25), (3, "R2", "A", None)],
        ),
        "bridge": write_table(
            directory / "bridge.parquet",
            "anchor VARCHAR, f21 VARCHAR, priority INTEGER",
            [("A", "F2", 1), ("A", "F1", 2), ("B", "F1", 1)],
        ),
        "sector_paths": write_table(
            directory / "sector_paths.parquet",
            "region VARCHAR, f21 VARCHAR, path VARCHAR, metric DOUBLE",
            [
                ("R1", "F1", "P", 1.0),
                ("R1", "F2", "P", 2.0),
                ("R1", "F1", "Q", 3.0),
                ("R2", "F1", "P", 4.0),
            ],
        ),
    }


# The portable model case for `view_sql`: an ALL branch UNION ALL the rows
# reached through bridge and paths, one per (observation, path), the bridge's
# explicit `priority` breaking ties.
TRANSITIONS_VIEW_SQL = """
SELECT obs_id, region, flow, 'ALL' AS path, CAST(NULL AS DOUBLE PRECISION) AS metric FROM {fact}
UNION ALL
SELECT obs_id, region, flow, path, metric FROM (
  SELECT f.obs_id, f.region, f.flow, p.path, p.metric,
         ROW_NUMBER() OVER (PARTITION BY f.obs_id, p.path ORDER BY b.priority) AS rn
  FROM {fact} f
  JOIN {bridge} b ON b.anchor = f.anchor
  JOIN {sector_paths} p ON p.region = f.region AND p.f21 = b.f21
) t
WHERE rn = 1
"""

TRANSITIONS_EXPECTED = sorted(
    [
        (1, "R1", 10.5, "ALL", None),
        (2, "R1", 20.25, "ALL", None),
        (3, "R2", None, "ALL", None),
        (1, "R1", 10.5, "P", 2.0),  # F2 first by priority (F1 would give 1.0)
        (1, "R1", 10.5, "Q", 3.0),
        (2, "R1", 20.25, "P", 1.0),
        (2, "R1", 20.25, "Q", 3.0),
        (3, "R2", None, "P", 4.0),
    ],
    key=lambda r: (r[0], r[3]),
)
