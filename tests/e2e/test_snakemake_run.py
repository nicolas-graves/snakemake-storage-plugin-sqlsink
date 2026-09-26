"""The workflow through Snakemake itself, on a tiny fixture project.

Opt-in: set SNAKEMAKE_SQL_TEST_PG_DSN to a *disposable* PostgreSQL database
(its `public` and `analytics_storage` schemas are dropped before every test;
never point it at a shared database). The per-table storage-plugin path needs
PostgreSQL because the Snakemake main process would hold a DuckDB file's lock.
Datasets are published to PostgreSQL and to a DuckDB file.

Snakemake is run with `--scheduler greedy` (the ILP solver's bundled binary is
missing on some systems) and without `SNAKEMAKE_PROFILE`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import pytest
import yaml
from sqlalchemy import text

from sqlsink.engine import make_engine

DSN = os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN")
WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
SNAKEMAKE = Path(sys.executable).parent / "snakemake"

pytestmark = pytest.mark.skipif(
    not DSN or not SNAKEMAKE.exists(), reason="needs SNAKEMAKE_SQL_TEST_PG_DSN and a venv with snakemake"
)

CONFIG = {
    "tables": ["t1"],
    "sinks": ["postgres", "duckdb"],
    "duckdb_sink_path": "results/sink.duckdb",
    "loader_version": 1,
    "datasets": [
        {
            "name": "zones",
            "fact_source": "facts",
            "contour_source": "contours",
            "contour_table": "contours",
            "geometry_column": "polygon_coords",
            "fact_join_columns": ["zone_id"],
            "contour_join_columns": ["zone_id"],
            "compact_schema": "analytics_storage",
            "output_columns": ["zone_id", "metric", "polygon_coords"],
        },
        # Manifest v2 over the same Parquets: a fact and a shared dimension, joined.
        {
            "name": "zones_v2",
            "components": ["v2_zone_contours", {"name": "v2_facts", "kind": "fact", "primary_key": ["zone_id"], "source": "facts"}],
            "view": {
                "base": "v2_facts",
                "joins": [{"component": "v2_zone_contours", "on": {"zone_id": "zone_id"}}],
                "select": ["zone_id", "metric", {"column": "polygon_coords", "from": "v2_zone_contours"}],
            },
        },
    ],
    "components": [{"name": "v2_zone_contours", "kind": "dimension", "primary_key": ["zone_id"], "source": "contours"}],
}


def _write_parquet(project: Path, value: int) -> None:
    con = duckdb.connect()
    for name, select in {
        "t1": f"SELECT 1 a, 'x' b UNION ALL SELECT {value}, 'y'",
        "facts": f"SELECT 'Z1' zone_id, {value} metric UNION ALL SELECT 'Z2', 2",
        "contours": "SELECT 'Z1' zone_id, 'p1' polygon_coords UNION ALL SELECT 'Z2', 'p2'",
    }.items():
        con.execute(f"COPY ({select}) TO '{project / 'results' / 'parquet' / (name + '.parquet')}' (FORMAT PARQUET)")
    con.close()


def _wipe_postgres() -> None:
    engine = make_engine(DSN)
    with engine.begin() as conn:
        for schema in ("public", "analytics_storage"):
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        conn.execute(text('CREATE SCHEMA "public"'))
    engine.dispose()


class Project:
    def __init__(self, root: Path):
        self.root = root

    def run(self, *args: str) -> str:
        env = {k: v for k, v in os.environ.items() if k != "SNAKEMAKE_PROFILE"}
        proc = subprocess.run(
            [str(SNAKEMAKE), "-s", "workflow/Snakefile", "--configfile", "config/config.yaml", "-c2",
             "--scheduler", "greedy", "--resources=duckdb_writer=1", *args],
            cwd=self.root, env=env, capture_output=True, text=True, timeout=300,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        return out

    def state(self) -> dict:
        pg = make_engine(DSN)
        with pg.connect() as conn:
            t1 = conn.execute(text("SELECT a, b FROM t1 ORDER BY 1, 2")).fetchall()
            zones = conn.execute(text("SELECT zone_id, metric FROM zones ORDER BY 1")).fetchall()
            v2 = conn.execute(text("SELECT zone_id, metric, polygon_coords FROM zones_v2 ORDER BY 1")).fetchall()
        pg.dispose()
        sink = make_engine(f"duckdb:///{self.root / 'results' / 'sink.duckdb'}")
        with sink.connect() as conn:
            duck_zones = conn.execute(text("SELECT zone_id, metric FROM zones ORDER BY 1")).fetchall()
            duck_v2 = conn.execute(text("SELECT zone_id, metric, polygon_coords FROM zones_v2 ORDER BY 1")).fetchall()
        sink.dispose()
        return {
            "t1": [tuple(r) for r in t1],
            "zones": [tuple(r) for r in zones],
            "duck_zones": [tuple(r) for r in duck_zones],
            "zones_v2": [tuple(r) for r in v2],
            "duck_zones_v2": [tuple(r) for r in duck_v2],
        }

    def wipe_databases(self) -> None:
        _wipe_postgres()
        for path in (self.root / "results").glob("sink.duckdb*"):
            path.unlink()

    def expect(self, value: int) -> dict:
        return {
            "t1": [(1, "x"), (value, "y")],
            "zones": [("Z1", value), ("Z2", 2)],
            "duck_zones": [("Z1", value), ("Z2", 2)],
            "zones_v2": [("Z1", value, "p1"), ("Z2", 2, "p2")],
            "duck_zones_v2": [("Z1", value, "p1"), ("Z2", 2, "p2")],
        }


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "results" / "parquet").mkdir(parents=True)
    (tmp_path / "workflow").symlink_to(WORKFLOW)
    (tmp_path / "config" / "config.yaml").write_text(yaml.safe_dump({**CONFIG, "db": {"dsn": DSN}}))
    _wipe_postgres()
    _write_parquet(tmp_path, 1)
    proj = Project(tmp_path)
    proj.run()
    proj.run()  # settle: the first run recorded the datasets as "missing" (see `_published_state`)
    return proj


def test_a_first_run_publishes_tables_and_datasets_to_both_sinks(project):
    assert project.state() == project.expect(1)


def test_an_unchanged_rerun_does_nothing(project):
    assert "Nothing to be done" in project.run()


def test_a_changed_input_republishes_into_the_existing_databases(project):
    _write_parquet(project.root, 7)
    out = project.run()
    assert "Nothing to be done" not in out
    assert project.state() == project.expect(7)


def test_a_table_dropped_out_of_band_is_regenerated_from_the_cached_parquet(project):
    engine = make_engine(DSN)
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE t1"))
    engine.dispose()

    project.run()
    assert project.state() == project.expect(1)


def test_a_dataset_dropped_out_of_band_is_regenerated_from_the_cached_parquet(project):
    engine = make_engine(DSN)
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW zones"))
    engine.dispose()

    project.run()
    assert project.state() == project.expect(1)


def test_wiped_databases_are_rebuilt_and_then_settle(project):
    project.wipe_databases()

    project.run()
    assert project.state() == project.expect(1)

    project.run()  # one no-op re-stage: the recorded `published` param flips to "ok"
    assert "Nothing to be done" in project.run()


def test_a_published_dataset_can_be_exported_to_parquet(project):
    project.run("results/exports/postgres/zones.parquet", "results/exports/duckdb/zones.parquet")
    for sink in ("postgres", "duckdb"):
        rows = duckdb.sql(f"SELECT * FROM '{project.root / 'results' / 'exports' / sink / 'zones.parquet'}' ORDER BY 1").fetchall()
        assert rows == [("Z1", 1, "p1"), ("Z2", 2, "p2")]
