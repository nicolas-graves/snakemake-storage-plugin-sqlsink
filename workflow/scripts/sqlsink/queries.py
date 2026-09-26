"""Pure SQL builders for the normalized representation of a dataset.

Nothing here touches a database. Both sinks (PostgreSQL, Parquet) build
their output from the same two DuckDB queries -- compact facts and shared
contours -- and describe the join back to the collapsed dataset with the
same `JoinSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb

from .join import physical_join_spec, render_join_sql
from .manifest import PART_COLUMN, DatasetMaterialization


def fetch_row(con: duckdb.DuckDBPyConnection, sql: str) -> tuple[Any, ...]:
    """The single row of an aggregate query (which always returns one)."""
    row = con.execute(sql).fetchone()
    assert row is not None, f"query returned no row: {sql}"
    return row


def read_parquet_sql(path: str) -> str:
    """`read_parquet('<path>')` with the path safely quoted as a SQL string."""
    escaped = str(path).replace("'", "''")
    return f"read_parquet('{escaped}')"


@dataclass(frozen=True, eq=False)
class ArrowSource:
    """An in-memory Arrow table used where a Parquet path would be. It is
    registered on the DuckDB connection under `name` before any query that
    reads it runs (`bind_sources`)."""

    table: object  # pyarrow.Table
    name: str

    @property
    def sql(self) -> str:
        return '"' + self.name.replace('"', '""') + '"'


@dataclass(frozen=True, eq=False)
class SinkTableSource:
    """A plain table already published in the sink's own database (by
    `stage_table` / `publish_tables`) used as the source of a component. Its
    fingerprint is the publish marker's `update_id`. It is bound to a DuckDB
    session by `bind`: read through DuckDB's postgres scanner on PostgreSQL
    (nothing is copied), or, for a DuckDB sink file that the engine holds open,
    copied into the session as one Arrow table (so keep such sources small)."""

    table: str
    sha256: str
    engine: Any
    schema: str | None = None

    @property
    def sql(self) -> str:
        if self.engine.dialect.name == "postgresql":
            return f'pg."{self.schema or "public"}"."{self.table}"'
        return '"' + f"__sinktable__{self.table}" + '"'

    def bind(self, con) -> None:
        if self.engine.dialect.name == "postgresql":
            from .sink_postgres import pg_attach

            attached = con.execute("SELECT count(*) FROM duckdb_databases() WHERE database_name = 'pg'").fetchone()[0]
            if not attached:
                pg_attach(con, self.engine.url)
            return
        with self.engine.connect() as conn:
            cur = conn.connection.driver_connection.cursor()
            cur.execute(f'SELECT * FROM "{self.table}"')
            table = cur.to_arrow_table() if hasattr(cur, "to_arrow_table") else cur.fetch_arrow_table()
        con.register(f"__sinktable__{self.table}", table)


def source_sql(source) -> str:
    """The `FROM` expression of a source: `read_parquet('<path>')` for a
    path, the registered relation name for an `ArrowSource`, the sink relation
    of a `SinkTableSource`."""
    if isinstance(source, (ArrowSource, SinkTableSource)):
        return source.sql
    return read_parquet_sql(source)


def bind_sources(con, *sources) -> None:
    """Register every in-memory source on `con` (paths need nothing)."""
    for source in sources:
        if isinstance(source, ArrowSource):
            con.register(source.name, source.table)
        elif isinstance(source, SinkTableSource):
            source.bind(con)


def compact_select_sql(manifest: DatasetMaterialization, fact_parquet_path) -> str:
    """DuckDB query recovering the logical facts: geometry column projected
    away, remaining columns deduplicated. Idempotent on an already-compact
    Parquet (same columns, no duplicates)."""
    compact_cols = ", ".join(f'"{c}"' for c in manifest.compact_columns())
    return f"SELECT DISTINCT {compact_cols} FROM {source_sql(fact_parquet_path)}"


def contour_select_sql(manifest: DatasetMaterialization, contour_parquet_path) -> str:
    """DuckDB query for the shared contour relation.

    Project down to (join columns + geometry column) and DISTINCT those,
    rather than the raw row. Real contour Parquets have been observed to
    carry near-duplicate rows per geometry part -- same join key, same
    geometry, but a differently-spelled label in some other column
    (whitespace/accent variants) -- so a whole-row DISTINCT does not
    collapse them and the compatibility view's JOIN fans out further than
    the original Parquet did. The other output columns for a dataset
    already come from the fact table (see view_select_sql), so the contour
    table only needs to carry whatever the join and the geometry require.
    """
    projected = ", ".join(f'"{c}"' for c in contour_columns(manifest))
    return f"SELECT DISTINCT {projected} FROM {source_sql(contour_parquet_path)}"


def contour_columns(manifest: DatasetMaterialization) -> list[str]:
    """Columns of the contour relation: join columns then the geometry."""
    return list(dict.fromkeys((*manifest.contour_join_columns, manifest.geometry_column)))


def view_select_sql(manifest: DatasetMaterialization, dialect_name: str) -> str:
    """The compatibility view body: an explicit select list (original
    column names and order), joining the compact fact table back to the
    shared contour table on the manifest's declared join columns."""
    return render_join_sql(physical_join_spec(manifest, dialect_name), dialect_name)


def parts_select_sql(manifest: DatasetMaterialization, contour_parquet_path) -> str:
    """Keyed form of the contour: one row per geometry part with a
    deterministic `part_no` (rank of the polygon text within its zone), so the
    key is stable across runs and the staged bytes are reproducible."""
    key = ", ".join(f'"{c}"' for c in manifest.contour_join_columns)
    geometry = f'"{manifest.geometry_column}"'
    return (
        f"SELECT {key}, {geometry}, "
        f"CAST(ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY {geometry}) AS INTEGER) AS {PART_COLUMN} "
        f"FROM ({contour_select_sql(manifest, contour_parquet_path)})"
    )


def zone_select_sql(manifest: DatasetMaterialization, contour_parquet_path) -> str:
    """One row per zone: the distinct join key of the contour."""
    key = ", ".join(f'"{c}"' for c in manifest.contour_join_columns)
    return f"SELECT DISTINCT {key} FROM ({contour_select_sql(manifest, contour_parquet_path)})"
