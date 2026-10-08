"""Per-table staging: compare against the marker, load into staging if stale.

Never touches the public table or the marker row — writing the marker is
coupled only to `publish.publish_tables`, so an interrupted stage run can
never make the system believe a table is published when it isn't.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

from sqlalchemy import Column, MetaData, Table, inspect

from . import metadata as meta_mod
from ._duckdb import require_duckdb
from .engine import bulk_load, upsert_by_pk
from .fingerprint import LOADER_VERSION, TYPE_MAP_VERSION, compute_update_id_for_file
from .queries import fetch_row, read_parquet_sql
from .sqlident import quote_ident


@dataclass
class StageReceipt:
    table: str
    status: str  # "current" | "staged"
    update_id: str
    parquet_sha256: str
    row_count: int
    null_counts: dict
    staging_table: str | None
    staged_at: str
    marker_update_id_at_stage_time: str | None
    loader_version: int = LOADER_VERSION
    type_map_version: int = TYPE_MAP_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _read_parquet_schema_and_stats(parquet_path: str) -> tuple[list[tuple[str, str]], int, dict]:
    duckdb = require_duckdb()
    con = duckdb.connect()
    try:
        rel = con.sql(f"SELECT * FROM {read_parquet_sql(parquet_path)}")
        columns = [(name, str(typ)) for name, typ in zip(rel.columns, rel.types)]
        row_count = fetch_row(con, f"SELECT count(*) FROM {read_parquet_sql(parquet_path)}")[0]
        # Column names come from the file: quote them, never splice them in.
        null_exprs = ", ".join(
            f"count(*) FILTER (WHERE {quote_ident(name)} IS NULL) AS {quote_ident(name)}" for name, _ in columns
        )
        null_row = fetch_row(con, f"SELECT {null_exprs} FROM {read_parquet_sql(parquet_path)}")
        null_counts = dict(zip((name for name, _ in columns), null_row)) if columns else {}
        return columns, row_count, null_counts
    finally:
        con.close()


# `sa.Float` renders FLOAT, which is 4 bytes on DuckDB (8 on PostgreSQL): a
# DOUBLE must map to `sa.Double`, or a DuckDB sink silently rounds to float32.
_DUCKDB_TO_SA = {
    "BIGINT": "BigInteger",
    "INTEGER": "Integer",
    "SMALLINT": "SmallInteger",
    "TINYINT": "SmallInteger",
    "DOUBLE": "Double",
    "FLOAT": "REAL",
    "VARCHAR": "Text",
    "BOOLEAN": "Boolean",
    "DATE": "Date",
    "TIMESTAMP": "DateTime",
}


def _staging_table_object(table_name: str, columns: list[tuple[str, str]]) -> Table:
    import sqlalchemy as sa

    md = MetaData()
    cols = []
    for name, duck_type in columns:
        sa_type_name = _DUCKDB_TO_SA.get(str(duck_type), "Text")
        sa_type = getattr(sa, sa_type_name)
        cols.append(Column(name, sa_type))
    return Table(meta_mod.staging_name(table_name), md, *cols)


def fetch_marker(engine, table_name: str) -> dict | None:
    """Public: the marker row for `table_name`, or None if never published."""
    return meta_mod.fetch_one(engine, meta_mod.analytics_table_updates, "table_name", table_name)


_fetch_marker = fetch_marker


def fetch_staged_marker(engine, table_name: str) -> dict | None:
    """Public: the staging-marker row for `table_name` ("has this exact
    fingerprint already been staged"), or None if never staged.
    """
    return meta_mod.fetch_one(engine, meta_mod.staged_table_updates, "table_name", table_name)


def is_current(engine, table_name: str, parquet_path: str, extra_config: dict | None = None) -> bool:
    """Cheap, read-only freshness check (file hash + marker lookups + a table
    existence check, no DuckDB read, no writes). True if this exact Parquet
    fingerprint has already been published (and its table is still there) or
    staged (and its staging table is still there). Used by the storage
    plugin's `exists()`, which must never mutate anything, and must be able
    to report True right after staging -- before publish ever runs --
    without that meaning "published".

    A marker alone is not enough: a table dropped or restored out of band
    leaves its marker behind, and trusting it would keep Snakemake from ever
    regenerating the table.
    """
    update_id, _ = compute_update_id_for_file(table_name, parquet_path, extra_config)
    inspector = inspect(engine)
    marker = _fetch_marker(engine, table_name)
    if marker is not None and marker["update_id"] == update_id and inspector.has_table(table_name):
        return True
    staged = fetch_staged_marker(engine, table_name)
    return (
        staged is not None
        and staged["update_id"] == update_id
        and inspector.has_table(meta_mod.staging_name(table_name))
    )


def stage_table(engine, table_name: str, parquet_path: str, extra_config: dict | None = None) -> StageReceipt:
    update_id, parquet_sha256 = compute_update_id_for_file(table_name, parquet_path, extra_config)
    # Staging is a write path: the marker tables are created here, never by a
    # read-only existence check.
    meta_mod.create_all(engine)
    marker = _fetch_marker(engine, table_name)
    inspector = inspect(engine)

    if marker is not None and marker["update_id"] == update_id and inspector.has_table(table_name):
        return StageReceipt(
            table=table_name,
            status="current",
            update_id=update_id,
            parquet_sha256=parquet_sha256,
            row_count=marker["row_count"],
            null_counts={},
            staging_table=None,
            staged_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            marker_update_id_at_stage_time=marker["update_id"],
        )

    staged = fetch_staged_marker(engine, table_name)
    if (
        staged is not None
        and staged["update_id"] == update_id
        and inspector.has_table(meta_mod.staging_name(table_name))
    ):
        # Already staged with this exact fingerprint (e.g. a prior
        # retrieve_object() call in this same run) -- no need to redo the
        # DuckDB read and bulk load.
        return StageReceipt(
            table=table_name,
            status="staged",
            update_id=update_id,
            parquet_sha256=parquet_sha256,
            row_count=staged["row_count"],
            null_counts={},
            staging_table=meta_mod.staging_name(table_name),
            staged_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            marker_update_id_at_stage_time=marker["update_id"] if marker else None,
        )

    columns, row_count, null_counts = _read_parquet_schema_and_stats(parquet_path)
    staging_table = _staging_table_object(table_name, columns)

    with engine.begin() as conn:
        staging_table.drop(conn, checkfirst=True)
        staging_table.create(conn)
        con = require_duckdb().connect()
        try:
            rows = con.sql(f"SELECT * FROM {read_parquet_sql(parquet_path)}").fetchall()
            col_names = [c[0] for c in columns]
        finally:
            con.close()
        bulk_load(conn, staging_table, [dict(zip(col_names, r)) for r in rows])
        upsert_by_pk(
            conn,
            meta_mod.staged_table_updates,
            "table_name",
            {
                "table_name": table_name,
                "update_id": update_id,
                "parquet_sha256": parquet_sha256,
                "row_count": row_count,
            },
            now_col="staged_at",
        )

    return StageReceipt(
        table=table_name,
        status="staged",
        update_id=update_id,
        parquet_sha256=parquet_sha256,
        row_count=row_count,
        null_counts=null_counts,
        staging_table=meta_mod.staging_name(table_name),
        staged_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        marker_update_id_at_stage_time=marker["update_id"] if marker else None,
    )
