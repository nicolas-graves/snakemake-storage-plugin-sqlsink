"""Per-table staging: compare against the marker, load into staging if stale.

Never touches the public table or the marker row — writing the marker is
coupled only to `publish.publish_tables`, so an interrupted stage run can
never make the system believe a table is published when it isn't.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import duckdb
from sqlalchemy import Column, MetaData, Table, inspect, select

from . import metadata as meta_mod
from .engine import bulk_load, upsert_by_pk
from .fingerprint import LOADER_VERSION, TYPE_MAP_VERSION, compute_update_id_for_file


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
    con = duckdb.connect()
    try:
        rel = con.sql(f"SELECT * FROM read_parquet('{parquet_path}')")
        columns = list(zip(rel.columns, rel.types))
        row_count = con.sql(
            f"SELECT count(*) FROM read_parquet('{parquet_path}')"
        ).fetchone()[0]
        null_exprs = ", ".join(
            f'count(*) FILTER (WHERE "{name}" IS NULL) AS "{name}"' for name, _ in columns
        )
        null_row = con.sql(
            f"SELECT {null_exprs} FROM read_parquet('{parquet_path}')"
        ).fetchone()
        null_counts = dict(zip((name for name, _ in columns), null_row)) if columns else {}
        return columns, row_count, null_counts
    finally:
        con.close()


_DUCKDB_TO_SA = {
    "BIGINT": "BigInteger",
    "INTEGER": "Integer",
    "DOUBLE": "Float",
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
    with engine.connect() as conn:
        stmt = select(meta_mod.analytics_table_updates).where(
            meta_mod.analytics_table_updates.c.table_name == table_name
        )
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None


_fetch_marker = fetch_marker


def fetch_staged_marker(engine, table_name: str) -> dict | None:
    """Public: the staging-marker row for `table_name` ("has this exact
    fingerprint already been staged"), or None if never staged.
    """
    with engine.connect() as conn:
        stmt = select(meta_mod.staged_table_updates).where(
            meta_mod.staged_table_updates.c.table_name == table_name
        )
        row = conn.execute(stmt).mappings().first()
        return dict(row) if row else None


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
        con = duckdb.connect()
        try:
            rows = con.sql(f"SELECT * FROM read_parquet('{parquet_path}')").fetchall()
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
