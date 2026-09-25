"""Low-level Parquet writers for the normalized components.

This module holds the Parquet writing pieces: atomic `COPY ... TO` of a
DuckDB query, the sort orders that make component bytes reproducible, and
`export_joined_parquet`, which derives the collapsed joined Parquet from the
two components with the same `join.render_join_sql` the PostgreSQL
compatibility view uses.

`write_components` is the standalone one-shot form (joined Parquet ->
compact + contour files under a directory); the sink API supersedes it for
pipeline use.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from . import metadata as meta_mod
from .fingerprint import sha256_file
from .join import Relation, join_spec, render_join_sql
from .manifest import DatasetMaterialization
from .queries import contour_columns, fetch_row, read_parquet_sql
from .sink import OrphanFactsError, check_no_orphans, duckdb_session, normalize  # noqa: F401


@dataclass
class ComponentReceipt:
    dataset: str
    compact_path: str
    contour_path: str
    compact_sha256: str
    contour_sha256: str
    compact_rows: int
    contour_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def compact_parquet_path(manifest: DatasetMaterialization, out_dir: str) -> str:
    return str(Path(out_dir) / f"{meta_mod.compact_table_name(manifest.name)}.parquet")


def contour_parquet_path(manifest: DatasetMaterialization, out_dir: str) -> str:
    return str(Path(out_dir) / f"{manifest.contour_table}.parquet")


def _copy_atomic(
    con: duckdb.DuckDBPyConnection, query: str, out_path: str, options: str = ""
) -> int:
    """`COPY (query) TO out_path` through a temporary sibling file, renamed
    into place only once complete. Returns the number of rows written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp")  # unique: sibling jobs may share a directory
    escaped = str(tmp).replace("'", "''")
    try:
        con.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD{options})")
        rows = fetch_row(con, f"SELECT COUNT(*) FROM {read_parquet_sql(str(tmp))}")[0]
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return rows


def _order_by_all(query: str, columns: list[str]) -> str:
    # A total order over DISTINCT rows makes the file bytes (hence its
    # sha256, hence the dataset update_id) reproducible across runs.
    order = ", ".join(f'"{c}"' for c in columns)
    return f"SELECT * FROM ({query}) ORDER BY {order}"


def _sort_columns(manifest: DatasetMaterialization) -> list[str]:
    """Compact-table sort order: join key first (so the derived join emits
    rows grouped by key), then every other column for a total order."""
    keys = list(manifest.fact_join_columns)
    return keys + [c for c in manifest.compact_columns() if c not in keys]


def write_components(
    manifest: DatasetMaterialization,
    fact_parquet_path: str,
    contour_parquet_path_in: str,
    out_dir: str,
    *,
    threads: int | None = 2,
    memory_limit: str | None = None,
) -> ComponentReceipt:
    """Normalize a joined fact Parquet + contour Parquet into the two
    component Parquets under `out_dir`.

    Raises `OrphanFactsError` (writing nothing) if any fact key has no
    contour: the derived join would drop those rows.
    """
    dataset = normalize(manifest, fact_parquet_path, contour_parquet_path_in)
    compact_query, contour_query = dataset.compact_query, dataset.contour_query

    with duckdb_session(threads, memory_limit, out_dir) as con:
        check_no_orphans(con, dataset)
        compact_out = compact_parquet_path(manifest, out_dir)
        contour_out = contour_parquet_path(manifest, out_dir)
        compact_rows = _copy_atomic(
            con, _order_by_all(compact_query, _sort_columns(manifest)), compact_out
        )
        contour_rows = _copy_atomic(
            con, _order_by_all(contour_query, contour_columns(manifest)), contour_out
        )

    return ComponentReceipt(
        dataset=manifest.name,
        compact_path=compact_out,
        contour_path=contour_out,
        compact_sha256=sha256_file(compact_out),
        contour_sha256=sha256_file(contour_out),
        compact_rows=compact_rows,
        contour_rows=contour_rows,
    )


def joined_select_sql(
    manifest: DatasetMaterialization, compact_parquet: str, contour_parquet: str
) -> str:
    """The join, rendered for DuckDB over the two component Parquets."""
    spec = join_spec(
        manifest,
        Relation(sql=read_parquet_sql(compact_parquet)),
        Relation(sql=read_parquet_sql(contour_parquet)),
    )
    return render_join_sql(spec, "duckdb")


def export_joined_parquet(
    manifest: DatasetMaterialization,
    compact_parquet: str,
    contour_parquet: str,
    out_path: str,
    *,
    threads: int | None = 2,
    memory_limit: str | None = None,
    row_group_size: int = 10_000,
) -> int:
    """Derive the collapsed joined Parquet (original column names and order)
    from the components. Returns the number of rows written.

    No global sort: the join re-expands the geometry (hundreds of thousands
    of rows x tens of KB), so an ORDER BY over it cannot fit in memory.
    Grouping by key comes from the compact component being sorted by key
    (the join preserves the probe side's order). Row groups are kept small
    so DuckDB never buffers a whole expanded group; row order is otherwise
    unspecified and consumers / the proof treat the file as a multiset.
    """
    query = joined_select_sql(manifest, compact_parquet, contour_parquet)
    with duckdb_session(threads, memory_limit, str(Path(out_path).parent)) as con:
        return _copy_atomic(con, query, out_path, f", ROW_GROUP_SIZE {int(row_group_size)}")
