"""Parquet export of a published dataset.

Parquet is an export format here, not a sink: the file is written from the
version the database currently publishes (never from the source files), so it
is by construction what consumers of the database would read.
"""

from __future__ import annotations

from .components import _copy_atomic
from .manifest import DatasetMaterialization
from .sink import Sink, duckdb_session


class DatasetNotPublished(LookupError):
    """The dataset has no published version in the sink, so there is nothing to export."""


def export_parquet(
    sink: Sink,
    manifest: DatasetMaterialization,
    path: str,
    *,
    threads: int | None = 2,
    memory_limit: str | None = None,
    row_group_size: int = 10_000,
) -> int:
    """Write the published joined relation of `manifest` to `path` as Parquet,
    atomically (temporary sibling, renamed into place). Returns the row count.

    Row order is unspecified; consumers treat the file as a multiset. Small
    row groups keep DuckDB from buffering a whole expanded geometry group.
    """
    if sink.current_update_id(manifest.name) is None:
        raise DatasetNotPublished(f"dataset {manifest.name!r} has not been published; nothing to export")
    with duckdb_session(threads, memory_limit, sink.spill_dir()) as con:
        relation = sink.joined_relation(manifest, con)
        return _copy_atomic(con, f"SELECT * FROM {relation}", path, f", ROW_GROUP_SIZE {int(row_group_size)}")
