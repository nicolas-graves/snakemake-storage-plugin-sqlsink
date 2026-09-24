"""Compatibility facade for the PostgreSQL-only entry points.

The logic now lives behind the sink API: query builders in `queries`, the
normalized representation and orchestration in `sink`, and the PostgreSQL
staging in `sink_postgres.SqlSink`. `stage_contours` / `stage_dataset`
keep their original signatures and receipts, so existing callers and the
receipts stored between Snakemake rules are unchanged.

They stage only, and skip the orphan-fact check of `sink.stage`; new code
should go through `sink.materialize` / `sink.stage` instead.
"""

from __future__ import annotations

from .manifest import DatasetMaterialization
from .queries import (  # noqa: F401 (re-exported)
    compact_select_sql,
    contour_select_sql,
    read_parquet_sql,
    view_select_sql,
)
from .sink import contour_source, duckdb_session, normalize
from .sink_postgres import (  # noqa: F401 (re-exported)
    ContourStageReceipt,
    DatasetStageReceipt,
    SqlSink,
    fetch_contour_marker,
    fetch_dataset_marker,
    fetch_staged_contour_marker,
    fetch_staged_dataset_marker,
)


def stage_contours(engine, manifest: DatasetMaterialization, contour_parquet_path: str) -> ContourStageReceipt:
    """Stage the shared contour relation into the SQL database."""
    sink = SqlSink(engine)
    with duckdb_session(None, None, sink.spill_dir()) as con:
        return sink.stage_contours(contour_source(manifest, contour_parquet_path), con)


def stage_dataset(
    engine,
    manifest: DatasetMaterialization,
    fact_parquet_path: str,
    contour_parquet_path: str,
) -> DatasetStageReceipt:
    """Stage the compact fact table of one dataset into the SQL database."""
    sink = SqlSink(engine)
    dataset = normalize(manifest, fact_parquet_path, contour_parquet_path)
    with duckdb_session(None, None, sink.spill_dir()) as con:
        return sink.stage_facts(dataset, con)
