"""One materialization API over the SQL sinks.

A dataset is reduced once to a *normalized representation* -- compact facts
plus shared contours, both expressed as DuckDB queries over the source
Parquet (`NormalizedDataset`). A `Sink` decides how that representation is
persisted and made visible: SQL tables + a compatibility view in PostgreSQL
or a DuckDB file (`sink_postgres.SqlSink`). Parquet is an export of a
published dataset (`export.export_parquet`), not a sink.
Callers only ever talk to `stage` / `publish` / `materialize`; the
differences between databases (transport, locking) live inside the sink.

A sink receives a DuckDB *query*, not rows: PostgreSQL streams its cursor
through a bulk loader (DuckDB through batched inserts).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import duckdb

from .fingerprint import compute_dataset_update_id, sha256_file
from .manifest import DatasetMaterialization
from .queries import compact_select_sql, contour_select_sql


class OrphanFactsError(ValueError):
    """Some fact join keys have no matching contour row. The compatibility
    join is INNER, so those facts would silently disappear."""


@dataclass(frozen=True)
class ContourSource:
    """The shared contour relation of a dataset, independent of any one
    dataset's facts (several datasets may share one contour table)."""

    manifest: DatasetMaterialization
    path: str
    sha256: str

    @property
    def query(self) -> str:
        return contour_select_sql(self.manifest, self.path)


@dataclass(frozen=True)
class NormalizedDataset:
    """One logical dataset in normalized form. `update_id` identifies "this
    dataset, from these source files, under this manifest, with this
    loader": it changes when the facts, the contours or the manifest do."""

    manifest: DatasetMaterialization
    fact_path: str
    fact_sha256: str
    contour: ContourSource
    update_id: str

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def compact_query(self) -> str:
        return compact_select_sql(self.manifest, self.fact_path)

    @property
    def contour_query(self) -> str:
        return self.contour.query


def contour_source(manifest: DatasetMaterialization, contour_parquet_path: str) -> ContourSource:
    return ContourSource(manifest, contour_parquet_path, sha256_file(contour_parquet_path))


def normalize(
    manifest: DatasetMaterialization, fact_parquet_path: str, contour_parquet_path: str
) -> NormalizedDataset:
    """Fingerprint the two source Parquets and bind them to the manifest.
    Reads the files once to hash them; nothing is written anywhere."""
    contour = contour_source(manifest, contour_parquet_path)
    fact_sha256 = sha256_file(fact_parquet_path)
    return NormalizedDataset(
        manifest=manifest,
        fact_path=fact_parquet_path,
        fact_sha256=fact_sha256,
        contour=contour,
        update_id=compute_dataset_update_id(manifest.manifest_hash(), fact_sha256, contour.sha256),
    )


class Sink(Protocol):
    """Where a normalized dataset goes. Receipts are objects exposing
    `status` (`"current"` | `"staged"`), `row_count` and `to_dict()`;
    `publish` takes them as dicts, as stored between Snakemake rules."""

    name: str
    # True when the sink keeps the source column types exactly (files);
    # False when it maps them through another type system (SQL database).
    preserves_types: bool

    def spill_dir(self) -> str:
        """Directory on a disk (not RAM) where DuckDB may spill."""

    def current_update_id(self, dataset_name: str) -> str | None:
        """`update_id` of the published version, or None."""

    def published_intact(self, manifest: DatasetMaterialization) -> bool:
        """True if the objects of the published version are all still there
        (a marker alone survives a table dropped out of band)."""

    def stage_contours(self, contour: ContourSource, con: duckdb.DuckDBPyConnection): ...

    def stage_facts(self, dataset: NormalizedDataset, con: duckdb.DuckDBPyConnection): ...

    def publish(
        self,
        manifests: list[DatasetMaterialization],
        dataset_receipts: list[dict],
        contour_receipts: list[dict],
    ) -> list[str]:
        """Make every staged receipt visible, all-or-nothing per sink.
        Returns the names of the datasets actually published."""

    def joined_relation(self, manifest: DatasetMaterialization, con: duckdb.DuckDBPyConnection) -> str:
        """Make the published, collapsed dataset readable from `con` and
        return the DuckDB relation expression addressing it."""


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@contextlib.contextmanager
def duckdb_session(
    threads: int | None, memory_limit: str | None, spill_dir: str
) -> Iterator[duckdb.DuckDBPyConnection]:
    """DuckDB connection that can spill to disk. An in-memory connection has
    no temp directory by default, so a large ORDER BY over wide geometry
    strings dies at `memory_limit` instead of spilling."""
    spill = Path(spill_dir) / f".duckdb_tmp.{os.getpid()}"
    spill.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET temp_directory = {_sql_string(str(spill))}")
        if threads is not None:
            con.execute(f"SET threads = {int(threads)}")
        if memory_limit is not None:
            con.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
        yield con
    finally:
        con.close()
        shutil.rmtree(spill, ignore_errors=True)


def orphan_count(con: duckdb.DuckDBPyConnection, dataset: NormalizedDataset) -> int:
    """Distinct compact fact rows whose join key has no contour."""
    manifest = dataset.manifest
    on = " AND ".join(
        f'f."{fc}" = c."{cc}"'
        for fc, cc in zip(manifest.fact_join_columns, manifest.contour_join_columns)
    )
    sql = (
        f"SELECT COUNT(*) FROM ({dataset.compact_query}) f "
        f"ANTI JOIN ({dataset.contour_query}) c ON {on}"
    )
    return con.execute(sql).fetchone()[0]


def check_no_orphans(con: duckdb.DuckDBPyConnection, dataset: NormalizedDataset) -> None:
    orphans = orphan_count(con, dataset)
    if orphans:
        raise OrphanFactsError(
            f"{dataset.name}: {orphans} distinct compact fact row(s) have no "
            f"matching contour on {list(dataset.manifest.fact_join_columns)}"
        )


@dataclass
class StageResult:
    contour: object
    dataset: object


def stage(
    dataset: NormalizedDataset,
    sink: Sink,
    *,
    threads: int | None = 2,
    memory_limit: str | None = None,
) -> StageResult:
    """Stage one dataset (contours, then facts) into `sink`.

    Nothing becomes visible until `publish`. A dataset whose `update_id` is
    already the sink's published version (and its objects are all still there) is not re-checked: its receipts come
    back with status "current". Otherwise orphan facts are rejected *before*
    the sink writes anything, so no sink silently loses rows to the INNER
    join.
    """
    is_current = sink.current_update_id(dataset.name) == dataset.update_id and sink.published_intact(dataset.manifest)
    with duckdb_session(threads, memory_limit, sink.spill_dir()) as con:
        if not is_current:
            check_no_orphans(con, dataset)
        contour_receipt = sink.stage_contours(dataset.contour, con)
        dataset_receipt = sink.stage_facts(dataset, con)
    return StageResult(contour=contour_receipt, dataset=dataset_receipt)


def publish(sink: Sink, manifests: list[DatasetMaterialization], results: list[StageResult]) -> list[str]:
    return sink.publish(
        manifests,
        [r.dataset.to_dict() for r in results],
        [r.contour.to_dict() for r in results],
    )


def materialize(
    manifest: DatasetMaterialization,
    fact_parquet_path: str,
    contour_parquet_path: str,
    sink: Sink,
    *,
    threads: int | None = 2,
    memory_limit: str | None = None,
) -> tuple[list[str], StageResult]:
    """Stage and publish a single dataset. Returns (published names, result)."""
    dataset = normalize(manifest, fact_parquet_path, contour_parquet_path)
    result = stage(dataset, sink, threads=threads, memory_limit=memory_limit)
    return publish(sink, [manifest], [result]), result


def make_sink(spec: dict) -> Sink:
    """Build a sink from a config mapping: `{"type": "postgres", "dsn": ...}`
    or `{"type": "duckdb", "path": ...}`."""
    kind = spec.get("type")
    if kind in ("postgres", "duckdb"):
        from .engine import make_engine
        from .sink_postgres import SqlSink

        if kind == "postgres":
            return SqlSink(make_engine(spec["dsn"]))
        from .metadata import create_all

        engine = make_engine(f"duckdb:///{spec['path']}")
        create_all(engine)  # a DuckDB file is created on first use: it needs its marker tables
        return SqlSink(engine)
    raise ValueError(f"unknown sink type {kind!r} (expected 'postgres' or 'duckdb')")


def dataset_is_published(spec: dict, manifest: DatasetMaterialization) -> bool:
    """Read-only: is `manifest`'s dataset published in the sink `spec` with
    all its objects still in place? Snakemake evaluates this while building
    the DAG (as a rule param), so a dataset dropped or wiped out of band
    changes the param and re-runs its staging. The engine is disposed before
    returning: a DuckDB file lock must not outlive the check."""
    if spec.get("type") == "duckdb" and not os.path.exists(spec["path"]):
        return False
    sink = make_sink(spec)
    try:
        return sink.current_update_id(manifest.name) is not None and sink.published_intact(manifest)
    finally:
        sink.engine.dispose()


def default_spill_dir() -> str:
    """Spill location for sinks with no directory of their own. Overridable
    because /tmp is often RAM-backed, where spilling defeats the purpose."""
    return os.environ.get("SNAKEMAKE_SQL_SPILL_DIR") or tempfile.gettempdir()
