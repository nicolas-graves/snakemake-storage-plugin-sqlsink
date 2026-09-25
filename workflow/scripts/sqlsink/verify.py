"""Multiset-equivalence check between an original expanded Parquet and the
published PostgreSQL compatibility view, within a bounded memory budget.

Both relations are reduced to (row count, sum of per-row hashes), which is
order-independent and sensitive to multiplicities. The work is split into
batches over the distinct values of one key column so that peak memory stays
small: aggregating a whole multi-million-row view through DuckDB's postgres
scanner in one query was observed to use ~10 GB, ignoring `memory_limit`.
Per-row hashes are computed from individually cast columns (not from one
concatenated string), which keeps huge geometry strings from being copied.
A checksum collision is theoretically possible but negligible at this scale.
"""

from __future__ import annotations

import resource
from dataclasses import dataclass, field

import duckdb

from .manifest import DatasetMaterialization
from .queries import fetch_row


@dataclass
class EquivalenceResult:
    dataset: str
    original: tuple[int, int]
    view: tuple[int, int]
    batches: int
    mismatched_batches: list[tuple[list[str], tuple, tuple]] = field(default_factory=list)
    peak_rss_mb: int = 0
    # (count, hash sum) of the rows whose key is NULL: they belong to no
    # key batch, so they are compared as one extra group.
    null_key_original: tuple[int, int] = (0, 0)
    null_key_view: tuple[int, int] = (0, 0)
    # (position, original column, original type, view column, view type)
    schema_mismatches: list[tuple] = field(default_factory=list)

    @property
    def equivalent(self) -> bool:
        return (
            self.original == self.view
            and not self.mismatched_batches
            and self.null_key_original == self.null_key_view
            and not self.schema_mismatches
        )


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _peak_rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def compare_batched(
    con: duckdb.DuckDBPyConnection,
    original_relation: str,
    view_relation: str,
    columns: list[str],
    key_column: str,
    *,
    dataset: str = "",
    batch_size: int = 1,
    max_rss_mb: int | None = None,
    progress=None,
) -> EquivalenceResult:
    """Compare two DuckDB-addressable relations (table names, or
    `read_parquet(...)` expressions) over `columns`, batching by `key_column`."""
    hashed = ", ".join(f"CAST({_quote(c)} AS VARCHAR)" for c in columns)
    key = _quote(key_column)
    sql = (
        f"SELECT COUNT(*), COALESCE(SUM(CAST(hash({hashed}) AS HUGEINT)), 0) "
        f"FROM @RELATION@ WHERE CAST({key} AS VARCHAR) IN (@KEYS@)"
    )
    # Keys come from both sides, so a key present only in the view is
    # compared too (against an empty original batch), not just counted.
    keys = [
        row[0]
        for row in con.execute(
            f"SELECT k FROM (SELECT CAST({key} AS VARCHAR) AS k FROM {original_relation} "
            f"UNION SELECT CAST({key} AS VARCHAR) FROM {view_relation}) ORDER BY 1"
        ).fetchall()
        if row[0] is not None
    ]
    total = -(-len(keys) // batch_size)
    original_total = [0, 0]
    view_total = [0, 0]
    mismatched: list = []
    for index in range(0, len(keys), batch_size):
        batch = keys[index : index + batch_size]
        in_list = ", ".join(_literal(k) for k in batch)
        original = fetch_row(con, sql.replace("@RELATION@", original_relation).replace("@KEYS@", in_list))
        view = fetch_row(con, sql.replace("@RELATION@", view_relation).replace("@KEYS@", in_list))
        original_total[0] += original[0]
        original_total[1] += original[1]
        view_total[0] += view[0]
        view_total[1] += view[1]
        if original != view:
            mismatched.append((batch, original, view))
        rss = _peak_rss_mb()
        if max_rss_mb is not None and rss > max_rss_mb:
            raise MemoryError(f"aborting verification: peak RSS {rss} MB exceeds {max_rss_mb} MB")
        if progress is not None:
            progress(index // batch_size + 1, total, rss)
    null_sql = (
        f"SELECT COUNT(*), COALESCE(SUM(CAST(hash({hashed}) AS HUGEINT)), 0) "
        f"FROM @RELATION@ WHERE {key} IS NULL"
    )
    null_original = fetch_row(con, null_sql.replace("@RELATION@", original_relation))
    null_view = fetch_row(con, null_sql.replace("@RELATION@", view_relation))
    return EquivalenceResult(
        dataset=dataset,
        original=(original_total[0], original_total[1]),
        view=(view_total[0], view_total[1]),
        batches=total,
        mismatched_batches=mismatched,
        peak_rss_mb=_peak_rss_mb(),
        null_key_original=null_original,
        null_key_view=null_view,
    )


def compare_schemas(
    con: duckdb.DuckDBPyConnection, original_relation: str, view_relation: str
) -> list[tuple]:
    """Column names, order and DuckDB types of two relations, position by
    position. Empty when identical."""
    left = con.execute(f"DESCRIBE SELECT * FROM {original_relation}").fetchall()
    right = con.execute(f"DESCRIBE SELECT * FROM {view_relation}").fetchall()
    diffs = []
    for index in range(max(len(left), len(right))):
        a = (left[index][0], left[index][1]) if index < len(left) else (None, None)
        b = (right[index][0], right[index][1]) if index < len(right) else (None, None)
        if a != b:
            diffs.append((index, *a, *b))
    return diffs


def verify_parquet_roundtrip(
    manifest: DatasetMaterialization,
    original_joined_parquet: str,
    rebuilt_joined_parquet: str,
    *,
    threads: int = 2,
    memory_limit: str = "2500MB",
    batch_size: int = 1,
    max_rss_mb: int | None = 4500,
    progress=None,
) -> EquivalenceResult:
    """Parquet-side counterpart of `verify_dataset`: the joined Parquet
    rebuilt from the normalized components must equal the original joined
    Parquet as a multiset, with the same columns, order and types."""
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit={_literal(memory_limit)}")
        original = f"read_parquet({_literal(original_joined_parquet)})"
        rebuilt = f"read_parquet({_literal(rebuilt_joined_parquet)})"
        result = compare_batched(
            con,
            original,
            rebuilt,
            list(manifest.output_columns),
            manifest.fact_join_columns[0],
            dataset=manifest.name,
            batch_size=batch_size,
            max_rss_mb=max_rss_mb,
            progress=progress,
        )
        result.schema_mismatches = compare_schemas(con, original, rebuilt)
        return result
    finally:
        con.close()


def verify_sink(
    manifest: DatasetMaterialization,
    original_joined_parquet: str,
    sink,
    *,
    threads: int = 2,
    memory_limit: str = "2500MB",
    batch_size: int = 1,
    max_rss_mb: int | None = 4500,
    progress=None,
) -> EquivalenceResult:
    """Prove a published dataset equals its original joined Parquet as a
    multiset, whatever the sink: the sink only says how to read its joined
    relation. Schemas (names, order, types) are compared too when the sink
    keeps the source types (`preserves_types`, i.e. files)."""
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit={_literal(memory_limit)}")
        original = f"read_parquet({_literal(original_joined_parquet)})"
        relation = sink.joined_relation(manifest, con)
        result = compare_batched(
            con,
            original,
            relation,
            list(manifest.output_columns),
            manifest.fact_join_columns[0],
            dataset=manifest.name,
            batch_size=batch_size,
            max_rss_mb=max_rss_mb,
            progress=progress,
        )
        if sink.preserves_types:
            result.schema_mismatches = compare_schemas(con, original, relation)
        return result
    finally:
        con.close()


def verify_dataset(
    manifest: DatasetMaterialization,
    fact_parquet_path: str,
    sqlalchemy_dsn: str,
    *,
    view_schema: str = "public",
    threads: int = 2,
    memory_limit: str = "2500MB",
    batch_size: int = 1,
    max_rss_mb: int | None = 4500,
    progress=None,
) -> EquivalenceResult:
    """Check one published dataset (PostgreSQL only) against its Parquet."""
    from sqlalchemy.engine import make_url

    from .sink_postgres import pg_attach

    url = make_url(sqlalchemy_dsn)
    if not url.get_backend_name().startswith("postgres"):
        raise ValueError("verify_dataset needs a PostgreSQL DSN")
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit={_literal(memory_limit)}")
        pg_attach(con, url)
        return compare_batched(
            con,
            f"read_parquet({_literal(fact_parquet_path)})",
            f"pg.{_quote(view_schema)}.{_quote(manifest.name)}",
            list(manifest.output_columns),
            manifest.fact_join_columns[0],
            dataset=manifest.name,
            batch_size=batch_size,
            max_rss_mb=max_rss_mb,
            progress=progress,
        )
    finally:
        con.close()
