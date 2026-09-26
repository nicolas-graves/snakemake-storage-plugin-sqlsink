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


# ---------------------------------------------------------------------------
# v2: a published view against a reference relation, with float tolerance.
# ---------------------------------------------------------------------------

_FLOAT_TYPES = ("DOUBLE", "FLOAT", "REAL")


def _is_float_type(duck_type: str) -> bool:
    return duck_type in _FLOAT_TYPES or duck_type.startswith("DECIMAL")


@dataclass
class ViewEquivalence:
    """Result of `compare_relations`. Rows are grouped by a 64-bit hash of their
    non-float columns; per group the count and, for every float column, the
    non-null count, min, max and sum must agree (floats within `tolerance`,
    relative). With no float column this is an exact multiset comparison; with
    float columns it is a necessary condition that only misses a compensating
    pair of errors inside one group of identical non-float values."""

    dataset: str
    reference_rows: int
    view_rows: int
    reference_groups: int = 0
    view_groups: int = 0
    mismatched_groups: int = 0
    float_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    samples: list[tuple] = field(default_factory=list)
    tolerance: float = 1e-9
    buckets: int = 1
    peak_rss_mb: int = 0

    @property
    def equivalent(self) -> bool:
        return (
            self.reference_rows == self.view_rows
            and self.reference_groups == self.view_groups
            and self.mismatched_groups == 0
            and not self.missing_columns
            and not self.extra_columns
        )


def _describe(con, relation: str) -> list[tuple[str, str]]:
    return [(row[0], str(row[1])) for row in con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()]


def _group_sql(
    relation: str, exact: list[str], floats: list[str], bucket: int, buckets: int, inner_where: str = ""
) -> str:
    """One row per group of rows sharing their non-float columns: streamed
    through a DuckDB aggregate, never materialized in Python. `inner_where`
    filters the relation itself, so it reaches the scan (a PostgreSQL view
    evaluates it); the bucket filter is on the hash and applies after."""
    key = f"hash({', '.join(f'CAST({_quote(c)} AS VARCHAR)' for c in exact)})" if exact else "CAST(0 AS UBIGINT)"
    inner = ", ".join([f"{key} AS h", *(f"CAST({_quote(c)} AS DOUBLE) AS f{i}" for i, c in enumerate(floats))])
    aggs = ["COUNT(*) AS n"]
    for i in range(len(floats)):
        aggs += [f"COUNT(f{i}) AS c{i}", f"MIN(f{i}) AS mn{i}", f"MAX(f{i}) AS mx{i}", f"SUM(f{i}) AS s{i}", f"SUM(ABS(f{i})) AS a{i}"]
    where = f" WHERE h % {int(buckets)} = {int(bucket)}" if buckets > 1 else ""
    return f"SELECT h, {', '.join(aggs)} FROM (SELECT {inner} FROM {relation}{inner_where}){where} GROUP BY h"


def _close(x: str, y: str, tolerance: float, scale: str | None = None) -> str:
    scale = scale or f"GREATEST(ABS({x}), ABS({y}))"
    return f"(({x} IS NULL AND {y} IS NULL) OR {x} = {y} OR ABS({x} - {y}) <= {tolerance!r} * {scale})"


def compare_relations(
    con,
    reference: str,
    view: str,
    *,
    dataset: str = "",
    columns: list[str] | None = None,
    tolerance: float = 1e-9,
    buckets: int = 1,
    key_column: str | None = None,
    key_batch_size: int = 1000,
    sample: int = 5,
    max_rss_mb: int | None = None,
) -> ViewEquivalence:
    """Compare two relations addressable from `con` (a DuckDB connection or
    cursor; e.g. `read_parquet('...')` and a published view) as multisets.

    Text, integers, booleans, dates and NULLs are compared exactly; DOUBLE/FLOAT/
    DECIMAL columns within `tolerance` (relative, NULL equal to NULL). Everything
    is aggregated by DuckDB (spilling to disk if it must); no relation is copied.
    Memory is bounded by splitting the work, at the cost of scanning the
    relations once per batch: by `key_column` (batches of `key_batch_size` distinct
    key values, filtered *inside* the scan, so a database view only evaluates
    its batch; rows with a NULL key form one more batch), else by group hash
    (`buckets`, filtered after the scan). Without either it is one pass, bounded
    only by DuckDB spilling to disk."""
    ref_cols, view_cols = _describe(con, reference), _describe(con, view)
    ref_names, view_names = [c for c, _ in ref_cols], [c for c, _ in view_cols]
    wanted = columns or ref_names
    result = ViewEquivalence(dataset=dataset, reference_rows=0, view_rows=0, tolerance=tolerance, buckets=buckets)
    result.missing_columns = [c for c in wanted if c not in view_names]
    result.extra_columns = [c for c in view_names if c not in ref_names] if columns is None else []
    result.reference_rows = fetch_row(con, f"SELECT COUNT(*) FROM {reference}")[0]
    result.view_rows = fetch_row(con, f"SELECT COUNT(*) FROM {view}")[0]
    if result.missing_columns or result.extra_columns:
        return result
    types = dict(ref_cols)
    view_types = dict(view_cols)
    floats = [c for c in wanted if _is_float_type(types[c]) or _is_float_type(view_types[c])]
    exact = [c for c in wanted if c not in floats]
    result.float_columns = floats

    per_float = []
    for i in range(len(floats)):
        a, b = f"r.mn{i}", f"v.mn{i}"
        per_float.append(
            f"NOT (r.c{i} = v.c{i} AND {_close(a, b, tolerance)} AND {_close(f'r.mx{i}', f'v.mx{i}', tolerance)} "
            f"AND {_close(f'r.s{i}', f'v.s{i}', tolerance, f'GREATEST(r.a{i}, v.a{i})')})"
        )
    differs = " OR ".join(["r.h IS NULL", "v.h IS NULL", "r.n <> v.n", *per_float])
    if key_column is not None:
        if key_column not in wanted:
            raise ValueError(f"key_column {key_column!r} is not one of the compared columns")
        key = _quote(key_column)
        keys = [
            row[0]
            for row in con.execute(
                f"SELECT k FROM (SELECT CAST({key} AS VARCHAR) AS k FROM {reference} "
                f"UNION SELECT CAST({key} AS VARCHAR) FROM {view}) ORDER BY 1"
            ).fetchall()
            if row[0] is not None
        ]
        filters = [
            f" WHERE CAST({key} AS VARCHAR) IN ({', '.join(_literal(k) for k in keys[i : i + key_batch_size])})"
            for i in range(0, len(keys), key_batch_size)
        ] + [f" WHERE {key} IS NULL"]
        batches = [(0, 1, f) for f in filters]
    else:
        batches = [(b, buckets, "") for b in range(max(buckets, 1))]
    result.buckets = len(batches)
    for bucket, nbuckets, inner_where in batches:
        ref_sql = _group_sql(reference, exact, floats, bucket, nbuckets, inner_where)
        view_sql = _group_sql(view, exact, floats, bucket, nbuckets, inner_where)
        head = f"WITH r AS ({ref_sql}), v AS ({view_sql}) "
        counts = fetch_row(
            con,
            head
            + f"SELECT (SELECT COUNT(*) FROM r), (SELECT COUNT(*) FROM v), "
            f"(SELECT COUNT(*) FROM r FULL OUTER JOIN v ON r.h = v.h WHERE {differs})",
        )
        result.reference_groups += counts[0]
        result.view_groups += counts[1]
        result.mismatched_groups += counts[2]
        if counts[2] and len(result.samples) < sample:
            rows = con.execute(
                head
                + f"SELECT COALESCE(r.h, v.h), r.n, v.n FROM r FULL OUTER JOIN v ON r.h = v.h WHERE {differs} "
                f"LIMIT {int(sample - len(result.samples))}"
            ).fetchall()
            result.samples.extend(tuple(r) for r in rows)
        rss = _peak_rss_mb()
        result.peak_rss_mb = rss
        if max_rss_mb is not None and rss > max_rss_mb:
            raise MemoryError(f"aborting verification: peak RSS {rss} MB exceeds {max_rss_mb} MB")
    return result


def verify_view(
    sink,
    manifest,
    reference: str,
    *,
    tolerance: float = 1e-9,
    buckets: int = 1,
    key_column: str | None = None,
    key_batch_size: int = 1000,
    threads: int | None = 2,
    memory_limit: str | None = "2500MB",
    max_rss_mb: int | None = 4500,
) -> ViewEquivalence:
    """Prove the published relation of a dataset equals `reference` (a Parquet
    path, or a DuckDB relation expression such as `read_parquet('...')` or a
    parenthesized subquery) as a multiset, floats within `tolerance`.

    The comparison runs where the published relation can be read in place: on
    PostgreSQL through DuckDB's postgres scanner, on a DuckDB sink inside the
    engine's own connection. Neither copies the relation."""
    reference_sql = reference if "(" in reference else f"read_parquet({_literal(reference)})"
    with sink.analysis_session(threads=threads, memory_limit=memory_limit) as (con, ref):
        return compare_relations(
            con,
            reference_sql,
            ref(manifest.name),
            dataset=manifest.name,
            tolerance=tolerance,
            buckets=buckets,
            key_column=key_column,
            key_batch_size=key_batch_size,
            max_rss_mb=max_rss_mb,
        )
