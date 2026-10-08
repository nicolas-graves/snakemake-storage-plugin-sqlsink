"""Engine construction and the dialect-specific fast paths.

Everything that genuinely differs across databases (locking, bulk COPY)
is isolated here, with a portable SQLAlchemy fallback for dialects that
don't have a native equivalent (DuckDB). Everything else in this package should
go through plain SQLAlchemy Core and needs no dialect branching at all.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path
from collections.abc import Iterator

from sqlalchemy import Engine, Table, create_engine, func, insert, select, text, update
from sqlalchemy.engine import URL

from .sqlident import qualified, quote_ident


def make_engine(dsn: str | URL, **kwargs) -> Engine:
    return create_engine(dsn, **kwargs)


_shared_engines: dict[str, Engine] = {}


def get_engine(dsn: str | URL) -> Engine:
    """Return the process-local engine used by planning consumers."""
    key = dsn.render_as_string(hide_password=False) if isinstance(dsn, URL) else str(dsn)
    if key not in _shared_engines:
        _shared_engines[key] = make_engine(dsn)
    return _shared_engines[key]


LOCK_TIMEOUT_ENV = "SQLSINK_LOCK_TIMEOUT"
KEEP_OLD_ENV = "SQLSINK_KEEP_OLD"
ANALYZE_ENV = "SQLSINK_ANALYZE"
_LOCK_TIMEOUT_RE = re.compile(r"^\d+\s*(us|ms|s|min|h|d)?$")
_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def resolve_lock_timeout(value: str | int | None = None) -> str | None:
    """The PostgreSQL `lock_timeout` a publish should run under: `value`, else
    the `SQLSINK_LOCK_TIMEOUT` environment variable, else None (the server's
    own setting applies). A bare number is milliseconds, as in PostgreSQL;
    `"45s"`, `"500ms"`, `"2min"` are accepted, `0` disables the timeout."""
    if value is None:
        value = os.environ.get(LOCK_TIMEOUT_ENV)
    if value is None or not str(value).strip():
        return None
    text_value = str(value).strip()
    if not _LOCK_TIMEOUT_RE.match(text_value):
        raise ValueError(f"invalid lock timeout {value!r}: expected a number of milliseconds or a value like '45s', '500ms', '2min'")
    return text_value


def apply_lock_timeout(conn, value: str | int | None = None) -> None:
    """`SET LOCAL lock_timeout` for the current transaction (PostgreSQL only;
    a no-op on DuckDB, whose single-writer model has no lock waits of this
    kind). Applies to every lock wait in the transaction, the publish advisory
    lock included, so a publish that cannot get its locks fails and rolls
    back instead of queueing (and stalling readers behind it) indefinitely."""
    resolved = resolve_lock_timeout(value)
    if resolved is None or conn.engine.dialect.name != "postgresql":
        return
    conn.execute(text("SELECT set_config('lock_timeout', :value, true)"), {"value": resolved})


def resolve_keep_old(value: bool | None = None) -> bool:
    """Whether a publish keeps replaced relations aside (`__old__<name>`)
    instead of dropping them: `value`, else the `SQLSINK_KEEP_OLD` environment
    variable (`1`/`true`/`yes`/`on`), else False."""
    if value is not None:
        return bool(value)
    return os.environ.get(KEEP_OLD_ENV, "").strip().lower() in _BOOL_TRUE


def resolve_analyze(value: bool | None = None) -> bool:
    """Whether a publish runs `ANALYZE` on the relations it just swapped in
    (PostgreSQL only; always a no-op on DuckDB, see `analyze_relations`):
    `value`, else the `SQLSINK_ANALYZE` environment variable, else **True**.

    Unlike `keep_old`, this defaults ON: right after a bulk COPY into a
    freshly renamed table, PostgreSQL's autovacuum has not yet gathered
    statistics, so the planner uses stale or default estimates until it
    does, on its own schedule. That is a real, measured multi-second
    latency regression on some queries; the one-`ANALYZE`-per-relation
    cost of avoiding it is cheap, and forgetting to opt in is the kind of
    thing that is only noticed after the fact. Set `SQLSINK_ANALYZE=0` (or
    `false`/`no`/`off`) to opt back out.
    """
    if value is not None:
        return bool(value)
    raw = os.environ.get(ANALYZE_ENV)
    if raw is None or not raw.strip():
        return True
    normalized = raw.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized in _BOOL_FALSE:
        return False
    raise ValueError(f"invalid {ANALYZE_ENV} value {raw!r}: expected 1/true/yes/on or 0/false/no/off")


@contextlib.contextmanager
def advisory_lock(conn, key: str, *, transactional: bool = False) -> Iterator[None]:
    """Serialize a critical section across concurrent processes.

    PostgreSQL: session-level advisory lock, released on function exit. With
    `transactional=True` (only for use inside a transaction that ends with the
    critical section) a transaction-level lock is taken instead: it is
    released by commit *and* rollback, so a failed statement (an aborted
    transaction can no longer run `pg_advisory_unlock`, and a rolled-back
    session lock would stay on the pooled connection) cannot leak it or mask
    the original error. Both kinds share one lock namespace.
    DuckDB: an exclusive `flock` on a sidecar file next to the database
    file (DuckDB is single-writer; this makes a second publisher wait
    instead of failing to open the file).
    """
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        lock_id = _lock_key_to_bigint(key)
        if transactional:
            conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": lock_id})
            yield
            return
        conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": lock_id})
        try:
            yield
        finally:
            try:
                conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_id})
            except Exception:  # aborted transaction: do not mask the original error
                pass
    elif dialect == "duckdb":
        with _file_lock(conn.engine.url.database):
            yield
    else:
        yield


@contextlib.contextmanager
def _file_lock(database: str | None) -> Iterator[None]:
    import fcntl

    if not database or database == ":memory:":
        yield
        return
    with open(f"{database}.lock", "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_schema(conn, schema: str | None) -> None:
    """Create the loader-owned storage schema if it doesn't exist yet.
    No-op when `schema` is None."""
    if schema is None:
        return
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}"))


def upsert_by_pk(conn, table: Table, pk_col: str, values: dict, now_col: str | None = None) -> None:
    """Insert-or-update a single row keyed by `pk_col`.

    `now_col`, if given, is set to the database's current timestamp on
    every write (e.g. `published_at`) rather than a Python-side value.
    Native `ON CONFLICT` on PostgreSQL; existence-check-then-insert/update
    elsewhere (DuckDB) (portable, and fine for the single-row writes this package
    does).
    """
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        stmt = pg_insert(table).values(**values)
        update_cols = {c: stmt.excluded[c] for c in values if c != pk_col}
        if now_col:
            update_cols[now_col] = text("now()")
        stmt = stmt.on_conflict_do_update(index_elements=[pk_col], set_=update_cols)
        conn.execute(stmt)
        return

    if now_col:
        values = {**values, now_col: func.now()}
    pk_column = table.c[pk_col]
    existing = conn.execute(select(pk_column).where(pk_column == values[pk_col])).scalar_one_or_none()
    if existing is None:
        conn.execute(insert(table).values(**values))
    else:
        conn.execute(update(table).where(pk_column == values[pk_col]).values(**values))


def _lock_key_to_bigint(key: str) -> int:
    import hashlib

    digest = hashlib.sha256(key.encode()).digest()[:8]
    value = int.from_bytes(digest, "big", signed=True)
    return value


def _csv_rows(rows) -> str:
    """CSV text for PostgreSQL `COPY ... (FORMAT csv)` that keeps NULL and the
    empty string apart: NULL is an unquoted empty field (COPY's default NULL),
    every other value is quoted, so `''` arrives as `""`, an empty string."""
    out = []
    for row in rows:
        out.append(",".join("" if v is None else '"' + str(v).replace('"', '""') + '"' for v in row))
    return "\n".join(out) + "\n" if out else ""


def bulk_load(conn, table: Table, rows: list[dict]) -> None:
    """Load rows into `table`, using COPY on PostgreSQL and a portable
    bulk INSERT elsewhere (fine for the small/medium tables this path
    needs to support; large-table COPY
    tuning is a known follow-up, not needed for correctness here).
    """
    if not rows:
        return
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        _bulk_load_postgres_copy(conn, table, rows)
    else:
        conn.execute(insert(table), rows)


def _bulk_load_postgres_copy(conn, table: Table, rows: list[dict]) -> None:
    columns = list(rows[0].keys())
    payload = _csv_rows([row[c] for c in columns] for row in rows)

    raw_conn = conn.connection
    cursor = raw_conn.driver_connection.cursor() if hasattr(raw_conn, "driver_connection") else raw_conn.cursor()
    ref = qualified(table.name, table.schema)
    col_list = ", ".join(quote_ident(c) for c in columns)
    with cursor.copy(f"COPY {ref} ({col_list}) FROM STDIN WITH (FORMAT csv)") as copy:
        copy.write(payload)


def bulk_load_streaming(conn, table: Table, duckdb_cursor, columns: list[str], batch_size: int = 50_000) -> int:
    """Load an already-executed DuckDB cursor's result into `table` in
    bounded batches, never materializing the full result as one Python
    list. `duckdb_cursor` must have had a query already run against it
    (`con.execute(sql)`); this only drives its `fetchmany()`.

    PostgreSQL: each batch is streamed straight into one `COPY ... FROM
    STDIN` (a single COPY call spanning all batches, so it's still one
    server-side operation) rather than building one giant CSV buffer up
    front. DuckDB: Arrow batches inserted with `INSERT ... SELECT`. Other dialects: a batched `INSERT`, same bound.

    Returns the total row count loaded.
    """
    dialect = conn.engine.dialect.name
    total = 0

    if dialect == "postgresql":
        raw_conn = conn.connection
        cursor = raw_conn.driver_connection.cursor() if hasattr(raw_conn, "driver_connection") else raw_conn.cursor()
        ref = qualified(table.name, table.schema)
        col_list = ", ".join(quote_ident(c) for c in columns)
        with cursor.copy(f"COPY {ref} ({col_list}) FROM STDIN WITH (FORMAT csv)") as copy:
            while True:
                batch = duckdb_cursor.fetchmany(batch_size)
                if not batch:
                    break
                copy.write(_csv_rows(batch))
                total += len(batch)
        return total

    if dialect == "duckdb":
        # Arrow batches into the engine's own connection (same transaction as
        # the staging table's DDL): no per-row Python objects, no per-row insert.
        import pyarrow as pa

        raw = conn.connection.driver_connection
        ref = qualified(table.name, table.schema)
        col_list = ", ".join(quote_ident(c) for c in columns)
        reader = (
            duckdb_cursor.to_arrow_reader(batch_size)
            if hasattr(duckdb_cursor, "to_arrow_reader")
            else duckdb_cursor.fetch_record_batch(batch_size)
        )
        for batch in reader:
            raw.register("__sqlsink_batch", pa.Table.from_batches([batch]))
            try:
                raw.execute(f"INSERT INTO {ref} ({col_list}) SELECT * FROM __sqlsink_batch")
            finally:
                raw.unregister("__sqlsink_batch")
            total += batch.num_rows
        return total

    while True:
        batch = duckdb_cursor.fetchmany(batch_size)
        if not batch:
            break
        conn.execute(insert(table), [dict(zip(columns, row)) for row in batch])
        total += len(batch)
    return total


def bulk_load_query(conn, table: Table, duckdb_con, sql: str, columns: list[str]) -> int:
    """Experimental binary loader for an already-created staging table.

    PostgreSQL's binary COPY stays on ``conn`` so the table and its marker
    share a transaction. The export file is private to this call and removed
    even if either COPY fails. It is not selected by staging until the remote
    payload and elapsed-time comparison has been made. Other sinks retain
    their Arrow batch path.
    """
    if conn.engine.dialect.name != "postgresql":
        return bulk_load_streaming(conn, table, duckdb_con.execute(sql), columns)

    ref = qualified(table.name, table.schema)
    col_list = ", ".join(quote_ident(c) for c in columns)
    with tempfile.TemporaryDirectory(prefix="sqlsink-pg-copy-") as directory:
        path = Path(directory) / "rows.bin"
        # LOAD is idempotent and works with an extension installed ahead of
        # time, avoiding a download during a staging transaction.
        duckdb_con.execute("LOAD postgres")
        count = duckdb_con.execute(
            f"COPY ({sql}) TO ? (FORMAT postgres_binary)", [str(path)]
        ).fetchone()[0]
        raw_conn = conn.connection
        cursor = raw_conn.driver_connection.cursor() if hasattr(raw_conn, "driver_connection") else raw_conn.cursor()
        with cursor.copy(f"COPY {ref} ({col_list}) FROM STDIN WITH (FORMAT binary)") as copy:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    copy.write(chunk)
    return count
