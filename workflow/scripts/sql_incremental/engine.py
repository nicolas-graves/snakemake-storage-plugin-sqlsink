"""Engine construction and the dialect-specific fast paths.

Everything that genuinely differs across databases (locking, bulk COPY)
is isolated here, with a portable SQLAlchemy fallback for dialects that
don't have a native equivalent (DuckDB). Everything else in this package should
go through plain SQLAlchemy Core and needs no dialect branching at all.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from sqlalchemy import Engine, Table, create_engine, func, insert, select, text, update


def make_engine(dsn: str, **kwargs) -> Engine:
    return create_engine(dsn, **kwargs)


@contextlib.contextmanager
def advisory_lock(conn, key: str) -> Iterator[None]:
    """Serialize a critical section across concurrent processes.

    PostgreSQL: session-level advisory lock, released on function exit.
    DuckDB: an exclusive `flock` on a sidecar file next to the database
    file (DuckDB is single-writer; this makes a second publisher wait
    instead of failing to open the file).
    """
    dialect = conn.engine.dialect.name
    if dialect == "postgresql":
        lock_id = _lock_key_to_bigint(key)
        conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": lock_id})
        try:
            yield
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_id})
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
    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))


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
    qualified = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
    col_list = ", ".join(f'"{c}"' for c in columns)
    with cursor.copy(f"COPY {qualified} ({col_list}) FROM STDIN WITH (FORMAT csv)") as copy:
        copy.write(payload)


def bulk_load_streaming(conn, table: Table, duckdb_cursor, columns: list[str], batch_size: int = 50_000) -> int:
    """Load an already-executed DuckDB cursor's result into `table` in
    bounded batches, never materializing the full result as one Python
    list. `duckdb_cursor` must have had a query already run against it
    (`con.execute(sql)`); this only drives its `fetchmany()`.

    PostgreSQL: each batch is streamed straight into one `COPY ... FROM
    STDIN` (a single COPY call spanning all batches, so it's still one
    server-side operation) rather than building one giant CSV buffer up
    front. Other dialects: a batched `INSERT`, same bound.

    Returns the total row count loaded.
    """
    dialect = conn.engine.dialect.name
    total = 0

    if dialect == "postgresql":
        raw_conn = conn.connection
        cursor = raw_conn.driver_connection.cursor() if hasattr(raw_conn, "driver_connection") else raw_conn.cursor()
        qualified = f'"{table.schema}"."{table.name}"' if table.schema else f'"{table.name}"'
        col_list = ", ".join(f'"{c}"' for c in columns)
        with cursor.copy(f"COPY {qualified} ({col_list}) FROM STDIN WITH (FORMAT csv)") as copy:
            while True:
                batch = duckdb_cursor.fetchmany(batch_size)
                if not batch:
                    break
                copy.write(_csv_rows(batch))
                total += len(batch)
        return total

    while True:
        batch = duckdb_cursor.fetchmany(batch_size)
        if not batch:
            break
        conn.execute(insert(table), [dict(zip(columns, row)) for row in batch])
        total += len(batch)
    return total
