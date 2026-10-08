"""Import DuckDB only when a DuckDB-backed operation is requested."""

from __future__ import annotations


def require_duckdb():
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        if exc.name != "duckdb":
            raise
        raise ImportError("This operation requires DuckDB; install sqlsink[duckdb].") from exc
    return duckdb
