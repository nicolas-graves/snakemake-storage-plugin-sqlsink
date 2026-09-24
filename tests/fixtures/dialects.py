"""Helpers for asserting behaviour that differs by SQL dialect."""

from __future__ import annotations

import contextlib

import pytest


def qualified(name: str, schema: str | None) -> str:
    return f'"{schema}"."{name}"' if schema else f'"{name}"'


@contextlib.contextmanager
def declared_keys(engine):
    """Assertions on declared foreign keys and secondary indexes.

    DuckDB cannot rename a table that a foreign key references or that has an
    index, and the rename-swap needs both, so its sink declares neither. On
    DuckDB the block must therefore *fail*; if it starts passing, the gap
    has closed and this guard should go.
    """
    if engine.dialect.name != "duckdb":
        yield
        return
    with pytest.raises((AssertionError, ValueError, pytest.fail.Exception)):
        yield


def pk_columns(engine, table: str, schema: str | None) -> list[str]:
    """Primary-key columns of a table. duckdb_engine's reflection returns no
    primary keys at all, so DuckDB is asked through its own catalog."""
    from sqlalchemy import inspect, text

    if engine.dialect.name != "duckdb":
        return inspect(engine).get_pk_constraint(table, schema=schema)["constrained_columns"]
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT constraint_column_names FROM duckdb_constraints() "
                "WHERE constraint_type = 'PRIMARY KEY' AND table_name = :t AND schema_name = :s"
            ),
            {"t": table, "s": schema or "main"},
        ).fetchall()
    return list(rows[0][0]) if rows else []
