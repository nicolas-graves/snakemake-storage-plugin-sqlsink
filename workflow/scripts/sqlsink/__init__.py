"""Per-table incremental publishing of Parquet data into a SQL database.

Engine-agnostic (SQLAlchemy Core): the marker schema, staging and publish
transaction work on PostgreSQL and DuckDB. PostgreSQL-specific fast
paths (COPY, advisory locks) are used there and fall back to portable
SQLAlchemy operations (plus a file lock) on DuckDB.

Publication is deliberately non-authoritative: omitted names are untouched.
PostgreSQL callers with a complete registry can use
``SqlSink.reconcile(DesiredState(...), authoritative=True)`` for a read-only
report, then explicitly opt into ``report.prune(execute=True)``.
"""

from .reconcile import DesiredState, ManagedObject, ReconciliationError, ReconciliationReport

__all__ = ["DesiredState", "ManagedObject", "ReconciliationError", "ReconciliationReport"]
