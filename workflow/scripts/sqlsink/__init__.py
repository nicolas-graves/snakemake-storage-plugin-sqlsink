"""Per-table incremental publishing of Parquet data into a SQL database.

Engine-agnostic (SQLAlchemy Core): the marker schema, staging and publish
transaction work on PostgreSQL and DuckDB. PostgreSQL-specific fast
paths (COPY, advisory locks) are used there and fall back to portable
SQLAlchemy operations (plus a file lock) on DuckDB.
"""
