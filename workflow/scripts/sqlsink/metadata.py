"""SQLAlchemy Core schema for the private marker table.

Declared as a Table/MetaData object (not raw SQL) so the same definition
works, unmodified, against PostgreSQL in production and DuckDB locally.

Staging vs. public tables are distinguished by a **name prefix**, not by
schemas: a staging table is renamed into place within its schema.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    Table,
    Text,
    func,
)

STAGING_PREFIX = "stg__"
BOOTSTRAP_LOCK_KEY = "snakemake_sql:create_marker_tables"

metadata = MetaData()

analytics_table_updates = Table(
    "_pipeline_meta_table_updates",
    metadata,
    Column("table_name", Text, primary_key=True),
    Column("update_id", Text, nullable=False),
    Column("parquet_sha256", Text, nullable=False),
    Column("loader_version", Integer, nullable=False),
    Column("type_map_version", Integer, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("published_by", Text, nullable=False),
)

# Separate from analytics_table_updates on purpose: writing *that* marker
# must stay coupled only to a successful publish transaction (see
# publish.py). This one records "this exact Parquet fingerprint has been
# staged" -- it's what lets the storage plugin's exists() honestly report
# True right after staging, without it meaning "published".
staged_table_updates = Table(
    "_pipeline_meta_staged_updates",
    metadata,
    Column("table_name", Text, primary_key=True),
    Column("update_id", Text, nullable=False),
    Column("parquet_sha256", Text, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("staged_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)


def staging_name(table_name: str) -> str:
    return f"{STAGING_PREFIX}{table_name}"


# Marker tables for dataset-shaped publications (compact fact table + shared
# contour table + public compatibility view), separate from
# analytics_table_updates/staged_table_updates above because a dataset's
# identity depends on two source Parquets and a materialization manifest,
# not just one Parquet fingerprint.
analytics_dataset_updates = Table(
    "_pipeline_meta_dataset_updates",
    metadata,
    Column("dataset_name", Text, primary_key=True),
    Column("update_id", Text, nullable=False),
    Column("manifest_hash", Text, nullable=False),
    Column("fact_sha256", Text, nullable=False),
    Column("contour_sha256", Text, nullable=False),
    Column("loader_version", Integer, nullable=False),
    Column("type_map_version", Integer, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("published_by", Text, nullable=False),
)

staged_dataset_updates = Table(
    "_pipeline_meta_staged_dataset_updates",
    metadata,
    Column("dataset_name", Text, primary_key=True),
    Column("update_id", Text, nullable=False),
    Column("manifest_hash", Text, nullable=False),
    Column("fact_sha256", Text, nullable=False),
    Column("contour_sha256", Text, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("staged_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# The contour/dimension relation is shared across datasets and staged
# independently of any one dataset's fact Parquet, keyed by its own name
# (there may eventually be more than one shared dimension table).
contour_updates = Table(
    "_pipeline_meta_contour_updates",
    metadata,
    Column("contour_table", Text, primary_key=True),
    Column("contour_sha256", Text, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

staged_contour_updates = Table(
    "_pipeline_meta_staged_contour_updates",
    metadata,
    Column("contour_table", Text, primary_key=True),
    Column("contour_sha256", Text, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("staged_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# v2 components: one row per physical component table, whether shared by
# several datasets or owned by one. All new tables: a marker table that already
# exists in a database is never altered.
component_updates = Table(
    "_pipeline_meta_component_updates",
    metadata,
    Column("component_name", Text, primary_key=True),
    Column("update_id", Text, nullable=False),
    Column("source_sha256", Text, nullable=False),
    Column("storage_schema", Text, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

staged_component_updates = Table(
    "_pipeline_meta_staged_component_updates",
    metadata,
    Column("component_name", Text, primary_key=True),
    Column("update_id", Text, nullable=False),
    Column("source_sha256", Text, nullable=False),
    Column("storage_schema", Text, nullable=False),
    Column("row_count", BigInteger, nullable=False),
    Column("staged_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Which components each published v2 dataset was built over: what tells a
# publish which views must be recreated when a shared component is swapped.
# The dataset's own marker (analytics_dataset_updates) carries composite
# digests in its existing NOT NULL columns.
dataset_components = Table(
    "_pipeline_meta_dataset_components",
    metadata,
    Column("dataset_name", Text, primary_key=True),
    Column("component_name", Text, primary_key=True),
    Column("component_update_id", Text, nullable=False),
    Column("materialize", Text, nullable=False),
)


def compact_table_name(dataset_name: str) -> str:
    return f"compact__{dataset_name}"


def physical_name_and_schema(dialect_name: str, bare_name: str, schema: str) -> tuple[str, str | None]:
    """Resolve a manifest-declared (bare_name, schema) pair to the physical
    (name, schema) SQLAlchemy should use. Both supported dialects
    (PostgreSQL, DuckDB) have real schemas, needed for the read-only grant
    story; the dialect is kept in the signature for callers' sake."""
    return bare_name, schema


def create_all(engine) -> None:
    """Idempotently ensure the marker tables exist.

    This is DDL: only code paths that stage or publish may call it. Planning
    (`snakemake -n`), existence checks, mtimes and inventories must stay
    read-only and go through `table_exists` / `fetch_one`, which treat a
    missing marker table as "no marker".

    Serialized under an advisory lock: on an empty database, concurrent jobs
    would otherwise all pass `checkfirst` and race on `CREATE TABLE`
    (PostgreSQL rejects the loser with a `pg_type_typname_nsp_index`
    unique violation)."""
    from .engine import advisory_lock

    with engine.begin() as conn, advisory_lock(conn, BOOTSTRAP_LOCK_KEY, transactional=True):
        metadata.create_all(conn, checkfirst=True)


def table_exists(engine, table) -> bool:
    """Read-only: does the marker table exist (catalog lookup, no DDL)?"""
    from sqlalchemy import inspect

    return inspect(engine).has_table(table.name)


def fetch_one(engine, table, key_col: str, key) -> dict | None:
    """Read-only: the marker row of `table` whose `key_col` is `key`, or None
    if there is none or the marker table itself does not exist (a database
    nothing was ever staged into)."""
    from sqlalchemy import select

    if not table_exists(engine, table):
        return None
    with engine.connect() as conn:
        row = conn.execute(select(table).where(table.c[key_col] == key)).mappings().first()
    return dict(row) if row else None
