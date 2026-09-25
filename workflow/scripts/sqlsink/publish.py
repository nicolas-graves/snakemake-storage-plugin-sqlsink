"""Atomic, all-or-nothing publication of staged tables.

`ALTER TABLE ... RENAME TO ...` is one of the rare pieces of DDL with
identical syntax on both PostgreSQL and DuckDB, so the rename-swap here
needs no dialect branching. Publication is wrapped in a single
`engine.begin()` block: on any exception the whole transaction rolls back
(driven by the target database's own transactional DDL semantics, exposed
uniformly through SQLAlchemy), so a crash mid-publish never leaves a
partial rename or a stale marker row behind.
"""

from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import inspect, select, text

from . import metadata as meta_mod
from .engine import advisory_lock, upsert_by_pk
from .manifest import DatasetMaterialization

log = logging.getLogger(__name__)

PUBLISH_LOCK_KEY = "snakemake_sql:publish_tables"


class PublishConflict(RuntimeError):
    """Raised when a table's marker changed between staging and publish
    (another process published a newer version first)."""


def publish_tables(engine, receipts: list[dict]) -> list[str]:
    """Publish every `status == "staged"` receipt in one transaction.

    Returns the list of table names actually published (skips tables that
    were already current).
    """
    staged = [r for r in receipts if r["status"] == "staged"]
    if not staged:
        return []

    published: list[str] = []
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    old_names: list[str] = []

    with engine.begin() as conn:
        with advisory_lock(conn, PUBLISH_LOCK_KEY):
            inspector = inspect(conn)
            for receipt in staged:
                table_name = receipt["table"]
                _assert_marker_unchanged_since_staging(conn, table_name, receipt)

                staging_name = meta_mod.staging_name(table_name)
                if inspector.has_table(table_name):
                    old_name = f"{table_name}__old__{ts}"
                    conn.execute(text(f'ALTER TABLE "{table_name}" RENAME TO "{old_name}"'))
                    old_names.append(old_name)

                conn.execute(text(f'ALTER TABLE "{staging_name}" RENAME TO "{table_name}"'))
                _upsert_marker(conn, receipt)
                published.append(table_name)

    # Best-effort, non-transactional cleanup: failures here are logged,
    # not fatal — a leftover `*__old__*` table is wasted disk, not a
    # correctness problem, and leaving it around briefly gives a human a
    # window to inspect it if a downstream smoke test fails.
    _drop_old_tables(engine, old_names)

    return published


def _assert_marker_unchanged_since_staging(conn, table_name: str, receipt: dict) -> None:
    stmt = select(meta_mod.analytics_table_updates.c.update_id).where(
        meta_mod.analytics_table_updates.c.table_name == table_name
    )
    current = conn.execute(stmt).scalar_one_or_none()
    staged_from = receipt.get("marker_update_id_at_stage_time")
    if current is not None and current != staged_from:
        raise PublishConflict(
            f"table {table_name!r} was republished by another process "
            f"after this receipt was staged (marker is {current!r}, "
            f"receipt expected {staged_from!r}); aborting the whole "
            f"publish transaction rather than risk partial state"
        )


def _upsert_marker(conn, receipt: dict) -> None:
    values = {
        "table_name": receipt["table"],
        "update_id": receipt["update_id"],
        "parquet_sha256": receipt["parquet_sha256"],
        "loader_version": receipt.get("loader_version", 1),
        "type_map_version": receipt.get("type_map_version", 1),
        "row_count": receipt["row_count"],
        "published_by": receipt.get("published_by", "pipeline"),
    }
    upsert_by_pk(conn, meta_mod.analytics_table_updates, "table_name", values, now_col="published_at")


def _drop_old_tables(engine, old_names: list[str]) -> None:
    for name in old_names:
        try:
            with engine.begin() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        except Exception:
            # Non-fatal: leftover table, cleaned up on a later run/sweep.
            log.warning("could not drop leftover table %s", name, exc_info=True)


def publish_datasets(
    engine,
    manifests: list[DatasetMaterialization],
    dataset_receipts: list[dict],
    contour_receipts: list[dict],
) -> list[str]:
    """Publish every staged dataset (compact fact table + public
    compatibility view) and any staged shared contour tables, in one
    transaction under the same advisory lock as `publish_tables`.

    Contours are published before the datasets that join against them, so a
    freshly created/replaced view never points at a stale or missing
    contour table. `manifests` must include every dataset in
    `dataset_receipts` and cover every contour table referenced by them.
    """
    staged_contours = {r["contour_table"]: r for r in contour_receipts if r["status"] == "staged"}
    staged_datasets = [r for r in dataset_receipts if r["status"] == "staged"]
    if not staged_contours and not staged_datasets:
        return []

    manifest_by_name = {m.name: m for m in manifests}
    contour_schema = {m.contour_table: m.compact_schema for m in manifests}
    zone_table = {m.contour_table: m.zone_table for m in manifests if m.keyed}

    published: list[str] = []
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    old_physical: list[tuple[str, str | None]] = []

    with engine.begin() as conn:
        with advisory_lock(conn, PUBLISH_LOCK_KEY):
            dialect = conn.engine.dialect.name

            _assert_keyed_dependents_restaged(manifests, staged_contours, staged_datasets)

            for contour_table, receipt in staged_contours.items():
                schema = contour_schema[contour_table]
                zone = zone_table.get(contour_table)
                if zone:
                    _rename_swap_physical(
                        conn, dialect, zone, schema, meta_mod.staging_name(zone), old_physical, ts
                    )
                _rename_swap_physical(
                    conn, dialect, contour_table, schema, meta_mod.staging_name(contour_table), old_physical, ts
                )
                _upsert_contour_marker(conn, receipt)

            for receipt in staged_datasets:
                dataset_name = receipt["dataset"]
                manifest = manifest_by_name[dataset_name]
                _assert_dataset_marker_unchanged_since_staging(conn, dataset_name, receipt)

                compact_bare = meta_mod.compact_table_name(dataset_name)
                _rename_swap_physical(
                    conn,
                    dialect,
                    compact_bare,
                    manifest.compact_schema,
                    meta_mod.staging_name(compact_bare),
                    old_physical,
                    ts,
                )

                _replace_compatibility_view(conn, dataset_name, receipt["view_sql"], old_physical, ts)
                _upsert_dataset_marker(conn, receipt)
                published.append(dataset_name)

    _drop_old_tables_qualified(engine, old_physical)
    return published


def _assert_keyed_dependents_restaged(manifests, staged_contours, staged_datasets) -> None:
    """A foreign key follows its referenced table, so replacing a keyed zone
    while a dataset that references it keeps its old facts would leave that
    dataset pointing at the dropped zone. Its update_id includes the contour
    hash, so it is restaged whenever it is staged at all: require that."""
    restaged = {r["dataset"] for r in staged_datasets}
    for m in manifests:
        if m.keyed and m.contour_table in staged_contours and m.name not in restaged:
            raise PublishConflict(
                f"contour {m.contour_table!r} is being replaced but keyed dataset {m.name!r}, "
                "which references it, is not restaged in this publish"
            )


def _rename_swap_physical(
    conn,
    dialect: str,
    bare_name: str,
    schema: str,
    staging_bare_name: str,
    old_physical: list[tuple[str, str | None]],
    ts: str,
) -> None:
    """Rename-swap a schema-aware physical table into place, the same
    all-or-nothing pattern as `publish_tables`' plain-name rename, extended
    to a real PostgreSQL schema where the dialect supports one."""
    live_name, live_schema = meta_mod.physical_name_and_schema(dialect, bare_name, schema)
    staging_name, staging_schema = meta_mod.physical_name_and_schema(dialect, staging_bare_name, schema)
    live_ref = f'"{live_schema}"."{live_name}"' if live_schema else f'"{live_name}"'
    staging_ref = f'"{staging_schema}"."{staging_name}"' if staging_schema else f'"{staging_name}"'

    inspector = inspect(conn)
    if inspector.has_table(live_name, schema=live_schema):
        old_name = f"{live_name}__old__{ts}"
        conn.execute(text(f'ALTER TABLE {live_ref} RENAME TO "{old_name}"'))
        old_physical.append((old_name, live_schema))

    conn.execute(text(f'ALTER TABLE {staging_ref} RENAME TO "{live_name}"'))


def _replace_compatibility_view(
    conn, dataset_name: str, view_sql: str, old_physical: list[tuple[str, str | None]], ts: str
) -> None:
    """Point the public dataset name at the freshly published view body.

    The very first migration for a dataset replaces an existing *table*
    (the pre-normalization public table) with a view; every later
    republication replaces an existing *view* with a new one. Both are
    handled here since a receipt alone can't say which generation this is.
    """
    inspector = inspect(conn)
    is_table = dataset_name in set(inspector.get_table_names())
    is_view = dataset_name in set(inspector.get_view_names())

    if is_table and not is_view:
        old_name = f"{dataset_name}__old__{ts}"
        conn.execute(text(f'ALTER TABLE "{dataset_name}" RENAME TO "{old_name}"'))
        old_physical.append((old_name, None))
    else:
        conn.execute(text(f'DROP VIEW IF EXISTS "{dataset_name}"'))

    conn.execute(text(f'CREATE VIEW "{dataset_name}" AS {view_sql}'))


def _assert_dataset_marker_unchanged_since_staging(conn, dataset_name: str, receipt: dict) -> None:
    stmt = select(meta_mod.analytics_dataset_updates.c.update_id).where(
        meta_mod.analytics_dataset_updates.c.dataset_name == dataset_name
    )
    current = conn.execute(stmt).scalar_one_or_none()
    staged_from = receipt.get("marker_update_id_at_stage_time")
    if current is not None and current != staged_from:
        raise PublishConflict(
            f"dataset {dataset_name!r} was republished by another process "
            f"after this receipt was staged (marker is {current!r}, "
            f"receipt expected {staged_from!r}); aborting the whole "
            f"publish transaction rather than risk partial state"
        )


def _upsert_dataset_marker(conn, receipt: dict) -> None:
    values = {
        "dataset_name": receipt["dataset"],
        "update_id": receipt["update_id"],
        "manifest_hash": receipt["manifest_hash"],
        "fact_sha256": receipt["fact_sha256"],
        "contour_sha256": receipt["contour_sha256"],
        "loader_version": receipt.get("loader_version", 1),
        "type_map_version": receipt.get("type_map_version", 1),
        "row_count": receipt["row_count"],
        "published_by": receipt.get("published_by", "pipeline"),
    }
    upsert_by_pk(conn, meta_mod.analytics_dataset_updates, "dataset_name", values, now_col="published_at")


def _upsert_contour_marker(conn, receipt: dict) -> None:
    values = {
        "contour_table": receipt["contour_table"],
        "contour_sha256": receipt["contour_sha256"],
        "row_count": receipt["row_count"],
    }
    upsert_by_pk(conn, meta_mod.contour_updates, "contour_table", values, now_col="published_at")


def _drop_old_tables_qualified(engine, old_physical: list[tuple[str, str | None]]) -> None:
    # Reverse of swap order: dependents (facts, parts) before what they reference.
    for name, schema in reversed(old_physical):
        ref = f'"{schema}"."{name}"' if schema else f'"{name}"'
        try:
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {ref}"))
        except Exception:
            # Non-fatal: leftover table, cleaned up on a later run/sweep.
            log.warning("could not drop leftover table %s", ref, exc_info=True)
