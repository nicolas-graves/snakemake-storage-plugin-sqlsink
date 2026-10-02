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

from sqlalchemy import delete, func, insert, inspect, select, text, update

from . import metadata as meta_mod
from . import keep_old as keep_old_mod
from .engine import (
    advisory_lock,
    apply_lock_timeout,
    resolve_analyze,
    resolve_keep_old,
    resolve_lock_timeout,
    upsert_by_pk,
)
from .manifest import DatasetMaterialization, DatasetV2

log = logging.getLogger(__name__)

PUBLISH_LOCK_KEY = "snakemake_sql:publish_tables"


class PublishConflict(RuntimeError):
    """Raised when a table's marker changed between staging and publish
    (another process published a newer version first)."""


def _refresh_markers(conn, table, pk_col: str, names: list[str]) -> None:
    """Set `published_at` to the transaction's `now()` on the existing marker
    rows of `names`. `update_id` and every other column are left alone."""
    if names:
        conn.execute(update(table).where(table.c[pk_col].in_(names)).values(published_at=func.now()))


def _demote_published(
    conn,
    receipts: list[dict],
    marker_table,
    key_col: str,
    receipt_key: str,
    version_col: str = "update_id",
    receipt_version: str = "update_id",
    live_exists=None,
) -> list[dict]:
    """`receipts` with every "staged" one whose version already is the live
    marker's rewritten to "current" (a copy). Swapping it would be a no-op and
    its staging table is gone, so a receipt left on disk by an earlier publish
    must not be published again. A staged receipt for any other version is left
    alone and still hits the usual conflict checks. `live_exists(receipt)`, if
    given, must also hold: a marker that outlived a dropped relation is not
    "already published". Call under the publish lock."""
    keys = [r[receipt_key] for r in receipts if r["status"] == "staged"]
    if not keys:
        return list(receipts)
    live = dict(
        conn.execute(
            select(marker_table.c[key_col], marker_table.c[version_col]).where(marker_table.c[key_col].in_(keys))
        ).all()
    )
    out = []
    for r in receipts:
        if (
            r["status"] == "staged"
            and live.get(r[receipt_key]) == r[receipt_version]
            and (live_exists is None or live_exists(r))
        ):
            log.info("%s %r already published, skipping", receipt_key, r[receipt_key])
            r = dict(r, status="current")
        out.append(r)
    return out


def publish_tables(
    engine,
    receipts: list[dict],
    *,
    refresh: bool = False,
    keep_old: bool | None = None,
    lock_timeout: str | int | None = None,
    analyze: bool | None = None,
) -> list[str]:
    """Publish every `status == "staged"` receipt in one transaction.

    Returns the list of table names actually published (skips tables that
    were already current).

    With `refresh=True`, the marker of every table in `receipts` (staged or
    already current) also gets `published_at = now()` in the same transaction,
    so all of them carry one identical timestamp. A workflow rule with one
    `published/{table}` storage output per table needs this: Snakemake
    compares each input with the *oldest* output of a job, so outputs left at
    older publish times would make the job rerun forever.

    `keep_old` (default off; `SQLSINK_KEEP_OLD`) keeps every replaced table aside
    as `__old__<name>` instead of dropping it after the commit (see `keep_old.py`
    for rollback and cleanup). `lock_timeout` (default: `SQLSINK_LOCK_TIMEOUT`,
    else the server's setting) is applied as `SET LOCAL lock_timeout` on
    PostgreSQL: a publish that cannot get its locks in time fails and rolls back.

    `analyze` (default on; `SQLSINK_ANALYZE=0` to disable) runs `ANALYZE` on
    every table actually published, on PostgreSQL only, after the publish
    transaction has committed (see `_analyze_relations`).

    A staged receipt whose version is already live is treated as current, which
    makes a publish idempotent over receipts left on disk by an earlier publish.
    """
    if not any(r["status"] == "staged" for r in receipts) and not refresh:
        return []
    keep = resolve_keep_old(keep_old)
    lock_timeout = resolve_lock_timeout(lock_timeout)
    do_analyze = resolve_analyze(analyze)
    meta_mod.create_all(engine)  # a publish is a write path: the only place besides staging that creates markers

    published: list[str] = []
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    old_names: list[str] = []

    with engine.begin() as conn:
        apply_lock_timeout(conn, lock_timeout)
        with advisory_lock(conn, PUBLISH_LOCK_KEY, transactional=True):
            insp = inspect(conn)
            receipts = _demote_published(
                conn, receipts, meta_mod.analytics_table_updates, "table_name", "table",
                live_exists=lambda r: insp.has_table(r["table"]),
            )
            staged = [r for r in receipts if r["status"] == "staged"]
            inspector = inspect(conn)
            for receipt in staged:
                table_name = receipt["table"]
                _assert_marker_unchanged_since_staging(conn, table_name, receipt)

                staging_name = meta_mod.staging_name(table_name)
                if keep:
                    keep_old_mod.park(conn, table_name, None)
                elif inspector.has_table(table_name):
                    old_name = f"{table_name}__old__{ts}"
                    conn.execute(text(f'ALTER TABLE "{table_name}" RENAME TO "{old_name}"'))
                    old_names.append(old_name)

                conn.execute(text(f'ALTER TABLE "{staging_name}" RENAME TO "{table_name}"'))
                _upsert_marker(conn, receipt)
                published.append(table_name)

            if refresh:
                _refresh_markers(
                    conn, meta_mod.analytics_table_updates, "table_name", [r["table"] for r in receipts]
                )

    # Outside the publish transaction and before cleanup: the latency fix
    # (fresh statistics) should land as soon as possible after commit, ahead
    # of the best-effort drop of old tables below, which can itself wait on
    # a lock.
    if do_analyze:
        _analyze_relations(engine, [(t, None) for t in published], lock_timeout)

    # Best-effort, non-transactional cleanup: failures here are logged,
    # not fatal — a leftover `*__old__*` table is wasted disk, not a
    # correctness problem, and leaving it around briefly gives a human a
    # window to inspect it if a downstream smoke test fails.
    _drop_old_tables(engine, old_names, lock_timeout)

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


def _drop_old_tables(engine, old_names: list[str], lock_timeout: str | None = None) -> None:
    for name in old_names:
        try:
            with engine.begin() as conn:
                apply_lock_timeout(conn, lock_timeout)
                conn.execute(text(f'DROP TABLE IF EXISTS "{name}"'))
        except Exception:
            # Non-fatal: leftover table, cleaned up on a later run/sweep.
            log.warning("could not drop leftover table %s", name, exc_info=True)


def _analyze_relations(engine, relations: list[tuple[str, str | None]], lock_timeout: str | None = None) -> None:
    """`ANALYZE` each of `relations` (bare_name, schema_or_None), one at a
    time, each in its own transaction opened *after* the publish transaction
    has already committed -- never inside it, and never inside the same
    transaction as another relation's `ANALYZE`. This is what keeps
    gathering fresh statistics from ever extending how long the publish's
    advisory lock (or the brief `ACCESS EXCLUSIVE` lock the rename-swap
    itself takes) is held: by the time the first `ANALYZE` runs, that lock
    is already gone.

    A no-op on every dialect but PostgreSQL: DuckDB has its own `ANALYZE`,
    but computes a table's statistics as part of writing it (`CREATE TABLE
    AS` / the bulk loader builds them alongside the data), so a freshly
    published table there is never in PostgreSQL's post-COPY,
    statistics-less state -- there is nothing stale for a post-publish step
    to fix.

    Right after a bulk `COPY` into a freshly renamed PostgreSQL table,
    autovacuum has not run yet, so the planner has only default or stale
    estimates for it until it does, on its own schedule -- a real, measured
    multi-second query-latency regression right after a publish. This
    closes that window immediately instead of waiting.

    Best-effort, like `_drop_old_tables`: the new relation is live and
    correct either way, so a failure here (e.g. a lock-timeout raise if this
    relation is contended) is logged, not fatal -- it just means the planner
    stays under-informed about this one relation until autovacuum eventually
    gets to it.
    """
    if engine.dialect.name != "postgresql" or not relations:
        return
    seen: set[tuple[str, str | None]] = set()
    for name, schema in relations:
        key = (name, schema)
        if key in seen:
            continue
        seen.add(key)
        ref = f'"{schema}"."{name}"' if schema else f'"{name}"'
        try:
            with engine.begin() as conn:
                apply_lock_timeout(conn, lock_timeout)
                conn.execute(text(f"ANALYZE {ref}"))
        except Exception:
            log.warning("could not ANALYZE %s", ref, exc_info=True)


def publish_datasets(
    engine,
    manifests: list,
    dataset_receipts: list[dict],
    contour_receipts: list[dict],
    *,
    refresh: bool = False,
    component_receipts: list[dict] = (),  # type: ignore[assignment]
    keep_old: bool | None = None,
    lock_timeout: str | int | None = None,
    analyze: bool | None = None,
) -> list[str]:
    """Publish every staged dataset and any staged shared table, in one
    transaction under the same advisory lock as `publish_tables`.

    v1 datasets: compact fact table + public compatibility view, and staged
    shared contour tables. v2 datasets (`kind == "dataset_v2"`): staged
    components (facts, dimensions, bridges) and the public view or
    materialized view over them, which is recreated in the same transaction
    (so a materialized view is rebuilt from the new components atomically).

    `refresh=True` also sets `published_at = now()` on the markers of every
    dataset, contour and component in the receipts, staged or current (see
    `publish_tables`).

    Shared tables are published before the datasets that read them, so a
    freshly created/replaced view never points at a stale or missing table.
    `manifests` must include every dataset in `dataset_receipts` and cover
    every contour table referenced by them.

    A view is bound to the *table object* it reads: after a component is
    swapped, every view over it (the datasets built on it, including ones in
    other manifests) must be recreated or it would keep reading the old
    table. Such a dataset that is not restaged in this publish is a
    `PublishConflict`.

    `keep_old` and `lock_timeout`: see `publish_tables`. With `keep_old`, every
    replaced component, flat table and (PostgreSQL) view is renamed to
    `__old__<name>` in its schema and made unreadable to other roles, instead of
    being dropped; `keep_old.rollback_kept_old` / `cleanup_kept_old` act on them.

    `analyze`: see `publish_tables`. Every swapped-in component, compact
    fact, contour and zone table is analyzed; a swapped-in v2 public
    relation is too, unless it is a plain view (nothing to analyze -- its
    components already are) rather than a table or materialized view.

    A staged receipt whose version is already live is treated as current (see
    `publish_tables`), so a partial restage publishes only what really changed.
    """
    component_receipts = list(component_receipts)
    contour_receipts = list(contour_receipts)
    dataset_receipts = list(dataset_receipts)
    if not refresh and not any(
        r["status"] == "staged" for r in (*contour_receipts, *component_receipts, *dataset_receipts)
    ):
        return []
    keep = resolve_keep_old(keep_old)
    lock_timeout = resolve_lock_timeout(lock_timeout)
    do_analyze = resolve_analyze(analyze)
    meta_mod.create_all(engine)  # a publish is a write path (see publish_tables)

    v1_manifests = [m for m in manifests if isinstance(m, DatasetMaterialization)]
    v2_by_name = {m.name: m for m in manifests if isinstance(m, DatasetV2)}
    manifest_by_name = {m.name: m for m in v1_manifests}
    contour_schema = {m.contour_table: m.compact_schema for m in v1_manifests}
    zone_table = {m.contour_table: m.zone_table for m in v1_manifests if m.keyed}

    published: list[str] = []
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    old_physical: list[tuple[str, str | None]] = []
    fresh: list[tuple[str, str | None]] = []

    with engine.begin() as conn:
        apply_lock_timeout(conn, lock_timeout)
        with advisory_lock(conn, PUBLISH_LOCK_KEY, transactional=True):
            dialect = conn.engine.dialect.name

            insp = inspect(conn)

            def physical_exists(bare, schema):
                name, sch = meta_mod.physical_name_and_schema(dialect, bare, schema)
                return insp.has_table(name, schema=sch)

            def public_exists(r):
                return r["dataset"] in set(insp.get_table_names()) | set(insp.get_view_names()) | (
                    set(insp.get_materialized_view_names()) if dialect == "postgresql" else set()
                )

            contour_receipts = _demote_published(
                conn, contour_receipts, meta_mod.contour_updates, "contour_table", "contour_table",
                "contour_sha256", "contour_sha256",
                live_exists=lambda r: r["contour_table"] not in contour_schema
                or physical_exists(r["contour_table"], contour_schema[r["contour_table"]]),
            )
            component_receipts = _demote_published(
                conn, component_receipts, meta_mod.component_updates, "component_name", "component",
                live_exists=lambda r: physical_exists(r["component"], r["schema"]),
            )
            dataset_receipts = _demote_published(
                conn, dataset_receipts, meta_mod.analytics_dataset_updates, "dataset_name", "dataset",
                live_exists=public_exists,
            )
            staged_contours = {r["contour_table"]: r for r in contour_receipts if r["status"] == "staged"}
            staged_components = {r["component"]: r for r in component_receipts if r["status"] == "staged"}
            staged_all = [r for r in dataset_receipts if r["status"] == "staged"]
            staged_datasets = [r for r in staged_all if r.get("kind") != "dataset_v2"]
            staged_views = [r for r in staged_all if r.get("kind") == "dataset_v2"]

            _assert_keyed_dependents_restaged(v1_manifests, staged_contours, staged_datasets)
            _assert_view_dependents_restaged(conn, v2_by_name, staged_components, staged_views)

            for name, receipt in staged_components.items():
                _assert_component_marker_unchanged_since_staging(conn, name, receipt)
                _rename_swap_physical(
                    conn, dialect, name, receipt["schema"], meta_mod.staging_name(name), old_physical, ts, keep, fresh
                )
                _upsert_component_marker(conn, receipt)

            for contour_table, receipt in staged_contours.items():
                schema = contour_schema[contour_table]
                zone = zone_table.get(contour_table)
                if zone:
                    _rename_swap_physical(
                        conn, dialect, zone, schema, meta_mod.staging_name(zone), old_physical, ts, keep, fresh
                    )
                _rename_swap_physical(
                    conn, dialect, contour_table, schema, meta_mod.staging_name(contour_table), old_physical, ts, keep, fresh
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
                    keep,
                    fresh,
                )

                _replace_compatibility_view(conn, dataset_name, receipt["view_sql"], old_physical, ts, keep)
                _upsert_dataset_marker(conn, receipt)
                published.append(dataset_name)

            for receipt in staged_views:
                dataset_name = receipt["dataset"]
                _assert_dataset_marker_unchanged_since_staging(conn, dataset_name, receipt)
                _assert_components_live(conn, receipt)
                _replace_public_relation(
                    conn, dialect, dataset_name, receipt["view_sql"], receipt["materialize"], old_physical, ts, keep, fresh
                )
                _upsert_view_dataset_marker(conn, receipt)
                published.append(dataset_name)

            if refresh:
                _refresh_markers(
                    conn, meta_mod.contour_updates, "contour_table", [r["contour_table"] for r in contour_receipts]
                )
                _refresh_markers(
                    conn, meta_mod.component_updates, "component_name", [r["component"] for r in component_receipts]
                )
                _refresh_markers(
                    conn, meta_mod.analytics_dataset_updates, "dataset_name", [r["dataset"] for r in dataset_receipts]
                )

    if do_analyze:
        _analyze_relations(engine, fresh, lock_timeout)

    _drop_old_tables_qualified(engine, old_physical, lock_timeout)
    return published


def _assert_view_dependents_restaged(conn, v2_by_name, staged_components, staged_views) -> None:
    """Every published or declared v2 dataset over a component that is being
    swapped must be republished in the same transaction (see `publish_datasets`)."""
    for name in sorted({r["dataset"] for r in staged_views} - set(v2_by_name)):
        raise PublishConflict(f"dataset {name!r} is staged but its manifest was not passed to publish")
    if not staged_components:
        return
    swapped = set(staged_components)
    dependents = {m.name for m in v2_by_name.values() if swapped & set(m.component_names())}
    rows = conn.execute(
        select(meta_mod.dataset_components.c.dataset_name).where(meta_mod.dataset_components.c.component_name.in_(list(swapped)))
    ).all()
    dependents |= {r[0] for r in rows}
    restaged = {r["dataset"] for r in staged_views}
    for name in sorted(dependents - restaged):
        raise PublishConflict(
            f"component(s) {sorted(swapped)} are being replaced but dataset {name!r}, whose view reads them, "
            "is not restaged in this publish (its view would keep reading the replaced table)"
        )


def _assert_components_live(conn, receipt: dict) -> None:
    """The view about to be created claims to be built over `receipt["components"]`
    (component -> update_id). Each must be what the live component table now
    is (after this transaction's swaps), or the dataset marker would vouch for
    content the view does not read: a component receipt was missing, stale, or
    lost a merge to a "current" one."""
    live = dict(
        conn.execute(select(meta_mod.component_updates.c.component_name, meta_mod.component_updates.c.update_id)).all()
    )
    for name, update_id in receipt["components"].items():
        if live.get(name) != update_id:
            raise PublishConflict(
                f"dataset {receipt['dataset']!r} was staged over component {name!r} version {update_id[:12]}, "
                f"but the published component is {str(live.get(name))[:12]}: its staged component receipt "
                "is missing from this publish (or stale)"
            )


def merge_component_receipts(receipts) -> dict[str, dict]:
    """One receipt per component from receipts of several datasets. A shared
    component appears once per dataset that uses it: a "staged" receipt wins over
    a "current" one, and two staged receipts must be the same version."""
    merged: dict[str, dict] = {}
    for receipt in receipts:
        name = receipt["component"]
        kept = merged.get(name)
        if kept is None or (kept["status"] == "current" and receipt["status"] == "staged"):
            merged[name] = receipt
        elif kept["status"] == "staged" == receipt["status"] and kept["update_id"] != receipt["update_id"]:
            raise PublishConflict(f"component {name!r} was staged in two different versions in one publish")
    return merged


def _assert_component_marker_unchanged_since_staging(conn, name: str, receipt: dict) -> None:
    current = conn.execute(
        select(meta_mod.component_updates.c.update_id).where(meta_mod.component_updates.c.component_name == name)
    ).scalar_one_or_none()
    staged_from = receipt.get("marker_update_id_at_stage_time")
    if current is not None and current != staged_from:
        raise PublishConflict(
            f"component {name!r} was republished by another process after this receipt was staged "
            f"(marker is {current!r}, receipt expected {staged_from!r}); aborting the whole publish transaction"
        )


def _upsert_component_marker(conn, receipt: dict) -> None:
    values = {
        "component_name": receipt["component"],
        "update_id": receipt["update_id"],
        "source_sha256": receipt["source_sha256"],
        "storage_schema": receipt["schema"],
        "row_count": receipt["row_count"],
    }
    upsert_by_pk(conn, meta_mod.component_updates, "component_name", values, now_col="published_at")


def _replace_public_relation(
    conn,
    dialect: str,
    name: str,
    select_sql: str,
    materialize: str,
    old_physical: list[tuple[str, str | None]],
    ts: str,
    keep: bool = False,
    fresh: list[tuple[str, str | None]] | None = None,
) -> None:
    """Point the public name at a fresh view (`materialize == "view"`) or
    materialized view. Whatever is there is replaced, of whatever kind: a
    view, a materialized view, or a plain table (the pre-existing flat table
    of a migrated dataset, kept aside as `__old__` and dropped after commit).
    DuckDB has no materialized views: there it is a table built by `CREATE
    TABLE AS`, in the same transaction. With `keep`, whatever is there is kept
    aside as `__old__<name>` instead (`keep_old.park`).

    `fresh`, if given and `materialize != "view"`, gets `(name, None)`
    appended: a materialized view or table just got fresh data and is what
    `_analyze_relations` should later `ANALYZE`. A plain view holds no data
    of its own -- `ANALYZE` on a view is not meaningful in PostgreSQL, and
    the components it reads are analyzed on their own swap -- so it is never
    added."""
    inspector = inspect(conn)
    views = set(inspector.get_view_names())
    tables = set(inspector.get_table_names()) - views
    matviews = set(inspector.get_materialized_view_names()) if dialect == "postgresql" else set()
    if keep:
        keep_old_mod.park(conn, name, None)
    elif name in matviews:
        conn.execute(text(f'DROP MATERIALIZED VIEW "{name}"'))
    elif name in views:
        conn.execute(text(f'DROP VIEW "{name}"'))
    elif name in tables:
        old_name = f"{name}__old__{ts}"
        conn.execute(text(f'ALTER TABLE "{name}" RENAME TO "{old_name}"'))
        old_physical.append((old_name, None))
    if materialize == "view":
        conn.execute(text(f'CREATE VIEW "{name}" AS {select_sql}'))
    elif dialect == "postgresql":
        conn.execute(text(f'CREATE MATERIALIZED VIEW "{name}" AS {select_sql}'))
    else:
        conn.execute(text(f'CREATE TABLE "{name}" AS {select_sql}'))
    if materialize != "view" and fresh is not None:
        fresh.append((name, None))


def _upsert_view_dataset_marker(conn, receipt: dict) -> None:
    """Dataset marker (composite digests in the columns v1 uses for its two
    source hashes) and the component list that finds this dataset's view when
    one of its components is swapped later."""
    values = {
        "dataset_name": receipt["dataset"],
        "update_id": receipt["update_id"],
        "manifest_hash": receipt["manifest_hash"],
        "fact_sha256": receipt["components_digest"],
        "contour_sha256": "components",
        "loader_version": receipt.get("loader_version", 1),
        "type_map_version": receipt.get("type_map_version", 1),
        "row_count": receipt["row_count"],
        "published_by": receipt.get("published_by", "pipeline"),
    }
    upsert_by_pk(conn, meta_mod.analytics_dataset_updates, "dataset_name", values, now_col="published_at")
    table = meta_mod.dataset_components
    conn.execute(delete(table).where(table.c.dataset_name == receipt["dataset"]))
    for component, update_id in receipt["components"].items():
        conn.execute(
            insert(table).values(
                dataset_name=receipt["dataset"],
                component_name=component,
                component_update_id=update_id,
                materialize=receipt["materialize"],
            )
        )


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
    keep: bool = False,
    fresh: list[tuple[str, str | None]] | None = None,
) -> None:
    """Rename-swap a schema-aware physical table into place, the same
    all-or-nothing pattern as `publish_tables`' plain-name rename, extended
    to a real PostgreSQL schema where the dialect supports one.

    `fresh`, if given, gets `(live_name, live_schema)` appended: the relation
    now holds the new data, so it is what `_analyze_relations` should later
    `ANALYZE` (outside this transaction, once it has committed)."""
    live_name, live_schema = meta_mod.physical_name_and_schema(dialect, bare_name, schema)
    staging_name, staging_schema = meta_mod.physical_name_and_schema(dialect, staging_bare_name, schema)
    live_ref = f'"{live_schema}"."{live_name}"' if live_schema else f'"{live_name}"'
    staging_ref = f'"{staging_schema}"."{staging_name}"' if staging_schema else f'"{staging_name}"'

    inspector = inspect(conn)
    if keep:
        keep_old_mod.park(conn, live_name, live_schema)
    elif inspector.has_table(live_name, schema=live_schema):
        old_name = f"{live_name}__old__{ts}"
        conn.execute(text(f'ALTER TABLE {live_ref} RENAME TO "{old_name}"'))
        old_physical.append((old_name, live_schema))

    conn.execute(text(f'ALTER TABLE {staging_ref} RENAME TO "{live_name}"'))
    if fresh is not None:
        fresh.append((live_name, live_schema))


def _replace_compatibility_view(
    conn, dataset_name: str, view_sql: str, old_physical: list[tuple[str, str | None]], ts: str, keep: bool = False
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

    if keep:
        keep_old_mod.park(conn, dataset_name, None)
    elif is_table and not is_view:
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


def _drop_old_tables_qualified(
    engine, old_physical: list[tuple[str, str | None]], lock_timeout: str | None = None
) -> None:
    # Reverse of swap order: dependents (facts, parts) before what they reference.
    for name, schema in reversed(old_physical):
        ref = f'"{schema}"."{name}"' if schema else f'"{name}"'
        try:
            with engine.begin() as conn:
                apply_lock_timeout(conn, lock_timeout)
                conn.execute(text(f"DROP TABLE IF EXISTS {ref}"))
        except Exception:
            # Non-fatal: leftover table, cleaned up on a later run/sweep.
            log.warning("could not drop leftover table %s", ref, exc_info=True)
