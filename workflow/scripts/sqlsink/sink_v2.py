"""SQL sink for manifest v2: components (fact / dimension / bridge tables) and
the view or materialized view defined over them.

Each component is staged and published on its own, like the shared contour of
the v1 form: it has a per-component lock, a marker pair (staged / published) and
a source fingerprint, whether one dataset or several use it. A dataset stages
nothing but its identity (its manifest hash and its components' update ids) and
the view SQL; the relation itself is created by `publish.publish_datasets`.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
from sqlalchemy import inspect, select

from . import metadata as meta_mod
from .engine import advisory_lock, bulk_load_streaming, ensure_schema, upsert_by_pk
from .fingerprint import LOADER_VERSION, TYPE_MAP_VERSION, components_digest
from .manifest import Component, DatasetV2, ManifestError
from .queries import fetch_row, source_sql
from .view import physical_refs, render_view_sql


class ComponentKeyError(ValueError):
    """A component's source violates its declared key (duplicate or NULL)."""


class ComponentColumnsError(ValueError):
    """A component's source lacks a column its declaration relies on."""


@dataclass
class ComponentStageReceipt:
    kind: str  # "component"
    component: str
    schema: str
    status: str  # "current" | "staged"
    update_id: str
    source_sha256: str
    row_count: int
    staging_table: str | None
    staged_at: str
    marker_update_id_at_stage_time: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ViewDatasetReceipt:
    kind: str  # "dataset_v2"
    dataset: str
    status: str
    update_id: str
    manifest_hash: str
    components: dict[str, str]  # component name -> component update_id
    components_digest: str
    materialize: str
    view_sql: str
    row_count: int
    staged_at: str
    marker_update_id_at_stage_time: str | None
    loader_version: int = LOADER_VERSION
    type_map_version: int = TYPE_MAP_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def component_select_sql(component: Component, source) -> str:
    cols = ", ".join(_q(c) for c in component.columns) if component.columns else "*"
    return f"SELECT {cols} FROM {source_sql(source)}"


def describe_columns(con: duckdb.DuckDBPyConnection, sql: str) -> list[str]:
    return list(con.sql(sql).columns)


def check_component_source(con: duckdb.DuckDBPyConnection, component: Component, sql: str) -> list[str]:
    """Columns of the component's query; rejects a missing key or priority
    column and a key that is not unique / has NULLs."""
    columns = describe_columns(con, sql)
    wanted = [*component.primary_key, *([component.priority] if component.priority else [])]
    missing = [c for c in wanted if c not in columns]
    if missing:
        raise ComponentColumnsError(f"component {component.name!r}: source has no column(s) {missing} (has {columns})")
    if component.primary_key:
        key = ", ".join(_q(c) for c in component.primary_key)
        nulls = fetch_row(
            con, f"SELECT COUNT(*) FROM ({sql}) WHERE " + " OR ".join(f"{_q(c)} IS NULL" for c in component.primary_key)
        )[0]
        if nulls:
            raise ComponentKeyError(f"component {component.name!r}: {nulls} row(s) with a NULL in primary key {list(component.primary_key)}")
        dups = fetch_row(con, f"SELECT COUNT(*) FROM (SELECT {key} FROM ({sql}) GROUP BY {key} HAVING COUNT(*) > 1)")[0]
        if dups:
            raise ComponentKeyError(f"component {component.name!r}: {dups} duplicated primary key value(s) {list(component.primary_key)}")
    return columns


def orphan_count_v2(con: duckdb.DuckDBPyConnection, manifest: DatasetV2, sources: dict[str, Any], columns: dict[str, list[str]]) -> dict[str, int]:
    """Per inner join: distinct left keys with no match on the right side (their
    rows would silently vanish from the view)."""
    out = {}
    for join in manifest.joins:
        if join.type != "inner":
            continue
        left = join.left or manifest.base
        assert left is not None
        lcols = ", ".join(_q(l) for l, _ in join.on)
        rcols = ", ".join(_q(r) for _, r in join.on)
        on = " AND ".join(f"a.{_q(l)} = b.{_q(r)}" for l, r in join.on)
        sql = (
            f"SELECT COUNT(*) FROM (SELECT DISTINCT {lcols} FROM ({component_select_sql(manifest.component(left), sources[left])})) a "
            f"ANTI JOIN (SELECT DISTINCT {rcols} FROM ({component_select_sql(manifest.component(join.component), sources[join.component])})) b ON {on}"
        )
        count = fetch_row(con, sql)[0]
        if count:
            out[join.component] = count
    return out


def fetch_component_marker(engine, name: str) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(select(meta_mod.component_updates).where(meta_mod.component_updates.c.component_name == name)).mappings().first()
        return dict(row) if row else None


def fetch_staged_component_marker(engine, name: str) -> dict | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(meta_mod.staged_component_updates).where(meta_mod.staged_component_updates.c.component_name == name))
            .mappings()
            .first()
        )
        return dict(row) if row else None


def published_dataset_components(engine, dataset_name: str) -> dict[str, str]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(meta_mod.dataset_components.c.component_name, meta_mod.dataset_components.c.component_update_id).where(
                meta_mod.dataset_components.c.dataset_name == dataset_name
            )
        ).all()
    return {r[0]: r[1] for r in rows}


def dependents_of(engine, component_names) -> set[str]:
    """Published v2 datasets built over any of `component_names`."""
    names = list(component_names)
    if not names:
        return set()
    with engine.connect() as conn:
        rows = conn.execute(
            select(meta_mod.dataset_components.c.dataset_name).where(meta_mod.dataset_components.c.component_name.in_(names))
        ).all()
    return {r[0] for r in rows}


class _CursorAdapter:
    """`execute()` returning the cursor, as DuckDB's own connection does (the
    SQLAlchemy driver cursor returns None)."""

    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql: str):
        self._cursor.execute(sql)
        return self._cursor


class V2SinkMixin:
    """The v2 half of `SqlSink` (`self.engine`, `self.view_schema`)."""

    engine: Any
    view_schema: str | None

    # -- components -----------------------------------------------------

    def _component_physical(self, name: str, schema: str) -> tuple[str, str | None]:
        return meta_mod.physical_name_and_schema(self.engine.dialect.name, name, schema)

    def _has_component_table(self, name: str, schema: str) -> bool:
        pname, pschema = self._component_physical(name, schema)
        return inspect(self.engine).has_table(pname, schema=pschema)

    def stage_component(
        self, component: Component, schema: str, source: Any, con: duckdb.DuckDBPyConnection, source_sha256: str
    ) -> ComponentStageReceipt:
        """Stage one component into a private staging table. Idempotent, and
        serialized per component (a shared component may be staged by several
        datasets' jobs at once)."""
        with self.engine.connect() as lock_conn:
            with advisory_lock(lock_conn, f"snakemake_sql:stage_component:{schema}.{component.name}"):
                return self._stage_component(component, schema, source, con, source_sha256)

    def _stage_component(self, component, schema, source, con, source_sha256) -> ComponentStageReceipt:
        from .fingerprint import compute_component_update_id
        from .sink_postgres import _constrained_table, _duckdb_schema

        engine = self.engine
        update_id = compute_component_update_id(component.definition_hash(), source_sha256)
        published = fetch_component_marker(engine, component.name)

        def receipt(status: str, staging: str | None, rows: int) -> ComponentStageReceipt:
            return ComponentStageReceipt(
                kind="component",
                component=component.name,
                schema=schema,
                status=status,
                update_id=update_id,
                source_sha256=source_sha256,
                row_count=rows,
                staging_table=staging,
                staged_at=_now(),
                marker_update_id_at_stage_time=published["update_id"] if published else None,
            )

        if published and published["update_id"] == update_id and self._has_component_table(component.name, schema):
            return receipt("current", None, published["row_count"])
        staging_bare = meta_mod.staging_name(component.name)
        staged = fetch_staged_component_marker(engine, component.name)
        if staged and staged["update_id"] == update_id and self._has_component_table(staging_bare, schema):
            return receipt("staged", staging_bare, staged["row_count"])

        sql = component_select_sql(component, source)
        check_component_source(con, component, sql)
        columns = _duckdb_schema(con, sql)
        staging_name, staging_schema = self._component_physical(staging_bare, schema)
        table = _constrained_table(
            staging_name, columns, staging_schema, primary_key=tuple(component.primary_key), dialect=engine.dialect.name
        )
        with engine.begin() as conn:
            ensure_schema(conn, staging_schema)
            table.drop(conn, checkfirst=True)
            table.create(conn)
            rows = bulk_load_streaming(conn, table, con.execute(sql), [c for c, _ in columns])
            upsert_by_pk(
                conn,
                meta_mod.staged_component_updates,
                "component_name",
                {
                    "component_name": component.name,
                    "update_id": update_id,
                    "source_sha256": source_sha256,
                    "storage_schema": schema,
                    "row_count": rows,
                },
                now_col="staged_at",
            )
        return receipt("staged", staging_name, rows)

    # -- datasets ---------------------------------------------------------

    def _relation_kinds(self) -> tuple[set[str], set[str], set[str]]:
        """(tables, views, materialized views) of the view schema."""
        insp = inspect(self.engine)
        views = set(insp.get_view_names(schema=self.view_schema))
        tables = set(insp.get_table_names(schema=self.view_schema)) - views
        matviews = set(insp.get_materialized_view_names(schema=self.view_schema)) if self.engine.dialect.name == "postgresql" else set()
        return tables, views, matviews

    def published_intact_v2(self, manifest: DatasetV2) -> bool:
        """Every component table and the public relation, of the kind the
        manifest asks for (a materialized view is a table on DuckDB)."""
        if not all(self._has_component_table(c.name, manifest.schema_of(c)) for c in manifest.components):
            return False
        tables, views, matviews = self._relation_kinds()
        if manifest.materialize == "view":
            return manifest.name in views
        return manifest.name in (matviews if self.engine.dialect.name == "postgresql" else tables)

    def stage_view_dataset(self, dataset, con: duckdb.DuckDBPyConnection) -> ViewDatasetReceipt:
        """Decide "current" vs "staged" for a v2 dataset and render its view
        SQL for this sink's dialect. Before anything is published, rejects inner
        joins that would silently drop fact rows."""
        from .sink import OrphanFactsError

        manifest: DatasetV2 = dataset.manifest
        engine = self.engine
        marker = self._dataset_marker(manifest.name)
        columns = {c.name: (list(c.columns) if c.columns else describe_columns(con, component_select_sql(c, dataset.sources[c.name]))) for c in manifest.components}
        current = bool(marker and marker["update_id"] == dataset.update_id and self.published_intact_v2(manifest))
        if not current and manifest.joins:
            orphans = orphan_count_v2(con, manifest, dataset.sources, columns)
            if orphans:
                raise OrphanFactsError(
                    f"{manifest.name}: distinct join key(s) without a match (inner join would drop rows): {orphans}"
                )
        view_sql = render_view_sql(manifest, engine.dialect.name, physical_refs(manifest, engine.dialect.name), columns)
        return ViewDatasetReceipt(
            kind="dataset_v2",
            dataset=manifest.name,
            status="current" if current else "staged",
            update_id=dataset.update_id,
            manifest_hash=manifest.manifest_hash(),
            components=dict(dataset.component_update_ids),
            components_digest=components_digest(dataset.component_update_ids),
            materialize=manifest.materialize,
            view_sql=view_sql,
            row_count=marker["row_count"] if marker and current else 0,
            staged_at=_now(),
            marker_update_id_at_stage_time=marker["update_id"] if marker else None,
        )

    def _dataset_marker(self, name: str) -> dict | None:
        from .sink_postgres import fetch_dataset_marker

        return fetch_dataset_marker(self.engine, name)

    # -- reading published relations in place -----------------------------

    @contextlib.contextmanager
    def analysis_session(self, *, threads: int | None = 2, memory_limit: str | None = None):
        """Yield `(con, ref)`: a DuckDB connection from which published relations
        can be read in place, and `ref(name)`, the expression addressing one
        (a view, a materialized view or a table). PostgreSQL: a private DuckDB
        session with the database attached read-only. DuckDB sink: the engine's
        own connection (the file is already open and cannot be attached twice),
        whose settings are left alone."""
        from .sink import _sql_string, duckdb_session

        if self.engine.dialect.name == "postgresql":
            from .sink_postgres import pg_attach

            with duckdb_session(threads, None, self.spill_dir()) as con:  # type: ignore[attr-defined]
                if memory_limit is not None:
                    con.execute(f"SET memory_limit = {_sql_string(memory_limit)}")
                pg_attach(con, self.engine.url)
                schema = self.view_schema or "public"
                yield con, lambda name: f'pg."{schema}"."{name}"'
            return
        with self.engine.connect() as conn:
            cursor = conn.connection.driver_connection.cursor()
            try:
                yield _CursorAdapter(cursor), lambda name: f'"{name}"'
            finally:
                cursor.close()
