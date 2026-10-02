"""SQL sink (PostgreSQL or DuckDB): compact fact table + shared contour table + public
compatibility view, streamed in bounded batches
(see `engine.bulk_load_streaming`).

Mirrors `stage.py`'s split between "staged" (written to a private staging
table, private marker updated) and "published" (rename-swapped into the
public objects, public marker updated) -- `publish.publish_datasets` is the
only place that ever touches a public table, view or marker.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

import duckdb
from sqlalchemy import Column, ForeignKeyConstraint, Index, MetaData, PrimaryKeyConstraint, Table, Text, inspect, select, text

from . import metadata as meta_mod
from .engine import advisory_lock, bulk_load_streaming, ensure_schema, upsert_by_pk
from .fingerprint import LOADER_VERSION, TYPE_MAP_VERSION
from .manifest import PART_COLUMN, DatasetMaterialization, DatasetV2
from .publish import publish_datasets
from .queries import parts_select_sql, view_select_sql, zone_select_sql
from .sink import ContourSource, NormalizedDataset, default_spill_dir
from .sink_v2 import V2SinkMixin
from .sqlident import qualified, quote_ident
from .stage import _DUCKDB_TO_SA, fetch_marker


@dataclass
class DatasetStageReceipt:
    kind: str  # always "dataset", so publish_tables/publish_datasets can dispatch on receipt shape
    dataset: str
    status: str  # "current" | "staged"
    update_id: str
    manifest_hash: str
    fact_sha256: str
    contour_sha256: str
    row_count: int
    staging_compact_table: str | None
    view_sql: str
    staged_at: str
    marker_update_id_at_stage_time: str | None
    loader_version: int = LOADER_VERSION
    type_map_version: int = TYPE_MAP_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ContourStageReceipt:
    kind: str
    contour_table: str
    status: str
    contour_sha256: str
    row_count: int
    staging_table: str | None
    staged_at: str

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _duckdb_schema(con: duckdb.DuckDBPyConnection, sql: str) -> list[tuple[str, str]]:
    rel = con.sql(sql)
    return [(name, str(typ)) for name, typ in zip(rel.columns, rel.types)]


def _table_object_for(name: str, columns: list[tuple[str, str]], schema: str | None = None) -> Table:
    import sqlalchemy as sa

    md = MetaData()
    cols = []
    for col_name, duck_type in columns:
        sa_type_name = _DUCKDB_TO_SA.get(str(duck_type))
        if sa_type_name is None:
            raise TypeError(f"column {col_name!r}: no SQL type mapping for DuckDB type {duck_type}")
        cols.append(Column(col_name, getattr(sa, sa_type_name), autoincrement=False))
    return Table(name, md, *cols, schema=schema)


def _ref(name: str, schema: str | None) -> str:
    return qualified(name, schema)


def _constrained_table(
    name: str,
    columns: list[tuple[str, str]],
    schema: str | None,
    *,
    primary_key: tuple[str, ...] = (),
    foreign_key: tuple[tuple[str, ...], str | None, str, tuple[str, ...]] | None = None,
    index: tuple[str, ...] = (),
    index_tag: str = "",
    dialect: str = "postgresql",
) -> Table:
    """A staging table object carrying its keys from the start, so the
    database enforces them while loading. `foreign_key` is (columns,
    referenced schema, referenced table, referenced columns). A foreign key
    binds to the referenced table by identity, so it follows that table
    through the rename-swap.

    DuckDB cannot rename a table that a foreign key references or that
    carries an index, and the rename-swap needs both, so there the foreign
    key and the index are left undeclared. Orphan facts are still rejected
    before anything is written (`sink.check_no_orphans`)."""
    if dialect == "duckdb":
        foreign_key, index = None, ()
    table = _table_object_for(name, columns, schema=schema)
    if primary_key:
        table.append_constraint(PrimaryKeyConstraint(*primary_key))
    if foreign_key:
        cols, ref_schema, ref_table, ref_cols = foreign_key
        ref = f"{ref_schema}.{ref_table}" if ref_schema else ref_table
        if ref not in table.metadata.tables:
            # A stub so SQLAlchemy can resolve the reference; only `table` is
            # ever created, the referenced table already exists.
            Table(ref_table, table.metadata, *[Column(c, Text) for c in ref_cols], schema=ref_schema)
        table.append_constraint(ForeignKeyConstraint(list(cols), [f"{ref}.{c}" for c in ref_cols]))
    if index:
        # An index keeps its name through the rename-swap and its name is
        # schema-scoped, so each version needs its own or the next staging
        # table would collide with the live one's.
        Index(f"ix__{index_tag}__{'_'.join(index)}"[:63], *[table.c[c] for c in index])
    return table


def fetch_dataset_marker(engine, dataset_name: str) -> dict | None:
    return meta_mod.fetch_one(engine, meta_mod.analytics_dataset_updates, "dataset_name", dataset_name)


def fetch_staged_dataset_marker(engine, dataset_name: str) -> dict | None:
    return meta_mod.fetch_one(engine, meta_mod.staged_dataset_updates, "dataset_name", dataset_name)


def fetch_contour_marker(engine, contour_table: str) -> dict | None:
    return meta_mod.fetch_one(engine, meta_mod.contour_updates, "contour_table", contour_table)


def fetch_staged_contour_marker(engine, contour_table: str) -> dict | None:
    return meta_mod.fetch_one(engine, meta_mod.staged_contour_updates, "contour_table", contour_table)


def _libpq_value(value) -> str:
    # libpq keyword/value syntax: single-quoted, with \' and \\ escapes.
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def pg_conninfo(url) -> str:
    """libpq keyword/value connection string for a SQLAlchemy PostgreSQL URL.

    Every query parameter of the DSN (`sslmode`, `sslrootcert`, `options`,
    `host=` for a socket directory, ...) is passed through, as the URL's own
    driver gets it: dropping them would make the scanner connect with libpq's
    defaults, i.e. without the TLS verification the DSN asks for. A repeated
    key (multi-host DSN) is joined with commas, as libpq expects."""
    pairs = [
        ("dbname", url.database),
        ("user", url.username),
        ("password", url.password),
        ("host", url.host),
        ("port", url.port),
    ]
    for key, value in url.query.items():
        pairs.append((key, ",".join(value) if isinstance(value, tuple) else value))
    return " ".join(f"{key}={_libpq_value(value)}" for key, value in pairs if value is not None)


def pg_attach(con: duckdb.DuckDBPyConnection, url, alias: str = "pg") -> None:
    """Attach a PostgreSQL database read-only to a DuckDB connection. A failed
    attach is re-raised with the password masked: DuckDB's own message quotes
    the whole connection string."""
    if not url.get_backend_name().startswith("postgres"):
        raise ValueError("a PostgreSQL URL is needed")
    conninfo = pg_conninfo(url)
    con.execute("INSTALL postgres; LOAD postgres;")
    escaped = conninfo.replace("'", "''")
    try:
        con.execute(f"ATTACH '{escaped}' AS {quote_ident(alias)} (TYPE postgres, READ_ONLY)")
        return
    except duckdb.Error as err:
        error_type, message = type(err), str(err)
    if url.password:
        for secret in {str(url.password), _libpq_value(url.password)[1:-1]}:
            message = message.replace(secret, "***")
    # Raised outside the `except` block: the original exception must not ride
    # along as `__context__` (which `from None` only hides from the printout).
    raise error_type(message)


class SqlSink(V2SinkMixin):
    """Sink over any SQLAlchemy engine on a supported dialect: PostgreSQL
    (production) or a DuckDB database file (local deployment)."""

    # Columns go through the SQLAlchemy type mapping, so the joined
    # relation's types are not comparable one-to-one with the source Parquet.
    preserves_types = False
    name = "sql"

    def __init__(self, engine, *, view_schema: str | None = None):
        self.engine = engine
        # PostgreSQL views live in `public`; on DuckDB they live in the
        # connection's default schema unless the sink spec names one.
        self.view_schema = view_schema or ("public" if engine.dialect.name == "postgresql" else None)

    def spill_dir(self) -> str:
        return default_spill_dir()

    def current_update_id(self, dataset_name: str) -> str | None:
        marker = fetch_dataset_marker(self.engine, dataset_name)
        return marker["update_id"] if marker else None

    def _has_table(self, bare_name: str, manifest: DatasetMaterialization) -> bool:
        name, schema = meta_mod.physical_name_and_schema(self.engine.dialect.name, bare_name, manifest.compact_schema)
        return inspect(self.engine).has_table(name, schema=schema)

    def published_intact(self, manifest: DatasetMaterialization | DatasetV2) -> bool:
        """True if every published object a marker vouches for is still in the
        database: compact table, contour (and zone) table, public view (v2:
        every component table and the view or materialized view). A
        marker outlives a table dropped or restored out of band, and trusting
        it alone would never regenerate what is gone."""
        if isinstance(manifest, DatasetV2):
            return self.published_intact_v2(manifest)
        bare = [meta_mod.compact_table_name(manifest.name), manifest.contour_table]
        if manifest.keyed:
            bare.append(manifest.zone_table)
        return all(self._has_table(b, manifest) for b in bare) and manifest.name in inspect(
            self.engine
        ).get_view_names(schema=self.view_schema)

    def stage_contours(self, contour: ContourSource, con: duckdb.DuckDBPyConnection) -> ContourStageReceipt:
        """Stage the shared contour/dimension relation, independently of any
        one dataset's fact Parquet. Idempotent: a second dataset that shares
        the same contour table and an unchanged contour Parquet is a no-op.

        Datasets sharing a contour table stage the same physical tables, and
        the marker checks below are check-then-act, so concurrent stagings of
        one contour table are serialized under an advisory lock."""
        meta_mod.create_all(self.engine)  # staging is a write path; planning never creates markers
        with self.engine.connect() as lock_conn:
            with advisory_lock(lock_conn, f"snakemake_sql:stage_contours:{contour.manifest.contour_table}"):
                return self._stage_contours(contour, con)

    def _stage_contours(self, contour: ContourSource, con: duckdb.DuckDBPyConnection) -> ContourStageReceipt:
        engine, manifest = self.engine, contour.manifest
        contour_sha256 = contour.sha256
        published = fetch_contour_marker(engine, manifest.contour_table)
        live_bare = [manifest.contour_table, *([manifest.zone_table] if manifest.keyed else [])]
        if (
            published is not None
            and published["contour_sha256"] == contour_sha256
            and all(self._has_table(b, manifest) for b in live_bare)
        ):
            return ContourStageReceipt(
                kind="contour",
                contour_table=manifest.contour_table,
                status="current",
                contour_sha256=contour_sha256,
                row_count=published["row_count"],
                staging_table=None,
                staged_at=_now(),
            )

        staged = fetch_staged_contour_marker(engine, manifest.contour_table)
        if (
            staged is not None
            and staged["contour_sha256"] == contour_sha256
            and self._has_table(meta_mod.staging_name(manifest.contour_table), manifest)
        ):
            return ContourStageReceipt(
                kind="contour",
                contour_table=manifest.contour_table,
                status="staged",
                contour_sha256=contour_sha256,
                row_count=staged["row_count"],
                staging_table=meta_mod.staging_name(manifest.contour_table),
                staged_at=_now(),
            )

        keyed = manifest.keyed
        sql = parts_select_sql(manifest, contour.path) if keyed else contour.query
        columns = _duckdb_schema(con, sql)
        staging_bare = meta_mod.staging_name(manifest.contour_table)
        staging_name, staging_schema = meta_mod.physical_name_and_schema(
            engine.dialect.name, staging_bare, manifest.compact_schema
        )
        zone_bare = meta_mod.staging_name(manifest.zone_table)
        zone_name, zone_schema = meta_mod.physical_name_and_schema(
            engine.dialect.name, zone_bare, manifest.compact_schema
        )
        key = tuple(manifest.contour_join_columns)
        if keyed:
            zone_columns = _duckdb_schema(con, zone_select_sql(manifest, contour.path))
            zone_table = _constrained_table(
                zone_name, zone_columns, zone_schema, primary_key=key, dialect=engine.dialect.name
            )
            staging_table = _constrained_table(
                staging_name,
                columns,
                staging_schema,
                primary_key=(*key, PART_COLUMN),
                foreign_key=(key, zone_schema, zone_name, key),
                dialect=engine.dialect.name,
            )
        else:
            staging_table = _table_object_for(staging_name, columns, schema=staging_schema)

        with engine.begin() as conn:
            ensure_schema(conn, staging_schema)
            # Children first: the parts reference the zone.
            staging_table.drop(conn, checkfirst=True)
            if keyed:
                zone_table.drop(conn, checkfirst=True)
                zone_table.create(conn)
                bulk_load_streaming(
                    conn, zone_table, con.execute(zone_select_sql(manifest, contour.path)), [c for c, _ in zone_columns]
                )
            staging_table.create(conn)
            cur = con.execute(sql)
            row_count = bulk_load_streaming(conn, staging_table, cur, [c for c, _ in columns])
            upsert_by_pk(
                conn,
                meta_mod.staged_contour_updates,
                "contour_table",
                {
                    "contour_table": manifest.contour_table,
                    "contour_sha256": contour_sha256,
                    "row_count": row_count,
                },
                now_col="staged_at",
            )

        return ContourStageReceipt(
            kind="contour",
            contour_table=manifest.contour_table,
            status="staged",
            contour_sha256=contour_sha256,
            row_count=row_count,
            staging_table=staging_name,
            staged_at=_now(),
        )

    def stage_facts(self, dataset: NormalizedDataset, con: duckdb.DuckDBPyConnection) -> DatasetStageReceipt:
        """Project away the geometry column and deduplicate the remaining
        columns to recover the logical facts, entirely in DuckDB, then stream
        them into a private staging table. Expanded geometry never leaves
        DuckDB and is never sent to PostgreSQL.
        """
        engine, manifest = self.engine, dataset.manifest
        update_id = dataset.update_id
        meta_mod.create_all(engine)  # staging is a write path; planning never creates markers
        marker = fetch_dataset_marker(engine, manifest.name)

        def receipt(status, staging_compact_table, row_count) -> DatasetStageReceipt:
            return DatasetStageReceipt(
                kind="dataset",
                dataset=manifest.name,
                status=status,
                update_id=update_id,
                manifest_hash=manifest.manifest_hash(),
                fact_sha256=dataset.fact_sha256,
                contour_sha256=dataset.contour.sha256,
                row_count=row_count,
                staging_compact_table=staging_compact_table,
                view_sql=view_select_sql(manifest, engine.dialect.name),
                staged_at=_now(),
                marker_update_id_at_stage_time=marker["update_id"] if marker else None,
            )

        if marker is not None and marker["update_id"] == update_id and self.published_intact(manifest):
            return receipt("current", None, marker["row_count"])

        staging_bare = meta_mod.staging_name(meta_mod.compact_table_name(manifest.name))
        staged = fetch_staged_dataset_marker(engine, manifest.name)
        if staged is not None and staged["update_id"] == update_id and self._has_table(staging_bare, manifest):
            return receipt("staged", staging_bare, staged["row_count"])

        project_sql = dataset.compact_query
        columns = _duckdb_schema(con, project_sql)
        staging_name, staging_schema = meta_mod.physical_name_and_schema(
            engine.dialect.name, staging_bare, manifest.compact_schema
        )
        if manifest.keyed:
            key = tuple(manifest.fact_join_columns)
            zone_ref = self._zone_reference(manifest, dataset.contour.sha256)
            staging_table = _constrained_table(
                staging_name,
                columns,
                staging_schema,
                foreign_key=(key, zone_ref[0], zone_ref[1], tuple(manifest.contour_join_columns)),
                index=key,
                index_tag=f"{meta_mod.compact_table_name(manifest.name)}_{update_id[:10]}",
                dialect=engine.dialect.name,
            )
        else:
            staging_table = _table_object_for(staging_name, columns, schema=staging_schema)

        with engine.begin() as conn:
            ensure_schema(conn, staging_schema)
            staging_table.drop(conn, checkfirst=True)
            staging_table.create(conn)
            cur = con.execute(project_sql)
            row_count = bulk_load_streaming(conn, staging_table, cur, [c for c, _ in columns])
            upsert_by_pk(
                conn,
                meta_mod.staged_dataset_updates,
                "dataset_name",
                {
                    "dataset_name": manifest.name,
                    "update_id": update_id,
                    "manifest_hash": manifest.manifest_hash(),
                    "fact_sha256": dataset.fact_sha256,
                    "contour_sha256": dataset.contour.sha256,
                    "row_count": row_count,
                },
                now_col="staged_at",
            )

        return receipt("staged", staging_name, row_count)

    def _zone_reference(self, manifest: DatasetMaterialization, contour_sha256: str) -> tuple[str | None, str]:
        """(schema, table) of the zone table the facts must reference: the
        staged one when this exact contour is staged and not yet published,
        otherwise the live one."""
        dialect = self.engine.dialect.name
        staged = fetch_staged_contour_marker(self.engine, manifest.contour_table)
        live = fetch_contour_marker(self.engine, manifest.contour_table)
        staged_name, staged_schema = meta_mod.physical_name_and_schema(
            dialect, meta_mod.staging_name(manifest.zone_table), manifest.compact_schema
        )
        live_name, live_schema = meta_mod.physical_name_and_schema(dialect, manifest.zone_table, manifest.compact_schema)
        if live is not None and live["contour_sha256"] == contour_sha256:
            return live_schema, live_name
        if staged is not None and staged["contour_sha256"] == contour_sha256:
            return staged_schema, staged_name
        raise RuntimeError(
            f"contours of {manifest.contour_table!r} must be staged before the facts of {manifest.name!r}"
        )

    def publish(
        self,
        manifests: list[DatasetMaterialization],
        dataset_receipts: list[dict],
        contour_receipts: list[dict],
        *,
        refresh: bool = False,
        component_receipts: list[dict] = (),  # type: ignore[assignment]
        keep_old: bool | None = None,
        lock_timeout: str | int | None = None,
        analyze: bool | None = None,
    ) -> list[str]:
        return publish_datasets(
            self.engine,
            manifests,
            dataset_receipts,
            contour_receipts,
            refresh=refresh,
            component_receipts=list(component_receipts),
            keep_old=keep_old,
            lock_timeout=lock_timeout,
            analyze=analyze,
        )

    def table_source(self, table: str):
        """A plain table published in this sink (by `stage_table`) as a component source."""
        from .queries import SinkTableSource

        marker = fetch_marker(self.engine, table)
        if marker is None:
            raise LookupError(f"table {table!r} has not been published in this sink; it cannot be a component source")
        return SinkTableSource(table, marker["update_id"], self.engine, schema=self.view_schema)

    def reconcile(self, desired, authoritative: bool = False):
        """Report drift against a complete desired state without mutation.

        Publication intentionally never prunes omitted names. Call
        ``report.prune(execute=True)`` only after reconciling the complete
        registry with ``authoritative=True``. PostgreSQL is supported first;
        DuckDB reconciliation is intentionally out of scope.
        """
        from .reconcile import reconcile

        return reconcile(
            self.engine,
            desired,
            authoritative=authoritative,
            view_schema=self.view_schema or "public",
        )

    def joined_relation(self, manifest: DatasetMaterialization, con: duckdb.DuckDBPyConnection) -> str:
        """The public compatibility view. On PostgreSQL it is read through
        DuckDB's postgres scanner (nothing is copied). On DuckDB the database
        file is held open by the engine and cannot be attached a second time,
        so the view is copied into the session through the engine's own
        connection, keeping DuckDB's column types."""
        if self.engine.dialect.name == "postgresql":
            pg_attach(con, self.engine.url)
            return f"pg.{qualified(manifest.name, self.view_schema)}"

        view = quote_ident(manifest.name)
        with self.engine.connect() as conn:
            cur = conn.connection.driver_connection.cursor()
            cur.execute(f"DESCRIBE SELECT * FROM {view}")
            described = cur.fetchall()
            cur.execute(f"SELECT * FROM {view}")
            rows = cur.fetchall()
        table = quote_ident(f"sink_joined_{manifest.name}")
        column_defs = ", ".join(f"{quote_ident(name)} {duck_type}" for name, duck_type, *_ in described)
        con.execute(f"CREATE OR REPLACE TABLE {table} ({column_defs})")
        marks = ", ".join("?" for _ in described)
        if rows:
            con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
        return table


PostgresSink = SqlSink  # backwards-compatible name
