"""Authoritative, report-first reconciliation of PostgreSQL sink objects.

Publication replaces generations of names it was explicitly given. It never
removes names omitted from a publish. This module is the deliberately separate
operation for callers that possess a *complete* desired state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from sqlalchemy import delete, inspect, select, text

from . import metadata as meta
from .engine import advisory_lock
from .manifest import DEFAULT_STORAGE_SCHEMA, DatasetMaterialization, DatasetV2, check_manifests
from .publish import PUBLISH_LOCK_KEY


class ReconciliationError(RuntimeError):
    """The requested reconciliation is unsafe or unsupported."""


@dataclass(frozen=True, order=True)
class ManagedObject:
    """One loader-owned catalog object or metadata row."""

    kind: str
    name: str
    schema: str | None = None
    owner: str | None = field(default=None, compare=True)

    @property
    def qualified_name(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


@dataclass(frozen=True)
class DesiredState:
    """The complete manifest and plain-table registry of an authoritative caller."""

    manifests: tuple[DatasetMaterialization | DatasetV2, ...] = ()
    tables: tuple[str, ...] = ()
    storage_schemas: tuple[str, ...] = (DEFAULT_STORAGE_SCHEMA,)

    def __post_init__(self) -> None:
        check_manifests(self.manifests)
        if len(set(self.tables)) != len(self.tables):
            raise ValueError("duplicate plain table names in desired state")
        collisions = set(self.tables) & {manifest.name for manifest in self.manifests}
        if collisions:
            raise ValueError(f"names cannot be both plain tables and datasets: {sorted(collisions)}")


def desired_state(value) -> DesiredState:
    """Normalize a complete registry; refuse ambiguous bare manifest lists."""
    if isinstance(value, DesiredState):
        return value
    if isinstance(value, Mapping):
        if "tables" not in value or not ({"manifests", "datasets"} & set(value)):
            raise ReconciliationError(
                "a desired-state mapping must contain both the complete 'tables' registry and 'manifests'/'datasets'"
            )
        manifests = value.get("manifests", value.get("datasets", ()))
        return DesiredState(
            tuple(manifests or ()),
            tuple(value.get("tables", ()) or ()),
            tuple(value.get("storage_schemas", (DEFAULT_STORAGE_SCHEMA,)) or ()),
        )
    raise TypeError("desired must be DesiredState or a complete mapping with 'manifests' and 'tables'")


_MARKERS = {
    "plain_marker": (meta.analytics_table_updates, "table_name"),
    "dataset_marker": (meta.analytics_dataset_updates, "dataset_name"),
    "contour_marker": (meta.contour_updates, "contour_table"),
    "component_marker": (meta.component_updates, "component_name"),
}


def _rows(engine, table) -> list[dict]:
    if not meta.table_exists(engine, table):
        return []
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(select(table)).mappings()]


def _desired_objects(state: DesiredState, view_schema: str) -> set[ManagedObject]:
    out: set[ManagedObject] = set()
    for name in state.tables:
        out |= {ManagedObject("plain", name, view_schema), ManagedObject("plain_marker", name)}
    for manifest in state.manifests:
        out |= {
            ManagedObject("dataset", manifest.name, view_schema),
            ManagedObject("dataset_marker", manifest.name),
        }
        if isinstance(manifest, DatasetV2):
            for component in manifest.components:
                out |= {
                    ManagedObject("component", component.name, manifest.schema_of(component)),
                    ManagedObject("component_marker", component.name),
                    ManagedObject("component_link", component.name, owner=manifest.name),
                }
        else:
            schema = manifest.compact_schema
            out |= {
                ManagedObject("component", meta.compact_table_name(manifest.name), schema, manifest.name),
                ManagedObject("component", manifest.contour_table, schema),
                ManagedObject("contour_marker", manifest.contour_table),
            }
            if manifest.keyed:
                out.add(ManagedObject("component", manifest.zone_table, schema))
    return out


def _catalog_kind(insp, name: str, schema: str) -> str | None:
    if name in set(insp.get_materialized_view_names(schema=schema)):
        return "materialized"
    if name in set(insp.get_view_names(schema=schema)):
        return "view"
    if name in set(insp.get_table_names(schema=schema)):
        return "table"
    return None


def _inventory(
    engine, wanted: set[ManagedObject], view_schema: str, storage_schemas: tuple[str, ...]
) -> set[ManagedObject]:
    """Inventory names backed by markers, desired state, or private schemas."""
    marker_rows = {kind: _rows(engine, table) for kind, (table, _) in _MARKERS.items()}
    out: set[ManagedObject] = set()
    for kind, rows in marker_rows.items():
        _, key = _MARKERS[kind]
        out.update(ManagedObject(kind, row[key]) for row in rows)
    for row in _rows(engine, meta.dataset_components):
        out.add(ManagedObject("component_link", row["component_name"], owner=row["dataset_name"]))

    insp = inspect(engine)
    public_names = {
        o.name
        for o in out | wanted
        if o.kind in {"plain_marker", "dataset_marker", "plain", "dataset"}
    }
    for name in public_names:
        if _catalog_kind(insp, name, view_schema):
            kind = "plain" if ManagedObject("plain_marker", name) in out else "dataset"
            if ManagedObject("plain", name, view_schema) in wanted:
                kind = "plain"
            out.add(ManagedObject(kind, name, view_schema))

    component_schemas = {row["storage_schema"] for row in marker_rows["component_marker"]}
    component_schemas |= {o.schema for o in wanted if o.kind == "component" and o.schema}
    component_schemas.update(storage_schemas)
    known_schemas = set(insp.get_schema_names())
    for schema in sorted(component_schemas):
        if schema not in known_schemas:
            continue
        names = set(insp.get_table_names(schema=schema)) | set(insp.get_view_names(schema=schema))
        for name in names:
            if name.startswith(meta.STAGING_PREFIX) or "__old__" in name:
                continue
            owner = next(
                (o.owner for o in wanted if o.kind == "component" and o.name == name and o.schema == schema), None
            )
            out.add(ManagedObject("component", name, schema, owner))
    return out


@dataclass
class ReconciliationReport:
    desired: tuple[ManagedObject, ...]
    obsolete: tuple[ManagedObject, ...]
    missing: tuple[ManagedObject, ...]
    authoritative: bool = field(repr=False)
    _engine: object = field(repr=False)
    _view_schema: str = field(repr=False)

    @property
    def clean(self) -> bool:
        return not self.obsolete and not self.missing

    def to_dict(self) -> dict:
        """JSON-friendly report for command wrappers and report-only runs."""
        def objects(values):
            return [
                {
                    "kind": obj.kind,
                    "name": obj.name,
                    **({"schema": obj.schema} if obj.schema else {}),
                    **({"owner": obj.owner} if obj.owner else {}),
                }
                for obj in values
            ]

        return {
            "authoritative": self.authoritative,
            "desired": objects(self.desired),
            "obsolete": objects(self.obsolete),
            "missing": objects(self.missing),
        }

    def prune(self, execute: bool = False) -> "ReconciliationReport":
        """Return this dry run, or explicitly execute the obsolete deletions."""
        if not self.authoritative:
            raise ReconciliationError(
                "pruning requires reconcile(..., authoritative=True) with the complete manifest and plain-table registry"
            )
        if not execute:
            return self
        if self._engine.dialect.name != "postgresql":  # type: ignore[attr-defined]
            raise ReconciliationError("authoritative pruning is currently supported only on PostgreSQL")
        _execute_prune(self._engine, self.obsolete)
        return self


def reconcile(engine, desired, *, authoritative: bool = False, view_schema: str = "public") -> ReconciliationReport:
    """Return a mutation-free comparison of desired and loader-owned state."""
    if engine.dialect.name != "postgresql":
        raise ReconciliationError("authoritative reconciliation is currently supported only on PostgreSQL")
    state = desired_state(desired)
    wanted = _desired_objects(state, view_schema)
    actual = _inventory(engine, wanted, view_schema, state.storage_schemas)
    return ReconciliationReport(
        tuple(sorted(wanted)),
        tuple(sorted(actual - wanted)),
        tuple(sorted(wanted - actual)),
        authoritative,
        engine,
        view_schema,
    )


def _q(conn, value: str) -> str:
    return conn.dialect.identifier_preparer.quote(value)


def _drop_relation(conn, obj: ManagedObject) -> None:
    schema = obj.schema or "public"
    insp = inspect(conn)
    ref = f"{_q(conn, schema)}.{_q(conn, obj.name)}"
    if obj.name in set(insp.get_materialized_view_names(schema=schema)):
        conn.execute(text(f"DROP MATERIALIZED VIEW {ref}"))
    elif obj.name in set(insp.get_view_names(schema=schema)):
        conn.execute(text(f"DROP VIEW {ref}"))
    elif obj.name in set(insp.get_table_names(schema=schema)):
        conn.execute(text(f"DROP TABLE {ref}"))


def _execute_prune(engine, obsolete: Iterable[ManagedObject]) -> None:
    obsolete = tuple(obsolete)
    with engine.begin() as conn:
        with advisory_lock(conn, PUBLISH_LOCK_KEY, transactional=True):
            for kind in ("dataset", "plain"):
                for obj in obsolete:
                    if obj.kind == kind:
                        _drop_relation(conn, obj)
            # Drop mutually dependent obsolete component tables together.
            # External dependencies still abort because CASCADE is forbidden.
            by_schema: dict[str, list[ManagedObject]] = {}
            for obj in obsolete:
                if obj.kind == "component":
                    by_schema.setdefault(obj.schema or "public", []).append(obj)
            for schema, objects in by_schema.items():
                tables = set(inspect(conn).get_table_names(schema=schema))
                table_objects = [obj for obj in objects if obj.name in tables]
                for obj in objects:
                    if obj not in table_objects:
                        _drop_relation(conn, obj)
                if table_objects:
                    refs = ", ".join(
                        f"{_q(conn, schema)}.{_q(conn, obj.name)}" for obj in table_objects
                    )
                    conn.execute(text(f"DROP TABLE {refs}"))
            for obj in obsolete:
                if obj.kind == "component_link":
                    conn.execute(
                        delete(meta.dataset_components).where(
                            meta.dataset_components.c.dataset_name == obj.owner,
                            meta.dataset_components.c.component_name == obj.name,
                        )
                    )
                elif obj.kind in _MARKERS:
                    table, key = _MARKERS[obj.kind]
                    conn.execute(delete(table).where(table.c[key] == obj.name))
