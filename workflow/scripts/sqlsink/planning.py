"""Read-only catalog snapshot for planning predicates.

One PostgreSQL transaction supplies marker rows, relation kinds and explicit
SELECT grants. A snapshot belongs to one planning pass; callers must not cache
it across staging or publication.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text

from . import metadata as meta
from .manifest import DatasetV2


MARKERS = {
    "table": (meta.analytics_table_updates, "table_name"),
    "staged": (meta.staged_table_updates, "table_name"),
    "dataset": (meta.analytics_dataset_updates, "dataset_name"),
    "component": (meta.component_updates, "component_name"),
}


@dataclass
class PlanningState:
    markers: dict[str, dict[str, dict]]
    relations: dict[tuple[str, str], str]
    grants: set[tuple[str, str, str]]
    dataset_components: dict[str, dict[str, str]] | None = None

    def marker(self, kind: str, name: str) -> dict | None:
        return self.markers[kind].get(name)

    def has(self, name: str, schema: str = "public", kind: str = "r") -> bool:
        actual = self.relations.get((schema, name))
        return actual == kind or (kind == "r" and actual == "p")

    def table_current(self, name: str, update_id: str) -> bool:
        published = self.marker("table", name)
        staged = self.marker("staged", name)
        return bool(
            (published and published["update_id"] == update_id and self.has(name))
            or (staged and staged["update_id"] == update_id and self.has(meta.staging_name(name)))
        )

    def published_table_current(self, name: str) -> bool:
        published = self.marker("table", name)
        staged = self.marker("staged", name)
        return bool(published and self.has(name) and (not staged or staged["update_id"] == published["update_id"]))

    def dataset_intact(self, manifest, view_schema: str = "public") -> bool:
        if isinstance(manifest, DatasetV2):
            for component in manifest.components:
                if not self.has(component.name, manifest.schema_of(component)):
                    return False
            if self.dataset_components is not None:
                recorded = self.dataset_components.get(manifest.name, {})
                if set(recorded) != {component.name for component in manifest.components}:
                    return False
                for component in manifest.components:
                    marker = self.marker("component", component.name)
                    if marker is None or marker["update_id"] != recorded[component.name]:
                        return False
            kind = "v" if manifest.materialize == "view" else "m"
            return self.has(manifest.name, view_schema, kind)
        names = [meta.compact_table_name(manifest.name), manifest.contour_table]
        if manifest.keyed:
            names.append(manifest.zone_table)
        return all(self.has(name, manifest.compact_schema) for name in names) and self.has(manifest.name, view_schema, "v")

    def role_has_select(self, role: str, relations) -> bool:
        for relation in relations:
            schema, _, name = relation.rpartition(".")
            if (role, schema or "public", name) not in self.grants:
                return False
        return True


def read_planning_state(engine, *, roles=()) -> PlanningState:
    """Read metadata in one bounded, read-only PostgreSQL transaction."""
    if engine.dialect.name != "postgresql":
        raise NotImplementedError("planning snapshots require PostgreSQL")
    marker_rows = {kind: {} for kind in MARKERS}
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        conn.execute(text("SET LOCAL statement_timeout = '15s'"))
        existing = set(conn.execute(text(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = current_schema() "
            "AND tablename LIKE '_pipeline_meta_%'"
        )).scalars())
        for kind, (table, key) in MARKERS.items():
            if table.name in existing:
                marker_rows[kind] = {row[key]: dict(row) for row in conn.execute(select(table)).mappings()}
        dataset_components = {}
        if meta.dataset_components.name in existing:
            for row in conn.execute(select(meta.dataset_components)).mappings():
                dataset_components.setdefault(row["dataset_name"], {})[row["component_name"]] = row["component_update_id"]
        relations = {(row[0], row[1]): row[2] for row in conn.execute(text(
            "SELECT n.nspname, c.relname, c.relkind FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relkind IN ('r', 'p', 'v', 'm') "
            "AND n.nspname NOT IN ('pg_catalog', 'information_schema')"
        ))}
        grants = set()
        roles = sorted(set(roles))
        if roles:
            grants = {(row[0], row[1], row[2]) for row in conn.execute(text(
                "SELECT r.rolname, n.nspname, c.relname FROM pg_catalog.pg_class c "
                "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                "CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl, acldefault('r', c.relowner))) a "
                "JOIN pg_catalog.pg_roles r ON r.oid = a.grantee "
                "WHERE r.rolname = ANY(:roles) AND a.privilege_type = 'SELECT'"
            ), {"roles": roles})}
        conn.rollback()
    return PlanningState(marker_rows, relations, grants, dataset_components)
