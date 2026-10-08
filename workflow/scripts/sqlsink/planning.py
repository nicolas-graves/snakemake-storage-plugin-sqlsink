"""Read-only catalog snapshot for planning predicates.

One PostgreSQL transaction supplies marker rows, relation kinds and explicit
SELECT grants. The process-local snapshot is invalidated by the provider's
write hooks and workflow onstart hook. Callers must not cache it across staging
or publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text

from . import metadata as meta
from .manifest import DatasetV2
from .engine import get_engine


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


_shared_states: dict[str, tuple[frozenset[str], PlanningState]] = {}


def invalidate_shared_planning_state() -> None:
    _shared_states.clear()


def shared_planning_state(engine_or_url, *, roles=()) -> PlanningState:
    """Reuse one snapshot per URL, rereading only when new roles are needed."""
    engine = engine_or_url if hasattr(engine_or_url, "dialect") else get_engine(engine_or_url)
    key = engine.url.render_as_string(hide_password=False)
    requested = frozenset(roles)
    cached = _shared_states.get(key)
    if cached is not None and requested <= cached[0]:
        return cached[1]
    all_roles = requested | (cached[0] if cached else frozenset())
    state = read_planning_state(engine, roles=sorted(all_roles))
    _shared_states[key] = (all_roles, state)
    return state


def read_planning_state(engine, *, roles=()) -> PlanningState:
    """Read metadata in one bounded, read-only PostgreSQL transaction."""
    if engine.dialect.name != "postgresql":
        raise NotImplementedError("planning snapshots require PostgreSQL")
    marker_rows = {kind: {} for kind in MARKERS}
    with engine.connect() as conn:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
        conn.execute(text("SET LOCAL statement_timeout = '15s'"))
        roles = sorted(set(roles))
        catalog = conn.execute(text(
            "SELECT 'table' AS kind, to_jsonb(t) AS value FROM ("
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = current_schema() "
            "AND tablename LIKE '_pipeline_meta_%') t "
            "UNION ALL SELECT 'relation', to_jsonb(t) FROM ("
            "SELECT n.nspname, c.relname, c.relkind FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relkind IN ('r', 'p', 'v', 'm') "
            "AND n.nspname NOT IN ('pg_catalog', 'information_schema')) t "
            "UNION ALL SELECT 'grant', to_jsonb(t) FROM ("
            "SELECT r.rolname, n.nspname, c.relname FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "CROSS JOIN LATERAL aclexplode(COALESCE(c.relacl, acldefault('r', c.relowner))) a "
            "JOIN pg_catalog.pg_roles r ON r.oid = a.grantee "
            "WHERE r.rolname = ANY(:roles) AND a.privilege_type = 'SELECT') t"
        ), {"roles": roles})
        existing = set()
        relations = {}
        grants = set()
        for kind, row in catalog:
            if kind == "table":
                existing.add(row["tablename"])
            elif kind == "relation":
                relations[(row["nspname"], row["relname"])] = row["relkind"]
            else:
                grants.add((row["rolname"], row["nspname"], row["relname"]))
        queries = []
        for kind, (table, _) in MARKERS.items():
            if table.name in existing:
                queries.append(f"SELECT '{kind}' AS kind, to_jsonb(t) AS value FROM \"{table.name}\" t")
        if meta.dataset_components.name in existing:
            queries.append(f"SELECT 'dataset_components' AS kind, to_jsonb(t) AS value FROM \"{meta.dataset_components.name}\" t")
        dataset_components = {}
        if queries:
            for kind, row in conn.execute(text(" UNION ALL ".join(queries))):
                if kind == "dataset_components":
                    dataset_components.setdefault(row["dataset_name"], {})[row["component_name"]] = row["component_update_id"]
                else:
                    table, key = MARKERS[kind]
                    for column in table.columns:
                        if column.name in row and isinstance(row[column.name], str) and column.type.python_type is datetime:
                            row[column.name] = datetime.fromisoformat(row[column.name])
                    marker_rows[kind][row[key]] = row
        conn.rollback()
    return PlanningState(marker_rows, relations, grants, dataset_components)
