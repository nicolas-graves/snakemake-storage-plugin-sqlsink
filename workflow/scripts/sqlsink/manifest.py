"""Declarative dataset/materialization manifest.

A `DatasetMaterialization` describes one *logical dataset* that has two
physical shapes: a Parquet export (handled entirely upstream, outside this
package) and a normalized PostgreSQL representation -- a compact fact table
without the repeated geometry column, a shared contour/dimension table, and
a public compatibility view that joins them back together under the
original dataset name, column names and order.

This is intentionally data-agnostic: join columns, the geometry column to
omit and the output column order are declared here explicitly, never
inferred from names, per the migration plan's correctness constraint.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .relations import Edge, Entity, FanOutError, JoinGraph, RelationError

MANIFEST_VERSION = 1
MANIFEST_VERSION_V2 = 2
PART_COLUMN = "part_no"


@dataclass(frozen=True)
class DatasetMaterialization:
    name: str
    geometry_column: str
    contour_table: str
    fact_join_columns: tuple[str, ...]
    contour_join_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    fact_source: str | None = None
    contour_source: str = "zone_emploi_contours"
    compact_schema: str = "analytics_storage"
    # Relational form: the contour becomes `zone` (PK: join columns) plus
    # `zone_part` (PK: join columns + part_no, FK -> zone), and the compact
    # facts get a foreign key to `zone`. Same view output either way.
    keyed: bool = False

    def __post_init__(self):
        if len(self.fact_join_columns) != len(self.contour_join_columns):
            raise ValueError(
                f"dataset {self.name!r}: fact_join_columns and contour_join_columns "
                f"must declare the same number of columns, in corresponding order "
                f"({self.fact_join_columns!r} vs {self.contour_join_columns!r})"
            )
        if self.geometry_column not in self.output_columns:
            raise ValueError(
                f"dataset {self.name!r}: geometry_column {self.geometry_column!r} "
                f"must appear in output_columns (it is reconstructed via the join, "
                f"not stored in the compact fact table)"
            )
        for col in self.fact_join_columns:
            if col not in self.output_columns:
                raise ValueError(
                    f"dataset {self.name!r}: fact join column {col!r} is not in output_columns"
                )

    def fact_parquet_key(self) -> str:
        return self.fact_source or self.name

    def compact_columns(self) -> tuple[str, ...]:
        """Compact fact columns: the declared output, minus the geometry
        column that the compatibility view reconstructs via the join."""
        return tuple(c for c in self.output_columns if c != self.geometry_column)

    @property
    def zone_table(self) -> str:
        return f"{self.contour_table}_zone"

    def relation_graph(self):
        """The declared relations of the keyed form: fact -> zone <- part."""
        from .relations import Edge, Entity, JoinGraph

        return JoinGraph(
            (
                Entity(self.name, ()),
                Entity(self.zone_table, self.contour_join_columns),
                Entity(self.contour_table, (*self.contour_join_columns, PART_COLUMN)),
            ),
            (
                Edge(self.name, self.zone_table, self.fact_join_columns),
                Edge(self.contour_table, self.zone_table, self.contour_join_columns),
            ),
        )

    def canonical_dict(self) -> dict:
        d = self._base_canonical_dict()
        if self.keyed:  # absent when False, so pre-existing markers stay valid
            d["keyed"] = True
        return d

    def _base_canonical_dict(self) -> dict:
        return {
            "manifest_version": MANIFEST_VERSION,
            "name": self.name,
            "geometry_column": self.geometry_column,
            "contour_table": self.contour_table,
            "contour_source": self.contour_source,
            "fact_join_columns": list(self.fact_join_columns),
            "contour_join_columns": list(self.contour_join_columns),
            "output_columns": list(self.output_columns),
            "compact_schema": self.compact_schema,
        }

    def manifest_hash(self) -> str:
        """Canonical hash of everything that determines the compact table's
        layout and the view's join/select. Changing a join column, a cast,
        the geometry column or the output order must change this hash so
        that it invalidates every publication that depends on it."""
        blob = json.dumps(self.canonical_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _load_manifest_v1(spec: dict) -> DatasetMaterialization:
    """Build a `DatasetMaterialization` from a plain dict (e.g. parsed from
    `config.yaml`), converting list fields to the tuples the dataclass
    expects."""
    kwargs = dict(spec)
    for key in ("fact_join_columns", "contour_join_columns", "output_columns"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return DatasetMaterialization(**kwargs)


# ---------------------------------------------------------------------------
# Manifest v2: components + a view over them.
#
# A dataset is a set of named *components* (facts, dimensions, bridges), each a
# physical table with its own primary key, source, lock, marker and fingerprint,
# and one public relation (a view, or a materialized view) defined over them.
# A component declared once at the top level of a manifest file is *shared*:
# any number of datasets reference it by name, and it is staged and published
# once, like `zone_emploi_contours` in the v1 form. v1 manifests keep their own
# class and code path (their markers and hashes are unchanged).
# ---------------------------------------------------------------------------

COMPONENT_KINDS = ("fact", "dimension", "bridge")
MATERIALIZE_MODES = ("view", "materialized")
JOIN_TYPES = ("inner", "left")
DEFAULT_STORAGE_SCHEMA = "analytics_storage"
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ManifestError(ValueError):
    """A v2 manifest is inconsistent."""


@dataclass(frozen=True)
class Component:
    """One physical table of a dataset.

    `source` is the key of its Parquet file (default: the component name);
    `source_table` instead names a plain table already published in the sink's
    own database. `priority` (bridges) names the explicit ordering column a
    tie-breaking view must order by: it is checked to exist at staging."""

    name: str
    kind: str
    primary_key: tuple[str, ...] = ()
    source: str | None = None
    source_table: str | None = None
    priority: str | None = None
    columns: tuple[str, ...] | None = None
    schema: str | None = None
    shared: bool = field(default=False, compare=False)

    def __post_init__(self):
        if not _IDENT.match(self.name):
            raise ManifestError(f"component name {self.name!r} must match [A-Za-z_][A-Za-z0-9_]*")
        if self.kind not in COMPONENT_KINDS:
            raise ManifestError(f"component {self.name!r}: kind must be one of {COMPONENT_KINDS}, not {self.kind!r}")
        if self.kind == "dimension" and not self.primary_key:
            raise ManifestError(f"component {self.name!r}: a dimension needs a primary_key")
        if self.source is not None and self.source_table is not None:
            raise ManifestError(f"component {self.name!r}: declare `source` or a table source, not both")
        if self.priority is not None and self.kind != "bridge":
            raise ManifestError(f"component {self.name!r}: `priority` only applies to a bridge")
        if len(set(self.primary_key)) != len(self.primary_key):
            raise ManifestError(f"component {self.name!r}: duplicate primary key column")
        if self.columns is not None:
            missing = [c for c in (*self.primary_key, *([self.priority] if self.priority else [])) if c not in self.columns]
            if missing:
                raise ManifestError(f"component {self.name!r}: {missing} not in declared columns")

    @property
    def source_key(self) -> str:
        return self.source or self.name

    def canonical_dict(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "kind": self.kind, "primary_key": list(self.primary_key)}
        if self.source is not None:
            d["source"] = self.source
        if self.source_table is not None:
            d["source_table"] = self.source_table
        if self.priority is not None:
            d["priority"] = self.priority
        if self.columns is not None:
            d["columns"] = list(self.columns)
        if self.schema is not None:
            d["schema"] = self.schema
        return d

    def definition_hash(self) -> str:
        """Identity of the component's *definition* (not its data)."""
        return hashlib.sha256(json.dumps(self.canonical_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class Join:
    """`left` (default: the base component) joined to `component` on `on`
    pairs (left column, component column). Many-to-one unless `fan_out` is
    declared: `on` must then cover the component's whole primary key."""

    component: str
    on: tuple[tuple[str, str], ...]
    left: str | None = None
    type: str = "inner"
    fan_out: bool = False


@dataclass(frozen=True)
class SelectItem:
    """One output column: `column` of `component` (default: found by name),
    published as `alias` (default: the column name)."""

    column: str
    component: str | None = None
    alias: str | None = None

    @property
    def output(self) -> str:
        return self.alias or self.column


@dataclass(frozen=True)
class DatasetV2:
    name: str
    components: tuple[Component, ...]
    base: str | None = None
    joins: tuple[Join, ...] = ()
    select: tuple[SelectItem, ...] = ()
    # dialect ("duckdb" | "postgresql" | "default") -> SQL template
    view_sql: tuple[tuple[str, str], ...] = ()
    materialize: str = "view"
    storage_schema: str = DEFAULT_STORAGE_SCHEMA

    def __post_init__(self):
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ManifestError(f"dataset {self.name!r}: duplicate component names")
        if self.materialize not in MATERIALIZE_MODES:
            raise ManifestError(f"dataset {self.name!r}: materialize must be one of {MATERIALIZE_MODES}")
        if bool(self.view_sql) == bool(self.joins):
            raise ManifestError(f"dataset {self.name!r}: declare exactly one of `joins` or `view_sql`")
        if self.view_sql:
            if self.select or self.base:
                raise ManifestError(f"dataset {self.name!r}: `select`/`base` belong to `joins`, not `view_sql`")
            for dialect, template in self.view_sql:
                if dialect not in ("duckdb", "postgresql", "default"):
                    raise ManifestError(f"dataset {self.name!r}: unknown view_sql dialect {dialect!r}")
                self._check_template(template)
        else:
            self._check_joins()

    def component(self, name: str) -> Component:
        for c in self.components:
            if c.name == name:
                return c
        raise ManifestError(f"dataset {self.name!r}: unknown component {name!r}")

    def schema_of(self, component: Component) -> str:
        return component.schema or self.storage_schema

    def _check_template(self, template: str) -> None:
        known = {c.name for c in self.components}
        for ref in TEMPLATE_REF.findall(template):
            if ref not in known:
                raise ManifestError(
                    f"dataset {self.name!r}: view_sql references {{{ref}}}, which is not one of its components {sorted(known)}"
                )

    def _check_joins(self) -> None:
        if self.base is None:
            raise ManifestError(f"dataset {self.name!r}: `joins` needs a `base` component")
        base = self.component(self.base)
        joined = {base.name}
        entities = tuple(Entity(c.name, c.primary_key) for c in self.components)
        edges: list[Edge] = []
        m2o_reachable = {base.name}
        for join in self.joins:
            target = self.component(join.component)
            left = join.left or base.name
            if left not in joined:
                raise ManifestError(f"dataset {self.name!r}: join to {target.name!r} starts from {left!r}, which is not yet joined")
            if target.name in joined:
                raise ManifestError(f"dataset {self.name!r}: component {target.name!r} is joined twice")
            if join.type not in JOIN_TYPES:
                raise ManifestError(f"dataset {self.name!r}: join type must be one of {JOIN_TYPES}")
            if not join.on:
                raise ManifestError(f"dataset {self.name!r}: join to {target.name!r} declares no columns")
            right_cols = tuple(r for _, r in join.on)
            covers_key = bool(target.primary_key) and set(right_cols) == set(target.primary_key)
            if not covers_key and not join.fan_out:
                raise FanOutError(
                    f"dataset {self.name!r}: joining {target.name!r} on {list(right_cols)} multiplies rows "
                    f"(its primary key is {list(target.primary_key)}); declare `fan_out: true` if that is intended"
                )
            if covers_key:
                by_right = {r: l for l, r in join.on}
                edges.append(Edge(left, target.name, tuple(by_right[k] for k in target.primary_key)))
                if left in m2o_reachable:
                    m2o_reachable.add(target.name)
            joined.add(target.name)
        graph = JoinGraph(entities, tuple(edges))
        for name in m2o_reachable - {base.name}:
            try:
                graph.path(base.name, name, max_hops=max(len(self.joins), 1))
            except RelationError as exc:
                raise ManifestError(f"dataset {self.name!r}: {exc}") from exc
        for item in self.select:
            if item.component is not None and item.component not in joined:
                raise ManifestError(f"dataset {self.name!r}: select {item.column!r} from unjoined component {item.component!r}")

    def canonical_dict(self) -> dict:
        d: dict[str, Any] = {
            "manifest_version": MANIFEST_VERSION_V2,
            "name": self.name,
            "components": [c.canonical_dict() for c in sorted(self.components, key=lambda c: c.name)],
            "materialize": self.materialize,
            "storage_schema": self.storage_schema,
        }
        if self.joins:
            d["base"] = self.base
            d["joins"] = [
                {"component": j.component, "on": [list(p) for p in j.on], "left": j.left, "type": j.type, "fan_out": j.fan_out}
                for j in self.joins
            ]
            d["select"] = [{"column": s.column, "component": s.component, "alias": s.alias} for s in self.select]
        else:
            d["view_sql"] = [list(p) for p in self.view_sql]
        return d

    def manifest_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical_dict(), sort_keys=True).encode()).hexdigest()

    def component_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.components)


TEMPLATE_REF = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def is_v2(spec: Mapping) -> bool:
    return "components" in spec or spec.get("manifest_version") == MANIFEST_VERSION_V2


def load_component(spec: Mapping, *, shared: bool = False) -> Component:
    kwargs = dict(spec)
    source = kwargs.pop("source", None)
    if isinstance(source, Mapping):
        if "table" in source:
            kwargs["source_table"] = source["table"]
        elif "parquet" in source:
            kwargs["source"] = source["parquet"]
        else:
            raise ManifestError(f"component {spec.get('name')!r}: source must be a name, {{table: ...}} or {{parquet: ...}}")
    elif source is not None:
        kwargs["source"] = source
    if "primary_key" in kwargs:
        kwargs["primary_key"] = tuple(kwargs["primary_key"])
    if kwargs.get("columns") is not None:
        kwargs["columns"] = tuple(kwargs["columns"])
    return Component(shared=shared, **kwargs)


def _load_join(spec: Mapping) -> Join:
    on = spec["on"]
    pairs = tuple(tuple(p) for p in on.items()) if isinstance(on, Mapping) else tuple(tuple(p) for p in on)
    return Join(
        component=spec["component"],
        on=pairs,  # type: ignore[arg-type]
        left=spec.get("left"),
        type=spec.get("type", "inner"),
        fan_out=bool(spec.get("fan_out", False)),
    )


def _load_select(item) -> SelectItem:
    if isinstance(item, str):
        return SelectItem(item)
    return SelectItem(item["column"], item.get("from") or item.get("component"), item.get("as") or item.get("alias"))


def load_dataset_v2(spec: Mapping, shared: Mapping[str, Component] | None = None) -> DatasetV2:
    shared = shared or {}
    components = []
    for entry in spec["components"]:
        if isinstance(entry, str):
            if entry not in shared:
                raise ManifestError(f"dataset {spec.get('name')!r}: component {entry!r} is not declared at the top level")
            components.append(shared[entry])
        else:
            if entry.get("name") in shared:
                raise ManifestError(
                    f"dataset {spec.get('name')!r}: component {entry['name']!r} is already shared; reference it by name"
                )
            components.append(load_component(entry))
    view = spec.get("view", {})
    view_sql = view.get("view_sql", spec.get("view_sql"))
    if isinstance(view_sql, str):
        view_sql_t: tuple[tuple[str, str], ...] = (("default", view_sql),)
    elif view_sql:
        view_sql_t = tuple(sorted(dict(view_sql).items()))
    else:
        view_sql_t = ()
    joins = tuple(_load_join(j) for j in view.get("joins", spec.get("joins", ())))
    select = tuple(_load_select(i) for i in view.get("select", spec.get("select", ())))
    return DatasetV2(
        name=spec["name"],
        components=tuple(components),
        base=view.get("base", spec.get("base")),
        joins=joins,
        select=select,
        view_sql=view_sql_t,
        materialize=spec.get("materialize", "view"),
        storage_schema=spec.get("storage_schema", DEFAULT_STORAGE_SCHEMA),
    )


def load_shared_components(specs: Sequence[Mapping] | None) -> dict[str, Component]:
    """The top-level shared components of a manifest file, by name."""
    shared: dict[str, Component] = {}
    for spec in specs or ():
        component = load_component(spec, shared=True)
        if component.name in shared:
            raise ManifestError(f"shared component {component.name!r} declared twice")
        shared[component.name] = component
    return shared


def load_manifests(data: Any) -> list:
    """Every dataset of a manifest file: a list of specs (v1 and/or v2), or a
    mapping with `datasets` and, for v2, top-level shared `components`.

    Components sharing a name across datasets must be the same definition; a
    v2 component may not take the physical name of a v1 compact or contour table."""
    if isinstance(data, Mapping):
        specs = data.get("datasets", []) or []
        shared_specs = data.get("components", []) or []
    else:
        specs, shared_specs = data or [], []
    shared = load_shared_components(shared_specs)
    manifests = []
    for spec in specs:
        manifests.append(load_dataset_v2(spec, shared) if is_v2(spec) else load_manifest(spec))
    check_manifests(manifests)
    return manifests


def check_manifests(manifests: Sequence) -> None:
    seen: dict[str, Component] = {}
    schema_of_name: dict[str, str] = {}
    names = [m.name for m in manifests]
    if len(set(names)) != len(names):
        raise ManifestError("duplicate dataset names")
    physical: set[str] = set()
    for m in manifests:
        if isinstance(m, DatasetMaterialization):
            physical.add(m.contour_table)
            physical.add(f"compact__{m.name}")
            if m.keyed:
                physical.add(m.zone_table)
    for m in manifests:
        if not isinstance(m, DatasetV2):
            continue
        for c in m.components:
            key = f"{m.schema_of(c)}.{c.name}"
            if schema_of_name.setdefault(c.name, m.schema_of(c)) != m.schema_of(c):
                raise ManifestError(f"component {c.name!r} is placed in two schemas (component markers are keyed by name)")
            if key in seen and seen[key].canonical_dict() != c.canonical_dict():
                raise ManifestError(f"component {c.name!r} is defined differently by two datasets")
            seen[key] = c
            if c.name in physical or c.name in names:
                raise ManifestError(f"component {c.name!r} collides with another dataset's table or view name")


def load_manifest(spec: dict, shared: Mapping[str, Component] | None = None):
    """Build a manifest from a plain dict (e.g. parsed from `config.yaml`):
    v1 (`DatasetMaterialization`) unless it declares `components` (v2)."""
    if is_v2(spec):
        return load_dataset_v2(spec, shared)
    return _load_manifest_v1(spec)
