"""Entities, keys and a derived join graph. Pure: no I/O, no database.

A relation is declared once, as an `Edge` from a child entity to the parent
entity whose primary key its columns reference. Everything else is derived:

* a chain of many-to-one edges is many-to-one, so a fact reaches a
  grandparent (fact -> zone -> region) with its grain preserved;
* the reverse of an edge is one-to-many, a fan-out: it multiplies rows and is
  only offered when explicitly requested;
* a second route to the same entity is ambiguous and is an error unless the
  caller picks one (`via`);
* multi-hop paths are capped (default 2 hops, as dbt MetricFlow does).

The same rules apply whether the graph is rendered as PostgreSQL constraints,
as views, or as validation for Parquet (see `schema.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

MANY_TO_ONE = "many_to_one"
ONE_TO_ONE = "one_to_one"
ONE_TO_MANY = "one_to_many"
MANY_TO_MANY = "many_to_many"

DEFAULT_MAX_HOPS = 2


class RelationError(ValueError):
    """The declared relations are inconsistent, or a requested path is unsafe."""


class AmbiguousPathError(RelationError):
    """Several distinct routes lead to the same entity."""


class FanOutError(RelationError):
    """The only route multiplies rows (a one-to-many hop) and was not allowed."""


@dataclass(frozen=True)
class Entity:
    name: str
    primary_key: tuple[str, ...]

    # An entity with no primary key (a deduplicated fact table) can reference
    # others but can never be referenced: `JoinGraph` rejects an edge into it.


@dataclass(frozen=True)
class Edge:
    """`child.columns` reference `parent`'s primary key, positionally.

    `nullable=False` is required: a NULL key silently breaks a chain (the row
    drops out of every inner join beyond it)."""

    child: str
    parent: str
    columns: tuple[str, ...]
    cardinality: str = MANY_TO_ONE
    nullable: bool = False
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or f"{self.child}({','.join(self.columns)})->{self.parent}"


@dataclass(frozen=True)
class Hop:
    edge: Edge
    forward: bool  # True: child -> parent (many-to-one); False: parent -> child (one-to-many)

    @property
    def source(self) -> str:
        return self.edge.child if self.forward else self.edge.parent

    @property
    def target(self) -> str:
        return self.edge.parent if self.forward else self.edge.child

    @property
    def fan_out(self) -> bool:
        return not self.forward and self.edge.cardinality != ONE_TO_ONE


@dataclass(frozen=True)
class Path:
    source: str
    hops: tuple[Hop, ...]

    @property
    def target(self) -> str:
        return self.hops[-1].target if self.hops else self.source

    @property
    def fan_out(self) -> bool:
        return any(h.fan_out for h in self.hops)

    @property
    def cardinality(self) -> str:
        return ONE_TO_MANY if self.fan_out else MANY_TO_ONE

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(h.edge.label for h in self.hops)


@dataclass(frozen=True)
class JoinGraph:
    entities: tuple[Entity, ...]
    edges: tuple[Edge, ...]
    _by_name: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False)

    def __post_init__(self):
        by_name = {}
        for entity in self.entities:
            if entity.name in by_name:
                raise RelationError(f"duplicate entity {entity.name!r}")
            by_name[entity.name] = entity
        object.__setattr__(self, "_by_name", by_name)

        labels = set()
        for edge in self.edges:
            if edge.label in labels:
                raise RelationError(f"duplicate edge {edge.label!r}")
            labels.add(edge.label)
            self._check_edge(edge)
        self._check_acyclic()

    def entity(self, name: str) -> Entity:
        try:
            return self._by_name[name]
        except KeyError:
            raise RelationError(f"unknown entity {name!r}") from None

    def _check_edge(self, edge: Edge) -> None:
        child, parent = self.entity(edge.child), self.entity(edge.parent)
        if edge.cardinality == MANY_TO_MANY:
            raise RelationError(
                f"{edge.label}: many-to-many is not joinable; declare a bridge entity "
                "with two many-to-one edges"
            )
        if edge.cardinality not in (MANY_TO_ONE, ONE_TO_ONE):
            raise RelationError(
                f"{edge.label}: an edge is declared child -> parent, so its cardinality is "
                f"many_to_one or one_to_one, not {edge.cardinality!r}"
            )
        if edge.nullable:
            raise RelationError(f"{edge.label}: a NULL key breaks join chains; columns must be NOT NULL")
        if len(edge.columns) != len(parent.primary_key):
            raise RelationError(
                f"{edge.label}: {len(edge.columns)} column(s) cannot reference the "
                f"{len(parent.primary_key)}-column primary key of {parent.name!r} "
                "(a foreign key must target a key)"
            )
        if child.name == parent.name:
            raise RelationError(f"{edge.label}: self reference")

    def _check_acyclic(self) -> None:
        children: dict[str, list[str]] = {e.name: [] for e in self.entities}
        for edge in self.edges:
            children[edge.child].append(edge.parent)
        state: dict[str, int] = {}

        def visit(node: str, trail: tuple[str, ...]) -> None:
            if state.get(node) == 1:
                raise RelationError("cycle: " + " -> ".join((*trail[trail.index(node):], node)))
            if state.get(node) == 2:
                return
            state[node] = 1
            for nxt in children[node]:
                visit(nxt, (*trail, node))
            state[node] = 2

        for name in children:
            visit(name, ())

    def _paths(self, source: str, target: str, max_hops: int) -> list[Path]:
        self.entity(source), self.entity(target)
        found: list[Path] = []

        def walk(node: str, hops: tuple[Hop, ...], seen: frozenset[str]) -> None:
            if node == target and hops:
                found.append(Path(source, hops))
                return
            if len(hops) == max_hops:
                return
            for edge in self.edges:
                for hop in (Hop(edge, True), Hop(edge, False)):
                    if hop.source == node and hop.target not in seen:
                        walk(hop.target, (*hops, hop), seen | {hop.target})

        walk(source, (), frozenset({source}))
        return found

    def path(
        self,
        source: str,
        target: str,
        *,
        via: tuple[str, ...] | None = None,
        allow_fan_out: bool = False,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> Path:
        """The one route from `source` to `target`.

        Raises `RelationError` when there is none within `max_hops`,
        `FanOutError` when only fan-out routes exist and they were not allowed,
        `AmbiguousPathError` when several safe routes remain and `via` (edge
        labels the route must contain) does not single one out."""
        candidates = self._paths(source, target, max_hops)
        if via is not None:
            candidates = [p for p in candidates if set(via) <= set(p.labels)]
        if not candidates:
            raise RelationError(f"no path from {source!r} to {target!r} within {max_hops} hop(s)")
        safe = [p for p in candidates if allow_fan_out or not p.fan_out]
        if not safe:
            raise FanOutError(
                f"{source!r} -> {target!r} multiplies rows (one-to-many via "
                f"{candidates[0].labels}); pass allow_fan_out=True to request it"
            )
        if len(safe) > 1:
            raise AmbiguousPathError(
                f"{source!r} -> {target!r} has {len(safe)} routes: "
                + "; ".join(" / ".join(p.labels) for p in safe)
                + "; choose one with via="
            )
        return safe[0]

    def reachable(self, source: str, *, max_hops: int = DEFAULT_MAX_HOPS) -> dict[str, Path]:
        """Every entity reachable from `source` along many-to-one hops only
        (grain preserved), with its unique route. Entities with several routes
        are omitted from the result and reported by `ambiguous`."""
        out: dict[str, Path] = {}
        for entity in self.entities:
            if entity.name == source:
                continue
            try:
                out[entity.name] = self.path(source, entity.name, max_hops=max_hops)
            except (FanOutError, AmbiguousPathError):
                continue
            except RelationError:
                continue
        return out

    def ambiguous(self, source: str, *, max_hops: int = DEFAULT_MAX_HOPS) -> list[str]:
        result = []
        for entity in self.entities:
            if entity.name == source:
                continue
            try:
                self.path(source, entity.name, max_hops=max_hops)
            except AmbiguousPathError:
                result.append(entity.name)
            except RelationError:
                pass
        return result


def graph_from_spec(spec: dict) -> JoinGraph:
    """Build a graph from a plain dict (parsed `config.yaml`), Frictionless-like:

        entities:
          zone: {primaryKey: ["Code ZE"]}
        references:
          - {child: fact, parent: zone, fields: ["Code ZE"], cardinality: many_to_one}
    """
    entities = tuple(
        Entity(name, tuple(body["primaryKey"])) for name, body in spec.get("entities", {}).items()
    )
    edges = tuple(
        Edge(
            child=ref["child"],
            parent=ref["parent"],
            columns=tuple(ref["fields"]),
            cardinality=ref.get("cardinality", MANY_TO_ONE),
            nullable=ref.get("nullable", False),
            name=ref.get("name"),
        )
        for ref in spec.get("references", [])
    )
    return JoinGraph(entities, edges)
