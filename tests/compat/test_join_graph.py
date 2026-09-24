"""MetricFlow / Cube join rules, as an auditable cardinality table."""

from __future__ import annotations

import pytest

from compat_support import capability, has_feature
from sql_incremental.relations import (
    AmbiguousPathError,
    Edge,
    Entity,
    FanOutError,
    JoinGraph,
    ONE_TO_MANY,
    ONE_TO_ONE,
    MANY_TO_MANY,
    MANY_TO_ONE,
    RelationError,
)

E = lambda n, *pk: Entity(n, tuple(pk or ("id",)))  # noqa: E731
CHAIN = JoinGraph(
    (E("fact"), E("dim"), E("region"), E("country")),
    (
        Edge("fact", "dim", ("id",)),
        Edge("dim", "region", ("id",)),
        Edge("region", "country", ("id",)),
    ),
)


@capability("metricflow", "many_to_one_traversal")
def test_a_fact_reaches_its_parent_with_its_grain_preserved():
    assert CHAIN.path("fact", "dim").cardinality == MANY_TO_ONE


@capability("metricflow", "transitive_join")
def test_a_chain_of_many_to_one_hops_is_many_to_one():
    path = CHAIN.path("fact", "region")
    assert path.cardinality == MANY_TO_ONE and len(path.hops) == 2


@capability("metricflow", "hop_cap", note="MetricFlow limits multi-hop joins to 2")
def test_three_hops_are_refused_unless_the_cap_is_raised():
    with pytest.raises(RelationError, match="within 2 hop"):
        CHAIN.path("fact", "country")
    assert CHAIN.path("fact", "country", max_hops=3).cardinality == MANY_TO_ONE


@capability("metricflow", "fan_out_rejected", note="primary -> foreign")
def test_going_from_parent_to_child_is_refused_unless_requested():
    with pytest.raises(FanOutError):
        CHAIN.path("dim", "fact")
    assert CHAIN.path("dim", "fact", allow_fan_out=True).cardinality == ONE_TO_MANY


@capability("cube", "one_to_one_is_safe")
def test_a_one_to_one_edge_never_fans_out_in_either_direction():
    graph = JoinGraph((E("a"), E("b")), (Edge("a", "b", ("id",), cardinality=ONE_TO_ONE),))
    assert not graph.path("a", "b").fan_out and not graph.path("b", "a").fan_out


@capability("cube", "many_to_many_needs_bridge")
def test_a_many_to_many_edge_is_not_joinable():
    with pytest.raises(RelationError, match="bridge"):
        JoinGraph((E("a"), E("b")), (Edge("a", "b", ("id",), cardinality=MANY_TO_MANY),))


@capability("cube", "primary_key_required", note="a foreign key must target a key")
def test_an_edge_must_reference_the_whole_primary_key():
    with pytest.raises(RelationError, match="must target a key"):
        JoinGraph((E("a"), E("b", "x", "y")), (Edge("a", "b", ("id",)),))


@capability("metricflow", "null_keys_rejected")
def test_a_nullable_key_is_refused_because_it_breaks_join_chains():
    with pytest.raises(RelationError, match="NOT NULL"):
        JoinGraph((E("a"), E("b")), (Edge("a", "b", ("id",), nullable=True),))


@capability("metricflow", "cycles_rejected")
def test_a_cycle_is_refused():
    with pytest.raises(RelationError, match="cycle"):
        JoinGraph(
            (E("a"), E("b")),
            (Edge("a", "b", ("id",), name="ab"), Edge("b", "a", ("id",), name="ba")),
        )


@capability("metricflow", "ambiguous_path", note="two routes: refuse, or the caller picks with via=")
def test_two_routes_are_ambiguous_until_the_caller_chooses():
    graph = JoinGraph(
        (E("fact"), E("dim")),
        (Edge("fact", "dim", ("id",), name="billing"), Edge("fact", "dim", ("id",), name="shipping")),
    )
    with pytest.raises(AmbiguousPathError):
        graph.path("fact", "dim")
    assert graph.path("fact", "dim", via=("billing",)).labels == ("billing",)


@capability(
    "cube",
    "fan_and_chasm_trap_detection",
    "gap",
    note="close: a multi-target check refusing two fan-outs from one source (small); adopt: Cube needs its own server",
)
def test_a_query_combining_two_fan_outs_is_refused():
    assert has_feature(JoinGraph, "check_fan_traps")
