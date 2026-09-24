import pytest

from sqlsink.relations import (
    AmbiguousPathError,
    Edge,
    Entity,
    FanOutError,
    JoinGraph,
    RelationError,
    graph_from_spec,
)


def chain():
    """fact -> zone -> region -> country, all many-to-one."""
    return JoinGraph(
        (
            Entity("fact", ("id",)),
            Entity("zone", ("ze",)),
            Entity("region", ("reg",)),
            Entity("country", ("cc",)),
            Entity("zone_part", ("ze", "part")),
        ),
        (
            Edge("fact", "zone", ("ze",)),
            Edge("zone", "region", ("reg",)),
            Edge("region", "country", ("cc",)),
            Edge("zone_part", "zone", ("ze",)),
        ),
    )


def test_many_to_one_chain_is_transitive():
    p = chain().path("fact", "region")
    assert [h.target for h in p.hops] == ["zone", "region"]
    assert p.cardinality == "many_to_one" and not p.fan_out


def test_hop_cap_is_enforced_and_configurable():
    g = chain()
    with pytest.raises(RelationError, match="no path"):
        g.path("fact", "country")  # 3 hops, default cap 2
    assert g.path("fact", "country", max_hops=3).target == "country"


def test_reachable_lists_only_grain_preserving_routes():
    reach = chain().reachable("fact", max_hops=3)
    assert set(reach) == {"zone", "region", "country"}
    assert all(not p.fan_out for p in reach.values())


def test_reverse_hop_is_a_fan_out_and_needs_opt_in():
    g = chain()
    with pytest.raises(FanOutError):
        g.path("zone", "zone_part")
    p = g.path("zone", "zone_part", allow_fan_out=True)
    assert p.fan_out and p.cardinality == "one_to_many"


def test_primary_to_foreign_direction_is_refused_even_through_a_chain():
    # fact -> zone is fine; fact -> zone -> zone_part fans out.
    with pytest.raises(FanOutError):
        chain().path("fact", "zone_part")


def test_two_routes_are_ambiguous_until_one_is_chosen():
    g = JoinGraph(
        (Entity("a", ("id",)), Entity("b", ("id",)), Entity("c", ("id",)), Entity("d", ("id",))),
        (
            Edge("a", "b", ("b_id",), name="a_b"),
            Edge("a", "c", ("c_id",), name="a_c"),
            Edge("b", "d", ("d_id",), name="b_d"),
            Edge("c", "d", ("d_id",), name="c_d"),
        ),
    )
    with pytest.raises(AmbiguousPathError):
        g.path("a", "d")
    assert g.ambiguous("a") == ["d"]
    assert g.path("a", "d", via=("a_b",)).labels == ("a_b", "b_d")
    assert "d" not in g.reachable("a")


def test_cycle_is_rejected():
    with pytest.raises(RelationError, match="cycle"):
        JoinGraph(
            (Entity("a", ("id",)), Entity("b", ("id",))),
            (Edge("a", "b", ("b_id",)), Edge("b", "a", ("a_id",))),
        )


def test_many_to_many_is_rejected():
    with pytest.raises(RelationError, match="many-to-many"):
        JoinGraph(
            (Entity("a", ("id",)), Entity("b", ("id",))),
            (Edge("a", "b", ("b_id",), cardinality="many_to_many"),),
        )


def test_nullable_key_is_rejected():
    with pytest.raises(RelationError, match="NOT NULL"):
        JoinGraph(
            (Entity("a", ("id",)), Entity("b", ("id",))),
            (Edge("a", "b", ("b_id",), nullable=True),),
        )


def test_foreign_key_must_target_the_whole_primary_key():
    with pytest.raises(RelationError, match="must target a key"):
        JoinGraph(
            (Entity("a", ("id",)), Entity("part", ("ze", "n"))),
            (Edge("a", "part", ("ze",)),),
        )


def test_unknown_entity_and_duplicates():
    with pytest.raises(RelationError, match="unknown entity"):
        JoinGraph((Entity("a", ("id",)),), (Edge("a", "ghost", ("x",)),))
    with pytest.raises(RelationError, match="duplicate entity"):
        JoinGraph((Entity("a", ("id",)), Entity("a", ("id",))), ())


def test_graph_from_spec_roundtrip():
    g = graph_from_spec(
        {
            "entities": {"zone": {"primaryKey": ["Code ZE"]}, "fact": {"primaryKey": ["id"]}},
            "references": [{"child": "fact", "parent": "zone", "fields": ["Code ZE"]}],
        }
    )
    assert g.path("fact", "zone").target == "zone"
