"""Manifest v2 end to end on toy tables: several components, `view_sql` with a
bridge priority tie-break, contour datasets as fact + contour + `joins`, a
shared component republished under two datasets, materialized views, grants
and fingerprints over views, verification. Runs on DuckDB, and on PostgreSQL
with SNAKEMAKE_SQL_TEST_PG_DSN (where views bind to table objects and
materialized views exist)."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from fixtures.make_fixtures import (
    TRANSITIONS_EXPECTED,
    TRANSITIONS_VIEW_SQL,
    make_multipart_geometry_fixtures,
    make_transitions_fixtures,
    write_table,
)
from sqlsink import metadata as meta_mod
from sqlsink.fingerprint import fingerprint_json
from sqlsink.grants import dataset_relations, manifests_relations
from sqlsink.manifest import load_manifest, load_manifests
from sqlsink.publish import PublishConflict, publish_datasets
from sqlsink.sink import (
    OrphanFactsError,
    materialize_v2,
    normalize_v2,
    publish_v2,
    stage_v2,
)
from sqlsink.sink_postgres import SqlSink
from sqlsink.sink_v2 import ComponentKeyError
from sqlsink.verify import verify_view

TRANSITIONS_COMPONENTS = [
    {"name": "fact", "kind": "fact", "primary_key": ["obs_id"]},
    {"name": "bridge", "kind": "bridge", "primary_key": ["anchor", "f21"], "priority": "priority"},
    {"name": "sector_paths", "kind": "dimension", "primary_key": ["region", "f21", "path"]},
]


def transitions(name="transitions", materialize="view"):
    return load_manifest(
        {"name": name, "materialize": materialize, "components": TRANSITIONS_COMPONENTS, "view_sql": TRANSITIONS_VIEW_SQL}
    )


def rows(engine, sql):
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(sql)).fetchall()]


def view_rows(engine, name="transitions"):
    return sorted(rows(engine, f'SELECT obs_id, region, flow, path, metric FROM "{name}"'), key=lambda r: (r[0], r[3]))


def relation_kinds(engine, name):
    insp = inspect(engine)
    kinds = set()
    if name in insp.get_view_names():
        kinds.add("view")
    if engine.dialect.name == "postgresql" and name in insp.get_materialized_view_names():
        kinds.add("materialized")
    if name in insp.get_table_names() and "view" not in kinds:
        kinds.add("table")
    return kinds


def leftover_old_tables(engine):
    insp = inspect(engine)
    found = []
    for schema in (None, "analytics_storage"):
        if schema and schema not in insp.get_schema_names():
            continue
        found += [t for t in insp.get_table_names(schema=schema) if "__old__" in t]
    return found


@pytest.fixture
def sink(engine):
    return SqlSink(engine)


@pytest.fixture
def toy(tmp_path):
    return make_transitions_fixtures(tmp_path / "toy")


def test_multi_component_view_sql_with_bridge_priority_tie_break(engine, sink, toy):
    manifest = transitions()
    published, result = materialize_v2(manifest, toy, sink)
    assert published == ["transitions"]
    assert [r.status for r in result.components] == ["staged"] * 3
    assert result.dataset.status == "staged"
    # Ordered by priority (F2 first), not by f21 code (F1 first would give 1.0).
    assert view_rows(engine) == TRANSITIONS_EXPECTED
    assert relation_kinds(engine, "transitions") == {"view"}
    # Components are real tables of their own, at their native grain.
    assert rows(engine, 'SELECT count(*) FROM "analytics_storage"."fact"') == [(3,)]
    assert rows(engine, 'SELECT count(*) FROM "analytics_storage"."sector_paths"') == [(4,)]
    assert leftover_old_tables(engine) == []
    # The dataset marker and the component list it was published over.
    marker = meta_mod.analytics_dataset_updates
    with engine.connect() as conn:
        (name, update_id) = conn.execute(text(f"SELECT dataset_name, update_id FROM {marker.name}")).one()
        components = conn.execute(text(f"SELECT component_name FROM {meta_mod.dataset_components.name} ORDER BY 1")).fetchall()
    assert (name, update_id) == ("transitions", result.dataset.update_id)
    assert [c[0] for c in components] == ["bridge", "fact", "sector_paths"]


def test_rerun_is_current_and_publishes_nothing(engine, sink, toy):
    manifest = transitions()
    materialize_v2(manifest, toy, sink)
    published, result = materialize_v2(manifest, toy, sink)
    assert published == []
    assert [r.status for r in result.components] == ["current"] * 3
    assert result.dataset.status == "current"


def test_a_changed_component_restages_only_it_and_its_dataset(engine, sink, toy):
    manifest = transitions()
    materialize_v2(manifest, toy, sink)
    write_table(
        toy["bridge"], "anchor VARCHAR, f21 VARCHAR, priority INTEGER", [("A", "F2", 2), ("A", "F1", 1), ("B", "F1", 1)]
    )
    published, result = materialize_v2(manifest, toy, sink)
    assert published == ["transitions"]
    assert {r.component: r.status for r in result.components} == {"fact": "current", "bridge": "staged", "sector_paths": "current"}
    # Priorities flipped: now F1 (metric 1.0) wins for observation 1 / path P.
    assert (1, "R1", 10.5, "P", 1.0) in view_rows(engine)
    assert leftover_old_tables(engine) == []


def test_bridge_priority_column_and_component_keys_are_checked_at_staging(engine, sink, toy):
    no_priority = write_table(toy["bridge"].parent / "b2.parquet", "anchor VARCHAR, f21 VARCHAR", [("A", "F1")])
    with pytest.raises(Exception, match="priority"):
        materialize_v2(transitions(), {**toy, "bridge": no_priority}, sink)
    duplicated = write_table(
        toy["fact"].parent / "f2.parquet",
        "obs_id BIGINT, region VARCHAR, anchor VARCHAR, flow DOUBLE",
        [(1, "R1", "A", 1.0), (1, "R1", "B", 2.0)],
    )
    with pytest.raises(ComponentKeyError, match="duplicated primary key"):
        materialize_v2(transitions(), {**toy, "fact": duplicated}, sink)
    # Nothing was published by either failure.
    assert "transitions" not in inspect(engine).get_view_names()


def test_contour_dataset_as_fact_plus_contour_with_joins_matches_the_v1_form(tmp_path, engine, sink):
    from sqlsink.materialize import stage_contours, stage_dataset
    from test_materialize_dataset import MANIFEST

    v1 = make_multipart_geometry_fixtures(tmp_path / "v1")
    publish_datasets(
        engine,
        [MANIFEST],
        [stage_dataset(engine, MANIFEST, str(v1["facts"]), str(v1["contours"])).to_dict()],
        [stage_contours(engine, MANIFEST, str(v1["contours"])).to_dict()],
    )
    zone_facts = write_table(tmp_path / "v2" / "zone_facts.parquet", "zone_id VARCHAR, metric INTEGER", [("Z1", 10), ("Z2", 20)])
    zone_parts = write_table(
        tmp_path / "v2" / "zone_parts.parquet",
        "zone_id VARCHAR, part_no INTEGER, polygon_coords VARCHAR",
        [("Z1", 1, "part-Z1-a"), ("Z1", 2, "part-Z1-b"), ("Z2", 1, "part-Z2-a")],
    )
    manifest = load_manifest(
        {
            "name": "zones_v2",
            "components": [
                {"name": "zone_facts", "kind": "fact", "primary_key": ["zone_id"]},
                {"name": "zone_parts", "kind": "dimension", "primary_key": ["zone_id", "part_no"]},
            ],
            "view": {
                "base": "zone_facts",
                "joins": [{"component": "zone_parts", "on": {"zone_id": "zone_id"}, "fan_out": True}],
                "select": ["zone_id", "metric", {"column": "polygon_coords", "from": "zone_parts"}],
            },
        }
    )
    materialize_v2(manifest, {"zone_facts": zone_facts, "zone_parts": zone_parts}, sink)
    v1_rows = rows(engine, 'SELECT zone_id, metric, polygon_coords FROM "zones"')
    v2_rows = rows(engine, 'SELECT zone_id, metric, polygon_coords FROM "zones_v2"')
    assert sorted(v2_rows) == sorted(v1_rows) == sorted(
        [("Z1", 10, "part-Z1-a"), ("Z1", 10, "part-Z1-b"), ("Z2", 20, "part-Z2-a")]
    )


def test_v1_and_v2_datasets_publish_in_one_transaction(tmp_path, engine, sink, toy):
    from sqlsink.materialize import stage_contours, stage_dataset
    from test_materialize_dataset import MANIFEST

    v1 = make_multipart_geometry_fixtures(tmp_path / "v1")
    manifest = transitions()
    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    d1 = stage_dataset(engine, MANIFEST, str(v1["facts"]), str(v1["contours"])).to_dict()
    c1 = stage_contours(engine, MANIFEST, str(v1["contours"])).to_dict()
    published = publish_datasets(
        engine,
        [MANIFEST, manifest],
        [d1, result.dataset.to_dict()],
        [c1],
        component_receipts=[r.to_dict() for r in result.components],
    )
    assert sorted(published) == ["transitions", "zones"]
    assert sink.published_intact(manifest) and sink.published_intact(MANIFEST)


# -- shared components ------------------------------------------------------

SHARED = {
    "components": [{"name": "dim", "kind": "dimension", "primary_key": ["k"]}],
    "datasets": [
        {"name": "ds_a", "components": ["dim", {"name": "facts_a", "kind": "fact"}],
         "view_sql": "SELECT a.id, a.k, d.label FROM {facts_a} a JOIN {dim} d ON d.k = a.k"},
        {"name": "ds_b", "components": ["dim", {"name": "facts_b", "kind": "fact"}],
         "view_sql": "SELECT b.id, b.k, d.label, b.w FROM {facts_b} b JOIN {dim} d ON d.k = b.k"},
    ],
}


@pytest.fixture
def shared_sources(tmp_path):
    d = tmp_path / "shared"
    return {
        "dim": write_table(d / "dim.parquet", "k VARCHAR, label VARCHAR", [("x", "old-x"), ("y", "old-y")]),
        "facts_a": write_table(d / "facts_a.parquet", "id INTEGER, k VARCHAR", [(1, "x"), (2, "y")]),
        "facts_b": write_table(d / "facts_b.parquet", "id INTEGER, k VARCHAR, w INTEGER", [(7, "y", 70)]),
    }


def stage_all(sink, manifests, sources):
    return [stage_v2(normalize_v2(m, sources, sink=sink), sink) for m in manifests]


def test_shared_component_republish_recreates_every_dependent_view(engine, sink, shared_sources):
    a, b = load_manifests(SHARED)
    results = stage_all(sink, [a, b], shared_sources)
    # The shared dimension is staged by the first dataset, found staged by the second.
    assert [r.status for r in results[0].components if r.component == "dim"] == ["staged"]
    assert publish_v2(sink, [a, b], results) == ["ds_a", "ds_b"]
    assert rows(engine, 'SELECT label FROM ds_a ORDER BY id') == [("old-x",), ("old-y",)]

    # The dimension changes; dataset B's own facts do not.
    write_table(shared_sources["dim"], "k VARCHAR, label VARCHAR", [("x", "new-x"), ("y", "new-y")])
    results = stage_all(sink, [a, b], shared_sources)
    assert {r.component: r.status for r in results[1].components} == {"dim": "staged", "facts_b": "current"}
    assert publish_v2(sink, [a, b], results) == ["ds_a", "ds_b"]
    # On PostgreSQL a view keeps reading the renamed table unless recreated.
    assert rows(engine, 'SELECT label FROM ds_a ORDER BY id') == [("new-x",), ("new-y",)]
    assert rows(engine, 'SELECT label, w FROM ds_b') == [("new-y", 70)]
    assert leftover_old_tables(engine) == []
    # Third run: everything current.
    assert publish_v2(sink, [a, b], stage_all(sink, [a, b], shared_sources)) == []


def test_swapping_a_shared_component_without_restaging_its_dependents_conflicts(engine, sink, shared_sources):
    a, b = load_manifests(SHARED)
    publish_v2(sink, [a, b], stage_all(sink, [a, b], shared_sources))
    write_table(shared_sources["dim"], "k VARCHAR, label VARCHAR", [("x", "new-x"), ("y", "new-y")])
    only_a = stage_v2(normalize_v2(a, shared_sources, sink=sink), sink)
    with pytest.raises(PublishConflict, match="ds_b"):
        # `b` is published in the database but absent from this publish.
        publish_datasets(engine, [a], [only_a.dataset.to_dict()], [], component_receipts=[r.to_dict() for r in only_a.components])
    # All-or-nothing: nothing changed.
    assert rows(engine, 'SELECT label FROM ds_a ORDER BY id') == [("old-x",), ("old-y",)]
    assert rows(engine, 'SELECT label FROM ds_b') == [("old-y",)]


def test_component_republished_elsewhere_after_staging_is_a_conflict(engine, sink, shared_sources):
    a, b = load_manifests(SHARED)
    first = stage_all(sink, [a, b], shared_sources)
    publish_v2(sink, [a, b], first)
    write_table(shared_sources["dim"], "k VARCHAR, label VARCHAR", [("x", "n1"), ("y", "n1")])
    stale = stage_all(sink, [a, b], shared_sources)
    write_table(shared_sources["dim"], "k VARCHAR, label VARCHAR", [("x", "n2"), ("y", "n2")])
    fresh = stage_all(sink, [a, b], shared_sources)
    publish_v2(sink, [a, b], fresh)
    with pytest.raises(PublishConflict, match="republished by another process"):
        publish_v2(sink, [a, b], stale)


def test_inner_join_orphans_are_rejected_before_anything_is_written(engine, sink, tmp_path):
    d = tmp_path / "orph"
    sources = {
        "f": write_table(d / "f.parquet", "id INTEGER, region_code VARCHAR", [(1, "01"), (2, "99")]),
        "region": write_table(d / "region.parquet", "code VARCHAR, label VARCHAR", [("01", "Alpha")]),
    }
    manifest = load_manifest(
        {
            "name": "with_region",
            "components": [{"name": "f", "kind": "fact"}, {"name": "region", "kind": "dimension", "primary_key": ["code"]}],
            "view": {"base": "f", "joins": [{"component": "region", "on": {"region_code": "code"}}]},
        }
    )
    with pytest.raises(OrphanFactsError, match="region"):
        materialize_v2(manifest, sources, sink)
    assert "with_region" not in inspect(engine).get_view_names()
    left = load_manifest({**{"name": "with_region", "components": manifest.canonical_dict()["components"]},
                          "view": {"base": "f", "joins": [{"component": "region", "on": {"region_code": "code"}, "type": "left"}]}})
    assert materialize_v2(left, sources, sink)[0] == ["with_region"]
    assert sorted(rows(engine, 'SELECT id, label FROM with_region'), key=lambda r: r[0]) == [(1, "Alpha"), (2, None)]


def test_a_component_can_read_a_table_of_the_sink(engine, sink, toy, tmp_path):
    from sqlsink.publish import publish_tables
    from sqlsink.stage import stage_table

    publish_tables(engine, [stage_table(engine, "fake_paths", str(toy["sector_paths"])).to_dict()])
    components = [
        c if c["name"] != "sector_paths" else {**c, "source": {"table": "fake_paths"}} for c in TRANSITIONS_COMPONENTS
    ]
    manifest = load_manifest({"name": "transitions", "components": components, "view_sql": TRANSITIONS_VIEW_SQL})
    published, result = materialize_v2(manifest, toy, sink)
    assert published == ["transitions"]
    assert view_rows(engine) == TRANSITIONS_EXPECTED


# -- materialized views ------------------------------------------------------


def test_materialize_switches_between_view_and_materialized_both_ways(engine, sink, toy):
    expected_materialized = {"materialized"} if engine.dialect.name == "postgresql" else {"table"}
    materialize_v2(transitions(materialize="view"), toy, sink)
    assert relation_kinds(engine, "transitions") == {"view"}

    m = transitions(materialize="materialized")
    assert materialize_v2(m, toy, sink)[0] == ["transitions"]
    assert relation_kinds(engine, "transitions") == expected_materialized
    assert view_rows(engine) == TRANSITIONS_EXPECTED
    assert sink.published_intact(m)
    assert not sink.published_intact(transitions(materialize="view"))

    assert materialize_v2(transitions(materialize="view"), toy, sink)[0] == ["transitions"]
    assert relation_kinds(engine, "transitions") == {"view"}
    assert view_rows(engine) == TRANSITIONS_EXPECTED
    assert leftover_old_tables(engine) == []


def test_materialized_relation_is_rebuilt_from_new_components_in_the_publish(engine, sink, toy):
    m = transitions(materialize="materialized")
    materialize_v2(m, toy, sink)
    write_table(
        toy["sector_paths"],
        "region VARCHAR, f21 VARCHAR, path VARCHAR, metric DOUBLE",
        [("R1", "F1", "P", 1.0), ("R1", "F2", "P", 9.0), ("R1", "F1", "Q", 3.0), ("R2", "F1", "P", 4.0)],
    )
    assert materialize_v2(m, toy, sink)[0] == ["transitions"]
    assert (1, "R1", 10.5, "P", 9.0) in view_rows(engine)
    assert leftover_old_tables(engine) == []


def test_a_pre_existing_flat_table_is_replaced_by_the_view(engine, sink, toy):
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE "transitions" (x INTEGER)'))
    materialize_v2(transitions(), toy, sink)
    assert relation_kinds(engine, "transitions") == {"view"}
    assert leftover_old_tables(engine) == []


def test_published_intact_notices_a_dropped_component_table(engine, sink, toy):
    m = transitions()
    materialize_v2(m, toy, sink)
    assert sink.published_intact(m)
    with engine.begin() as conn:
        conn.execute(text('DROP VIEW "transitions"'))
        conn.execute(text('DROP TABLE "analytics_storage"."bridge"'))
    assert not sink.published_intact(m)
    # The marker survived, the table did not: restaging rebuilds it.
    published, result = materialize_v2(m, toy, sink)
    assert published == ["transitions"] and view_rows(engine) == TRANSITIONS_EXPECTED


# -- grants and fingerprints ---------------------------------------------------


def test_grant_relations_cover_views_and_components(toy):
    m = transitions()
    assert dataset_relations(m) == [
        "analytics_storage.bridge",
        "analytics_storage.fact",
        "analytics_storage.sector_paths",
        "public.transitions",
    ]
    a, b = load_manifests(SHARED)
    assert manifests_relations([a, b]) == [
        "analytics_storage.dim", "analytics_storage.facts_a", "analytics_storage.facts_b", "public.ds_a", "public.ds_b",
    ]


class _RecordingConn:
    def __init__(self, log):
        self.log = log

    def execute(self, statement, *args):
        self.log.append(str(statement))


class _FakePostgres:
    """Just enough engine to see the GRANT statements without a server."""

    class dialect:  # noqa: N801
        name = "postgresql"

    def __init__(self):
        self.log = []

    def begin(self):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            yield _RecordingConn(self.log)

        return ctx()


def test_grants_are_issued_on_the_view_and_on_every_component():
    from sqlsink.grants import grant_runtime

    engine = _FakePostgres()
    grant_runtime(engine, "reader", dataset_relations(transitions()))
    assert 'GRANT USAGE ON SCHEMA "analytics_storage" TO "reader"' in engine.log
    assert 'GRANT SELECT ON "public"."transitions" TO "reader"' in engine.log
    assert 'GRANT SELECT ON "analytics_storage"."bridge" TO "reader"' in engine.log


@pytest.mark.parametrize("materialize", ["view", "materialized"])
def test_role_select_on_views_and_materialized_views(engine, sink, toy, materialize):
    if engine.dialect.name != "postgresql":
        pytest.skip("role grants need PostgreSQL")
    from sqlsink.grants import grant_runtime, role_has_select

    role = "sqlsink_test_reader"
    with engine.begin() as conn:
        conn.execute(text(f"DROP ROLE IF EXISTS {role}"))
        conn.execute(text(f"CREATE ROLE {role}"))
    try:
        m = transitions(materialize=materialize)
        materialize_v2(m, toy, sink)
        relations = dataset_relations(m)
        assert role_has_select(engine, role, relations) is False
        grant_runtime(engine, role, relations)
        assert role_has_select(engine, role, relations) is True
        # A republish replaces the relation: the grant is gone and is noticed.
        write_table(toy["fact"], "obs_id BIGINT, region VARCHAR, anchor VARCHAR, flow DOUBLE", [(1, "R1", "A", 1.0)])
        materialize_v2(m, toy, sink)
        assert role_has_select(engine, role, relations) is False
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS analytics_storage CASCADE"))
            conn.execute(text(f"DROP OWNED BY {role}"))
            conn.execute(text(f"DROP ROLE {role}"))


def _view_schema(engine):
    return "public" if engine.dialect.name == "postgresql" else "main"


def test_fingerprint_covers_components_and_the_view(engine, sink, toy):
    m = transitions()
    view = f"{_view_schema(engine)}.transitions"
    relations = dataset_relations(m, view_schema=_view_schema(engine))
    materialize_v2(m, toy, sink)
    before = fingerprint_json(engine, relations)
    assert before == fingerprint_json(engine, relations)  # byte stable
    import json

    parsed = json.loads(before)
    assert parsed["analytics_storage.fact"]["marker"]["kind"] == "component"
    assert parsed[view]["marker"]["kind"] == "dataset"
    assert all(entry["exists"] for entry in parsed.values())
    write_table(toy["bridge"], "anchor VARCHAR, f21 VARCHAR, priority INTEGER", [("A", "F2", 5), ("A", "F1", 6), ("B", "F1", 1)])
    materialize_v2(m, toy, sink)
    after = json.loads(fingerprint_json(engine, relations))
    # Only the changed component's marker moves; the untouched one keeps its token.
    assert after["analytics_storage.bridge"]["marker"] != parsed["analytics_storage.bridge"]["marker"]
    strip = lambda e: {k: v for k, v in e["marker"].items() if k != "published_at"}  # noqa: E731
    assert strip(after["analytics_storage.fact"]) == strip(parsed["analytics_storage.fact"])
    assert after[view]["marker"]["update_id"] != parsed[view]["marker"]["update_id"]


def test_fingerprint_of_a_materialized_view_sees_it(engine, sink, toy):
    m = transitions(materialize="materialized")
    materialize_v2(m, toy, sink)
    import json

    name = f"{_view_schema(engine)}.transitions"
    entry = json.loads(fingerprint_json(engine, [name]))[name]
    assert entry["exists"] is True and entry["row_count"] == len(TRANSITIONS_EXPECTED)


# -- verification -----------------------------------------------------------------


def test_verify_view_against_a_reference_parquet(engine, sink, toy, tmp_path):
    m = transitions()
    materialize_v2(m, toy, sink)
    reference = write_table(
        tmp_path / "ref.parquet",
        "obs_id BIGINT, region VARCHAR, flow DOUBLE, path VARCHAR, metric DOUBLE",
        [(o, r, f, p, mt) for o, r, f, p, mt in TRANSITIONS_EXPECTED],
    )
    result = verify_view(sink, m, str(reference), buckets=3)
    assert result.equivalent, result
    assert result.view_rows == len(TRANSITIONS_EXPECTED) and result.float_columns == ["flow", "metric"]

    # One metric off by 5e-10 relative is inside the tolerance; 2e-9 is not.
    def with_metric(delta):
        rows_ = [(o, r, f, p, (mt * (1 + delta) if (o, p) == (1, "P") else mt)) for o, r, f, p, mt in TRANSITIONS_EXPECTED]
        return write_table(
            tmp_path / f"ref{delta}.parquet", "obs_id BIGINT, region VARCHAR, flow DOUBLE, path VARCHAR, metric DOUBLE", rows_
        )

    assert verify_view(sink, m, str(with_metric(5e-10))).equivalent
    off = verify_view(sink, m, str(with_metric(2e-9)))
    assert not off.equivalent and off.mismatched_groups == 1 and off.reference_rows == off.view_rows


def test_verify_view_reports_row_column_and_value_differences(engine, sink, toy, tmp_path):
    m = transitions()
    materialize_v2(m, toy, sink)
    ddl = "obs_id BIGINT, region VARCHAR, flow DOUBLE, path VARCHAR, metric DOUBLE"
    fewer = write_table(tmp_path / "fewer.parquet", ddl, [tuple(r) for r in TRANSITIONS_EXPECTED[1:]])
    result = verify_view(sink, m, str(fewer))
    assert not result.equivalent and result.reference_rows + 1 == result.view_rows

    changed_text = write_table(
        tmp_path / "text.parquet", ddl, [(o, "RX" if o == 3 and p == "P" else r, f, p, mt) for o, r, f, p, mt in TRANSITIONS_EXPECTED]
    )
    assert not verify_view(sink, m, str(changed_text)).equivalent

    other_columns = write_table(tmp_path / "cols.parquet", "obs_id BIGINT, extra VARCHAR", [(1, "x")])
    result = verify_view(sink, m, str(other_columns))
    assert result.missing_columns and result.extra_columns and not result.equivalent


def test_compare_relations_treats_null_as_equal_and_multiplicity_as_significant():
    import duckdb

    from sqlsink.verify import compare_relations

    con = duckdb.connect()
    con.execute("CREATE TABLE a AS SELECT * FROM (VALUES (1, 'x', NULL::DOUBLE), (1, 'x', NULL), (2, NULL, 1.5)) t(i, s, v)")
    con.execute("CREATE TABLE b AS SELECT * FROM (VALUES (2, NULL, 1.5), (1, 'x', NULL), (1, 'x', NULL)) t(i, s, v)")
    assert compare_relations(con, "a", "b").equivalent
    con.execute("CREATE TABLE c AS SELECT * FROM (VALUES (2, NULL, 1.5), (1, 'x', NULL), (2, NULL, 1.5)) t(i, s, v)")
    assert not compare_relations(con, "a", "c").equivalent
    # Same distinct rows, different multiplicity: a plain DISTINCT comparison would miss it.
    con.execute("CREATE TABLE d AS SELECT * FROM (VALUES (2, NULL, 1.5), (1, 'x', NULL)) t(i, s, v)")
    assert not compare_relations(con, "a", "d").equivalent


# -- receipts composed badly ---------------------------------------------------


def test_a_staged_view_cannot_publish_over_a_component_that_was_not_swapped(engine, sink, toy):
    m = transitions()
    materialize_v2(m, toy, sink)
    write_table(toy["bridge"], "anchor VARCHAR, f21 VARCHAR, priority INTEGER", [("A", "F1", 1), ("A", "F2", 2), ("B", "F1", 1)])
    result = stage_v2(normalize_v2(m, toy, sink=sink), sink)
    assert result.dataset.status == "staged"
    before = view_rows(engine)
    # The dataset receipt alone (component receipts lost): the view would be
    # recreated over the old bridge while its marker claimed the new one.
    with pytest.raises(PublishConflict, match="component receipt"):
        publish_datasets(engine, [m], [result.dataset.to_dict()], [], component_receipts=[])
    # A stale "current" receipt for the changed component is no better.
    stale = [dict(r.to_dict(), status="current") for r in result.components]
    with pytest.raises(PublishConflict):
        publish_datasets(engine, [m], [result.dataset.to_dict()], [], component_receipts=stale)
    assert view_rows(engine) == before


def test_merging_component_receipts_prefers_staged_over_current(engine, sink, shared_sources):
    from sqlsink.publish import merge_component_receipts

    a, b = load_manifests(SHARED)
    publish_v2(sink, [a, b], stage_all(sink, [a, b], shared_sources))
    write_table(shared_sources["dim"], "k VARCHAR, label VARCHAR", [("x", "n"), ("y", "n")])
    ra = stage_v2(normalize_v2(a, shared_sources, sink=sink), sink)
    rb = stage_v2(normalize_v2(b, shared_sources, sink=sink), sink)
    dim_a = next(r.to_dict() for r in ra.components if r.component == "dim")
    dim_b = next(r.to_dict() for r in rb.components if r.component == "dim")
    assert merge_component_receipts([dict(dim_a, status="current"), dim_b])["dim"]["status"] == "staged"
    assert merge_component_receipts([dim_a, dict(dim_b, status="current")])["dim"]["status"] == "staged"
    with pytest.raises(PublishConflict, match="two different versions"):
        merge_component_receipts([dim_a, dict(dim_b, update_id="other")])


def test_verify_view_batched_by_key_reaches_the_same_verdict(engine, sink, toy, tmp_path):
    m = transitions()
    materialize_v2(m, toy, sink)
    ddl = "obs_id BIGINT, region VARCHAR, flow DOUBLE, path VARCHAR, metric DOUBLE"
    good = write_table(tmp_path / "good.parquet", ddl, [tuple(r) for r in TRANSITIONS_EXPECTED])
    result = verify_view(sink, m, str(good), key_column="obs_id", key_batch_size=2)
    assert result.equivalent and result.buckets == 3  # keys {1,2} {3} + the NULL-key batch
    bad = write_table(
        tmp_path / "bad.parquet", ddl, [(o, r, f, p, (mt + 1 if (o, p) == (3, "P") else mt)) for o, r, f, p, mt in TRANSITIONS_EXPECTED]
    )
    off = verify_view(sink, m, str(bad), key_column="obs_id", key_batch_size=2)
    assert not off.equivalent and off.mismatched_groups == 1
