"""Manifest v2: components, joins (JoinGraph-validated, fan-out declared),
`view_sql` templates. Pure: no database."""

from __future__ import annotations

import pytest

from sqlsink.manifest import DatasetMaterialization, DatasetV2, ManifestError, load_manifest, load_manifests
from sqlsink.relations import FanOutError
from sqlsink.view import physical_refs, render_template, render_view_sql

CONTOUR_SPEC = {
    "name": "zones",
    "components": [
        {"name": "zone_facts", "kind": "fact", "primary_key": ["zone_id"]},
        {"name": "zone_contours", "kind": "dimension", "primary_key": ["zone_id", "part_no"]},
    ],
    "view": {
        "base": "zone_facts",
        "joins": [{"component": "zone_contours", "on": {"zone_id": "zone_id"}, "fan_out": True}],
        "select": ["zone_id", "metric", {"column": "polygon_coords", "from": "zone_contours"}],
    },
}


def test_v1_specs_still_load_as_v1():
    spec = {
        "name": "d", "fact_source": "f", "contour_table": "c", "geometry_column": "g",
        "fact_join_columns": ["k"], "contour_join_columns": ["k"], "output_columns": ["k", "g"],
    }
    manifest = load_manifest(spec)
    assert isinstance(manifest, DatasetMaterialization)
    # Hash unchanged by the v2 work: the four production datasets must not republish.
    assert manifest.canonical_dict()["manifest_version"] == 1


def test_contour_dataset_is_a_fact_plus_contour_component_with_declared_fan_out():
    manifest = load_manifest(CONTOUR_SPEC)
    assert isinstance(manifest, DatasetV2)
    assert manifest.component_names() == ("zone_facts", "zone_contours")
    refs = {"zone_facts": "F", "zone_contours": "C"}
    # Bare names in `select` are resolved against the staged columns; a `from` needs none.
    columns = {"zone_facts": ["zone_id", "metric"], "zone_contours": ["zone_id", "part_no", "polygon_coords"]}
    sql = render_view_sql(manifest, "postgresql", refs, columns)
    with pytest.raises(ManifestError, match="column lists"):
        render_view_sql(manifest, "postgresql", refs)
    assert sql == (
        'SELECT f."zone_id" AS "zone_id", f."metric" AS "metric", j1."polygon_coords" AS "polygon_coords" '
        'FROM F f JOIN C j1 ON f."zone_id" = j1."zone_id"'
    )


def test_undeclared_fan_out_is_rejected_at_load():
    spec = {**CONTOUR_SPEC, "view": {**CONTOUR_SPEC["view"], "joins": [{"component": "zone_contours", "on": {"zone_id": "zone_id"}}]}}
    with pytest.raises(FanOutError, match="fan_out"):
        load_manifest(spec)


def test_many_to_one_join_on_the_full_key_needs_no_declaration():
    spec = {
        "name": "d",
        "components": [
            {"name": "f", "kind": "fact"},
            {"name": "region", "kind": "dimension", "primary_key": ["code"]},
        ],
        "view": {"base": "f", "joins": [{"component": "region", "on": {"region_code": "code"}, "type": "left"}]},
    }
    manifest = load_manifest(spec)
    sql = render_view_sql(manifest, "duckdb", {"f": "F", "region": "R"}, {"f": ["id", "region_code"], "region": ["code", "label"]})
    assert 'LEFT JOIN R j1 ON f."region_code" = j1."code"' in sql
    # No `select`: every base column, then the joined component's own columns.
    assert sql.startswith('SELECT f."id" AS "id", f."region_code" AS "region_code", j1."code" AS "code", j1."label" AS "label"')


def test_chained_joins_are_validated_through_the_join_graph():
    spec = {
        "name": "d",
        "components": [
            {"name": "f", "kind": "fact"},
            {"name": "zone", "kind": "dimension", "primary_key": ["z"]},
            {"name": "region", "kind": "dimension", "primary_key": ["r"]},
        ],
        "view": {
            "base": "f",
            "joins": [
                {"component": "zone", "on": {"z": "z"}},
                {"component": "region", "left": "zone", "on": {"r": "r"}},
            ],
        },
    }
    assert isinstance(load_manifest(spec), DatasetV2)
    spec["view"]["joins"][1]["left"] = "nowhere"
    with pytest.raises(ManifestError, match="not yet joined|unknown component"):
        load_manifest(spec)


def test_ambiguous_bare_column_must_name_its_component():
    spec = {
        "name": "d",
        "components": [{"name": "f", "kind": "fact"}, {"name": "a", "kind": "dimension", "primary_key": ["k"]},
                       {"name": "b", "kind": "dimension", "primary_key": ["k"]}],
        "view": {"base": "f", "joins": [{"component": "a", "on": {"x": "k"}}, {"component": "b", "on": {"y": "k"}}],
                 "select": ["label"]},
    }
    manifest = load_manifest(spec)
    with pytest.raises(ManifestError, match="name one with `from`"):
        render_view_sql(manifest, "duckdb", {"f": "F", "a": "A", "b": "B"}, {"f": ["x", "y"], "a": ["k", "label"], "b": ["k", "label"]})


def test_view_sql_template_substitutes_only_declared_components():
    spec = {
        "name": "d",
        "components": [{"name": "fact", "kind": "fact"}, {"name": "bridge", "kind": "bridge", "priority": "prio"}],
        "view_sql": {"duckdb": "SELECT {fact}.* FROM {fact} JOIN {bridge} USING (k) QUALIFY 1=1", "postgresql": "SELECT 1 FROM {fact}"},
    }
    manifest = load_manifest(spec)
    refs = physical_refs(manifest, "postgresql")
    assert refs["fact"] == '"analytics_storage"."fact"'
    assert render_view_sql(manifest, "postgresql", refs) == 'SELECT 1 FROM "analytics_storage"."fact"'
    assert "QUALIFY" in render_view_sql(manifest, "duckdb", refs)
    # No template for the dialect and no default: an error, not a guess.
    with pytest.raises(ManifestError, match="no view_sql"):
        render_view_sql(manifest, "mysql", refs)


def test_template_leaves_other_braces_alone_and_rejects_unknown_names():
    assert render_template("SELECT '{\"k\": 1}' FROM {t}", {"t": "T"}) == "SELECT '{\"k\": 1}' FROM T"
    with pytest.raises(ManifestError, match="unknown component"):
        render_template("SELECT * FROM {typo}", {"t": "T"})
    with pytest.raises(ManifestError, match="not one of its components"):
        load_manifest({"name": "d", "components": [{"name": "t", "kind": "fact"}], "view_sql": "SELECT * FROM {typo}"})


def test_exactly_one_of_joins_or_view_sql_and_valid_kinds():
    base = {"name": "d", "components": [{"name": "t", "kind": "fact"}]}
    with pytest.raises(ManifestError, match="exactly one"):
        load_manifest(base)
    with pytest.raises(ManifestError, match="kind must be"):
        load_manifest({"name": "d", "components": [{"name": "t", "kind": "weird"}], "view_sql": "SELECT 1"})
    with pytest.raises(ManifestError, match="needs a primary_key"):
        load_manifest({"name": "d", "components": [{"name": "t", "kind": "dimension"}], "view_sql": "SELECT 1"})
    with pytest.raises(ManifestError, match="only applies to a bridge"):
        load_manifest({"name": "d", "components": [{"name": "t", "kind": "fact", "priority": "p"}], "view_sql": "SELECT 1"})


def test_shared_components_are_declared_once_and_referenced_by_name():
    data = {
        "components": [{"name": "dim", "kind": "dimension", "primary_key": ["k"]}],
        "datasets": [
            {"name": "a", "components": ["dim", {"name": "fa", "kind": "fact"}], "view_sql": "SELECT * FROM {fa} JOIN {dim} USING (k)"},
            {"name": "b", "components": ["dim", {"name": "fb", "kind": "fact"}], "view_sql": "SELECT * FROM {fb} JOIN {dim} USING (k)"},
        ],
    }
    a, b = load_manifests(data)
    assert a.component("dim") is b.component("dim") and a.component("dim").shared
    assert not a.component("fa").shared
    with pytest.raises(ManifestError, match="not declared at the top level"):
        load_manifests({"datasets": [{"name": "c", "components": ["dim"], "view_sql": "SELECT 1"}]})
    with pytest.raises(ManifestError, match="already shared"):
        load_manifests({**data, "datasets": [{"name": "c", "components": [{"name": "dim", "kind": "fact"}], "view_sql": "SELECT 1"}]})


def test_conflicting_owned_definitions_and_v1_collisions_are_rejected():
    same = {"name": "shared_fact", "kind": "fact"}
    ok = [
        {"name": "a", "components": [same], "view_sql": "SELECT * FROM {shared_fact}"},
        {"name": "b", "components": [dict(same)], "view_sql": "SELECT * FROM {shared_fact}"},
    ]
    assert len(load_manifests(ok)) == 2
    ok[1]["components"] = [{**same, "primary_key": ["id"]}]
    with pytest.raises(ManifestError, match="defined differently"):
        load_manifests(ok)
    v1 = {
        "name": "old", "fact_source": "f", "contour_table": "contours", "geometry_column": "g",
        "fact_join_columns": ["k"], "contour_join_columns": ["k"], "output_columns": ["k", "g"],
    }
    clash = {"name": "new", "components": [{"name": "contours", "kind": "fact"}], "view_sql": "SELECT * FROM {contours}"}
    with pytest.raises(ManifestError, match="collides"):
        load_manifests([v1, clash])


def test_manifest_hash_tracks_every_part_of_the_definition():
    base = load_manifest(CONTOUR_SPEC)
    assert load_manifest(CONTOUR_SPEC).manifest_hash() == base.manifest_hash()
    for change in (
        lambda s: s["view"].update(select=["zone_id"]),
        lambda s: s.update(materialize="materialized"),
        lambda s: s["components"][1].update(primary_key=["zone_id", "polygon_coords"]),
    ):
        import copy

        spec = copy.deepcopy(CONTOUR_SPEC)
        change(spec)
        assert load_manifest(spec).manifest_hash() != base.manifest_hash()
