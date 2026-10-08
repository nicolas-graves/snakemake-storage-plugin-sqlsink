"""Planning snapshots preserve the same marker and relation decisions."""

from sqlsink.planning import PlanningState, read_planning_state, shared_planning_state, invalidate_shared_planning_state
from sqlsink.metadata import staging_name
from sqlsink.manifest import load_manifest


def state():
    return PlanningState(
        {"table": {}, "staged": {}, "dataset": {}, "component": {}},
        {},
        set(),
    )


def test_shared_snapshot_reuses_state_and_expands_roles(monkeypatch):
    from sqlalchemy import create_engine
    from sqlsink import planning

    engine = create_engine("sqlite:///:memory:")
    calls = []

    def read(_engine, *, roles):
        calls.append(roles)
        return state()

    monkeypatch.setattr(planning, "read_planning_state", read)
    invalidate_shared_planning_state()
    try:
        first = shared_planning_state(engine, roles=["a"])
        assert shared_planning_state(engine, roles=["a"]) is first
        second = shared_planning_state(engine, roles=["b"])
        assert second is not first
        assert calls == [["a"], ["a", "b"]]
        invalidate_shared_planning_state()
        assert shared_planning_state(engine) is not second
    finally:
        invalidate_shared_planning_state()
        engine.dispose()


def test_table_state_requires_matching_marker_and_relation():
    s = state()
    s.markers["table"]["facts"] = {"update_id": "a"}
    assert not s.table_current("facts", "a")
    s.relations[("public", "facts")] = "r"
    assert s.table_current("facts", "a")
    assert not s.table_current("facts", "b")
    s.markers["staged"]["facts"] = {"update_id": "b"}
    assert not s.published_table_current("facts")
    s.relations[("public", staging_name("facts"))] = "r"
    assert s.table_current("facts", "b")


def test_select_requires_every_relation():
    s = state()
    s.grants.add(("runtime", "public", "view_a"))
    assert s.role_has_select("runtime", ["view_a"])
    assert not s.role_has_select("runtime", ["view_a", "reporting.view_b"])


def test_relation_kinds_and_component_updates():
    manifest = load_manifest({
        "name": "report",
        "components": [{"name": "facts", "kind": "fact", "primary_key": ["id"]}],
        "view_sql": "SELECT * FROM {facts}",
    })
    s = state()
    s.dataset_components = {"report": {"facts": "v1"}}
    s.markers["component"]["facts"] = {"update_id": "v1"}
    s.relations[(manifest.schema_of(manifest.components[0]), "facts")] = "r"
    s.relations[("public", "report")] = "v"
    assert s.dataset_intact(manifest)
    s.relations[("public", "report")] = "r"
    assert not s.dataset_intact(manifest)
    s.relations[("public", "report")] = "v"
    s.markers["component"]["facts"]["update_id"] = "v2"
    assert not s.dataset_intact(manifest)


def test_postgres_snapshot_observes_revoked_select(engine):
    import pytest
    from sqlalchemy import text

    if engine.dialect.name != "postgresql":
        pytest.skip("needs PostgreSQL")
    with engine.begin() as conn:
        conn.execute(text("CREATE ROLE sqlsink_planning_reader"))
        conn.execute(text("CREATE TABLE planning_grant_probe (id integer)"))
        conn.execute(text("GRANT SELECT ON planning_grant_probe TO sqlsink_planning_reader"))
    try:
        granted = read_planning_state(engine, roles=["sqlsink_planning_reader"])
        assert granted.role_has_select("sqlsink_planning_reader", ["planning_grant_probe"])
        with engine.begin() as conn:
            conn.execute(text("REVOKE SELECT ON planning_grant_probe FROM sqlsink_planning_reader"))
        revoked = read_planning_state(engine, roles=["sqlsink_planning_reader"])
        assert not revoked.role_has_select("sqlsink_planning_reader", ["planning_grant_probe"])
    finally:
        with engine.begin() as conn:
            conn.execute(text("DROP ROLE sqlsink_planning_reader"))


def test_reader_does_not_create_markers():
    import os
    import pytest
    from sqlalchemy import create_engine, inspect

    dsn = os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN")
    if not dsn:
        pytest.skip("needs disposable PostgreSQL")
    engine = create_engine(dsn)
    try:
        before = set(inspect(engine).get_table_names())
        snapshot = read_planning_state(engine, roles=["runtime"])
        assert isinstance(snapshot.markers["table"], dict)
        assert set(inspect(engine).get_table_names()) == before
    finally:
        engine.dispose()


def test_postgres_reader_uses_at_most_two_data_statements(engine):
    import pytest
    from sqlalchemy import event
    from sqlsink.metadata import create_all

    if engine.dialect.name != "postgresql":
        pytest.skip("needs PostgreSQL")
    statements = []

    def record(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        read_planning_state(engine)
        assert len([s for s in statements if not s.startswith("SET ")]) <= 2
        statements.clear()
        create_all(engine)
        statements.clear()
        read_planning_state(engine, roles=["runtime"])
        assert len([s for s in statements if not s.startswith("SET ")]) <= 2
    finally:
        event.remove(engine, "before_cursor_execute", record)
