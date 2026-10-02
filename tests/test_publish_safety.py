"""Publish safety: read-only planning, `lock_timeout`, and the `keep_old`
rollback window. Runs on DuckDB, and on PostgreSQL with
SNAKEMAKE_SQL_TEST_PG_DSN (a *disposable* database: `public` and
`analytics_storage` are dropped and recreated; two throwaway roles are created
and dropped)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from types import SimpleNamespace

import pytest
import yaml
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError

from fixtures.make_fixtures import TRANSITIONS_VIEW_SQL, make_default_fixtures, make_transitions_fixtures, write_fixture, write_table
from sqlalchemy import event

from sqlsink import keep_old as ko
from sqlsink import metadata as meta_mod
from sqlsink.engine import (
    _lock_key_to_bigint,
    apply_lock_timeout,
    make_engine,
    resolve_analyze,
    resolve_keep_old,
    resolve_lock_timeout,
)
from sqlsink.manifest import load_manifest
from sqlsink.publish import PUBLISH_LOCK_KEY, PublishConflict, publish_tables
from sqlsink.sink import dataset_is_published, normalize_v2, publish_v2, stage_v2
from sqlsink.sink_postgres import SqlSink
from sqlsink.stage import fetch_marker, stage_table

PG_DSN = os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(not PG_DSN, reason="needs SNAKEMAKE_SQL_TEST_PG_DSN (a disposable PostgreSQL)")

COMPONENTS = [
    {"name": "fact", "kind": "fact", "primary_key": ["obs_id"]},
    {"name": "bridge", "kind": "bridge", "primary_key": ["anchor", "f21"], "priority": "priority"},
    {"name": "sector_paths", "kind": "dimension", "primary_key": ["region", "f21", "path"]},
]
VIEW_COLUMNS = "obs_id, region, flow, path, metric"


def transitions(materialize="view"):
    return load_manifest(
        {"name": "transitions", "components": COMPONENTS, "view_sql": TRANSITIONS_VIEW_SQL, "materialize": materialize}
    )


def rows(engine, sql, **params):
    with engine.connect() as conn:
        return [tuple(r) for r in conn.execute(text(sql), params).fetchall()]


def names(engine, schema=None):
    insp = inspect(engine)
    return set(insp.get_table_names(schema=schema)) | set(insp.get_view_names(schema=schema))


def pg_reset(dsn):
    eng = make_engine(dsn)
    with eng.begin() as conn:
        for schema in ("public", "analytics_storage"):
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        conn.execute(text('CREATE SCHEMA "public"'))
    return eng


@pytest.fixture
def empty_dsn(tmp_path):
    """A database in which nothing was ever staged: not even the marker tables."""
    if PG_DSN:
        pg_reset(PG_DSN).dispose()
        return PG_DSN
    return f"duckdb:///{tmp_path / 'empty.duckdb'}"


@pytest.fixture
def role():
    """A throwaway login-less role (PostgreSQL only)."""
    if not PG_DSN:
        yield None
        return
    eng = make_engine(PG_DSN)
    name = "sqlsink_test_runtime"
    with eng.begin() as conn:
        conn.execute(text(f'DROP OWNED BY "{name}"')) if conn.execute(
            text("SELECT 1 FROM pg_roles WHERE rolname = :n"), {"n": name}
        ).first() else None
        conn.execute(text(f'DROP ROLE IF EXISTS "{name}"'))
        conn.execute(text(f'CREATE ROLE "{name}"'))
    yield name
    with eng.begin() as conn:
        conn.execute(text(f'DROP OWNED BY "{name}"'))
        conn.execute(text(f'DROP ROLE IF EXISTS "{name}"'))
    eng.dispose()


def can_select(engine, role, relation):
    return rows(engine, "SELECT has_table_privilege(:r, :rel, 'SELECT')", r=role, rel=relation)[0][0]


# -- 1. read-only planning ---------------------------------------------------------


def _provider(dsn, tmp_path, **kw):
    pytest.importorskip("snakemake_storage_plugin_sqlsink")
    from snakemake_storage_plugin_sqlsink import StorageProvider, StorageProviderSettings

    return StorageProvider(
        local_prefix=tmp_path / "local", logger=logging.getLogger("test"), settings=StorageProviderSettings(dsn=dsn, **kw)
    )


def _obj(provider, query):
    from snakemake_storage_plugin_sqlsink import StorageObject

    return StorageObject(query=query, keep_local=True, retrieve=True, provider=provider)


def test_planning_on_an_empty_database_creates_no_table(empty_dsn, tmp_path):
    parquet = make_default_fixtures(tmp_path / "pq")
    make_transitions_fixtures(tmp_path / "pq")
    manifests = tmp_path / "manifests.yaml"
    manifests.write_text(yaml.safe_dump({"datasets": [{"name": "transitions", "components": COMPONENTS, "view_sql": TRANSITIONS_VIEW_SQL}]}))
    grants = tmp_path / "grants.yaml"
    grants.write_text("runtime:\n  - fake_a\n  - dataset:transitions\n")
    provider = _provider(
        empty_dsn, tmp_path, parquet_dir=str(tmp_path / "pq"), manifests=str(manifests), grants_file=str(grants)
    )
    engine = provider.engine
    before = names(engine)

    queries = ["fake_a", "published/fake_a", "dataset/transitions"]
    if engine.dialect.name == "postgresql":
        queries.append("grants/runtime")
    cache = SimpleNamespace(exists_in_storage={}, mtime={}, size={})
    for query in queries:
        obj = _obj(provider, query)
        assert obj.exists() is False, query
        assert obj.mtime() == 0.0, query
        asyncio.run(obj.inventory(cache))
        assert cache.exists_in_storage[obj.cache_key()] is False
    _obj(provider, "published/fake_a").retrieve_object()  # a receipt from markers only: no write either
    assert before == names(engine) == set()
    assert not any(n.startswith("_pipeline_meta") for n in names(engine))
    if engine.dialect.name == "postgresql":
        assert "analytics_storage" not in inspect(engine).get_schema_names()
        assert rows(engine, "SELECT count(*) FROM pg_class WHERE relnamespace = 'public'::regnamespace")[0][0] == 0
    del parquet


def test_dataset_is_published_is_read_only_on_a_database_without_markers(tmp_path):
    dsn_free = {"type": "duckdb", "path": str(tmp_path / "x.duckdb")}
    import duckdb

    duckdb.connect(dsn_free["path"]).close()
    assert dataset_is_published(dsn_free, transitions()) is False
    con = duckdb.connect(dsn_free["path"])
    assert con.execute("SELECT count(*) FROM duckdb_tables()").fetchone()[0] == 0
    con.close()


def test_staging_and_publishing_create_the_marker_tables(empty_dsn, tmp_path):
    parquet = make_default_fixtures(tmp_path / "pq")
    engine = make_engine(empty_dsn)
    assert names(engine) == set()
    assert fetch_marker(engine, "fake_a") is None  # a read on a database without markers
    receipt = stage_table(engine, "fake_a", str(parquet["fake_a"]))
    assert "_pipeline_meta_staged_updates" in names(engine)
    publish_tables(engine, [receipt.to_dict()])
    assert fetch_marker(engine, "fake_a")["row_count"] == 3
    engine.dispose()


def test_staging_a_v2_dataset_creates_the_marker_tables(empty_dsn, tmp_path):
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(make_engine(empty_dsn))
    manifest = transitions()
    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    assert "_pipeline_meta_staged_component_updates" in names(sink.engine)
    assert publish_v2(sink, [manifest], [result]) == ["transitions"]
    sink.engine.dispose()


# -- 2. lock_timeout ------------------------------------------------------------------


def test_resolve_lock_timeout(monkeypatch):
    monkeypatch.delenv("SQLSINK_LOCK_TIMEOUT", raising=False)
    assert resolve_lock_timeout() is None
    assert resolve_lock_timeout("45s") == "45s"
    assert resolve_lock_timeout(500) == "500"
    assert resolve_lock_timeout("0") == "0"
    monkeypatch.setenv("SQLSINK_LOCK_TIMEOUT", "2min")
    assert resolve_lock_timeout() == "2min"
    assert resolve_lock_timeout("1s") == "1s"  # an explicit value wins over the environment
    for bad in ("soon", "-1s", "1 fortnight", "1s; DROP TABLE x"):
        with pytest.raises(ValueError):
            resolve_lock_timeout(bad)


def test_lock_timeout_is_a_noop_on_duckdb_and_local_on_postgres(engine):
    with engine.begin() as conn:
        apply_lock_timeout(conn, "1s")
        if engine.dialect.name == "postgresql":
            assert conn.execute(text("SHOW lock_timeout")).scalar() == "1s"
    if engine.dialect.name == "postgresql":  # SET LOCAL: gone with the transaction
        assert rows(engine, "SHOW lock_timeout")[0][0] == "0"


def test_resolve_keep_old(monkeypatch):
    monkeypatch.delenv("SQLSINK_KEEP_OLD", raising=False)
    assert resolve_keep_old() is False and resolve_keep_old(True) is True
    monkeypatch.setenv("SQLSINK_KEEP_OLD", "1")
    assert resolve_keep_old() is True and resolve_keep_old(False) is False


def _change(path, rows_):
    write_fixture(path, rows_)


@needs_pg
def test_publish_blocked_by_another_session_fails_within_the_lock_timeout_and_rolls_back(engine, parquet_dir):
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    old_rows = rows(engine, "SELECT * FROM fake_a ORDER BY id")
    old_marker = fetch_marker(engine, "fake_a")
    _change(parquet_dir["fake_a"], [(1, "changed", 9.0)])
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    assert receipt["status"] == "staged"

    blocker = make_engine(PG_DSN).connect()
    blocker.execute(text("LOCK TABLE fake_a IN ACCESS SHARE MODE"))  # a running query: the rename must wait for it
    try:
        started = time.monotonic()
        with pytest.raises(DBAPIError, match="lock timeout|lock_timeout"):
            publish_tables(engine, [receipt], lock_timeout="400ms")
        assert time.monotonic() - started < 10
        # rolled back: live relation and marker untouched, staged copy still there, no leftovers
        assert rows(engine, "SELECT * FROM fake_a ORDER BY id") == old_rows
        assert fetch_marker(engine, "fake_a")["update_id"] == old_marker["update_id"]
        assert "stg__fake_a" in names(engine)
        assert not [n for n in names(engine) if "__old__" in n]
        # the publish advisory lock did not leak from the failed transaction
        other = make_engine(PG_DSN).connect()
        try:
            key = {"k": _lock_key_to_bigint(PUBLISH_LOCK_KEY)}
            assert other.execute(text("SELECT pg_try_advisory_lock(:k)"), key).scalar()
            other.execute(text("SELECT pg_advisory_unlock(:k)"), key)
        finally:
            other.close()
    finally:
        blocker.rollback()
        blocker.close()
    assert publish_tables(engine, [receipt], lock_timeout="5s") == ["fake_a"]  # and it works once the lock is gone
    assert rows(engine, "SELECT name FROM fake_a") == [("changed",)]


@needs_pg
def test_lock_timeout_from_the_environment_bounds_a_dataset_publish(engine, tmp_path, monkeypatch):
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(engine)
    manifest = transitions()
    with engine.begin() as conn:  # a flat table, as on the remote
        conn.execute(text("CREATE TABLE transitions (obs_id BIGINT, region TEXT, flow DOUBLE PRECISION, path TEXT, metric DOUBLE PRECISION)"))
        conn.execute(text("INSERT INTO transitions VALUES (1, 'R1', 1.0, 'ALL', NULL)"))
    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    blocker = make_engine(PG_DSN).connect()
    blocker.execute(text("LOCK TABLE transitions IN ACCESS SHARE MODE"))
    monkeypatch.setenv("SQLSINK_LOCK_TIMEOUT", "400ms")
    try:
        started = time.monotonic()
        with pytest.raises(DBAPIError, match="lock timeout|lock_timeout"):
            publish_v2(sink, [manifest], [result])
        assert time.monotonic() - started < 10
        insp = inspect(engine)
        assert "transitions" in insp.get_table_names() and "transitions" not in insp.get_view_names()
        assert rows(engine, "SELECT count(*) FROM transitions") == [(1,)]
        assert "analytics_storage" not in insp.get_schema_names() or not insp.get_table_names(schema="analytics_storage") or all(
            n.startswith("stg__") for n in insp.get_table_names(schema="analytics_storage")
        )
    finally:
        blocker.rollback()
        blocker.close()
    monkeypatch.delenv("SQLSINK_LOCK_TIMEOUT")
    assert publish_v2(sink, [manifest], [result]) == ["transitions"]
    assert "transitions" in inspect(engine).get_view_names()


def test_publish_accepts_a_lock_timeout_on_every_dialect(engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    assert publish_tables(engine, [receipt], lock_timeout="30s") == ["fake_a"]


def test_publishing_an_already_published_table_receipt_again_is_a_noop(engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    assert publish_tables(engine, [receipt]) == ["fake_a"]
    assert publish_tables(engine, [receipt]) == []  # its staging table is gone: must not be swapped again
    assert publish_tables(engine, [receipt], refresh=True) == []


def test_stale_staged_table_receipt_of_another_version_still_conflicts(engine, parquet_dir):
    stale = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    _change(parquet_dir["fake_a"], [(1, "v2", 1.0)])
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    with pytest.raises(PublishConflict):
        publish_tables(engine, [stale])


# -- 3. keep_old ------------------------------------------------------------------------


def old_names(engine, schema=None):
    return sorted(n for n in names(engine, schema) if n.startswith("__old__"))


def test_default_publish_keeps_nothing_aside(engine, parquet_dir):
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    _change(parquet_dir["fake_a"], [(1, "v2", 1.0)])
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    assert [n for n in names(engine) if "__old__" in n] == []
    assert ko.list_kept(engine) == []


def test_keep_old_tables_rollback_and_cleanup(engine, parquet_dir, role):
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    v1 = rows(engine, "SELECT * FROM fake_a ORDER BY id")
    if role:
        with engine.begin() as conn:
            conn.execute(text(f'GRANT SELECT ON fake_a TO "{role}"'))
            conn.execute(text(f'GRANT SELECT ON public.fake_a TO PUBLIC'))
    _change(parquet_dir["fake_a"], [(1, "v2", 1.0), (2, "v2b", 2.0)])
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    assert publish_tables(engine, [receipt], keep_old=True) == ["fake_a"]
    v2 = rows(engine, "SELECT * FROM fake_a ORDER BY id")
    assert v2 != v1 and len(v2) == 2
    assert old_names(engine) == ["__old__fake_a"]
    assert rows(engine, 'SELECT * FROM "__old__fake_a" ORDER BY id') == v1  # old data still readable
    assert [(k.name, k.kind) for k in ko.list_kept(engine)] == [("fake_a", "table")]
    if role:  # grants must not leak: the kept relation is readable by its owner only
        assert can_select(engine, role, 'public."__old__fake_a"') is False
        assert can_select(engine, "public", 'public."__old__fake_a"') is False
        assert rows(engine, "SELECT relacl IS NULL OR relacl::text NOT LIKE '%=r/%' FROM pg_class WHERE relname = '__old__fake_a'") == [(True,)]

    # a second keep_old publish must not silently mix generations
    _change(parquet_dir["fake_a"], [(1, "v3", 1.0)])
    receipt3 = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    with pytest.raises(ko.KeptOldExists):
        publish_tables(engine, [receipt3], keep_old=True)
    assert rows(engine, "SELECT * FROM fake_a ORDER BY id") == v2  # rolled back, nothing changed
    assert old_names(engine) == ["__old__fake_a"]

    # rollback: v1 is live again, v2 kept aside (so rolling back is itself undoable)
    assert ko.rollback_kept_old(engine) == [f"{'public' if engine.dialect.name == 'postgresql' else 'main'}.fake_a"]
    assert rows(engine, "SELECT * FROM fake_a ORDER BY id") == v1
    assert rows(engine, 'SELECT * FROM "__old__fake_a" ORDER BY id') == v2
    assert fetch_marker(engine, "fake_a") is None  # the marker vouched for v2: forgotten
    if role:  # the restored table has its grants back, the parked v2 has none
        assert can_select(engine, role, "public.fake_a") is True
        assert can_select(engine, "public", "public.fake_a") is True
        assert can_select(engine, role, 'public."__old__fake_a"') is False
        assert can_select(engine, "public", 'public."__old__fake_a"') is False
    ko.rollback_kept_old(engine)
    assert rows(engine, "SELECT * FROM fake_a ORDER BY id") == v2

    assert [r.rsplit(".", 1)[-1] for r in ko.cleanup_kept_old(engine)] == ["__old__fake_a"]
    assert old_names(engine) == [] and ko.list_kept(engine) == []
    assert ko.cleanup_kept_old(engine) == [] and ko.rollback_kept_old(engine) == []


def test_keep_old_from_the_environment(engine, parquet_dir, monkeypatch):
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    _change(parquet_dir["fake_a"], [(1, "v2", 1.0)])
    monkeypatch.setenv("SQLSINK_KEEP_OLD", "1")
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    assert old_names(engine) == ["__old__fake_a"]


def test_only_tagged_old_relations_are_ever_touched(engine, parquet_dir):
    with engine.begin() as conn:
        conn.execute(text('CREATE TABLE "__old__mine" (a INTEGER)'))
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    assert ko.list_kept(engine) == [] and ko.cleanup_kept_old(engine) == [] and ko.rollback_kept_old(engine) == []
    assert "__old__mine" in names(engine)


def test_keep_old_selection_by_name(engine, parquet_dir):
    publish_tables(engine, [stage_table(engine, n, str(parquet_dir[n])).to_dict() for n in ("fake_a", "fake_b")])
    _change(parquet_dir["fake_a"], [(1, "a2", 1.0)])
    _change(parquet_dir["fake_b"], [(1, "b2", 1.0)])
    publish_tables(engine, [stage_table(engine, n, str(parquet_dir[n])).to_dict() for n in ("fake_a", "fake_b")], keep_old=True)
    assert old_names(engine) == ["__old__fake_a", "__old__fake_b"]
    ko.cleanup_kept_old(engine, names=["fake_a"])
    assert old_names(engine) == ["__old__fake_b"]


def _flat_transitions(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE transitions (obs_id BIGINT, region VARCHAR, flow DOUBLE PRECISION, path VARCHAR, metric DOUBLE PRECISION)"))
        conn.execute(text("INSERT INTO transitions VALUES (1, 'FLAT', 1.0, 'ALL', NULL)"))


def _publish_transitions(sink, toy, **kw):
    manifest = transitions()
    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    return publish_v2(sink, [manifest], [result], **kw)


def _view_rows(engine, name="transitions"):
    return sorted(rows(engine, f'SELECT {VIEW_COLUMNS} FROM "{name}"'), key=lambda r: (r[0], r[3]))


def test_keep_old_replaces_a_flat_table_by_a_view_and_rolls_back(engine, tmp_path, role):
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(engine)
    _flat_transitions(engine)
    flat = _view_rows(engine)
    if role:
        with engine.begin() as conn:
            conn.execute(text(f'GRANT SELECT ON transitions TO "{role}"'))
    assert _publish_transitions(sink, toy, keep_old=True) == ["transitions"]
    insp = inspect(engine)
    assert "transitions" in insp.get_view_names()
    assert "__old__transitions" in insp.get_table_names() and "__old__transitions" not in insp.get_view_names()
    assert _view_rows(engine, "__old__transitions") == flat  # the flat table, still readable
    assert _view_rows(engine) != flat
    if role:
        assert can_select(engine, role, 'public."__old__transitions"') is False
        assert can_select(engine, role, "public.transitions") is False  # (no default privileges here)
    view = _view_rows(engine)

    ko.rollback_kept_old(engine)
    insp = inspect(engine)
    assert "transitions" in insp.get_table_names() and "transitions" not in insp.get_view_names()
    assert _view_rows(engine) == flat
    if role:
        assert can_select(engine, role, "public.transitions") is True
    assert rows(engine, "SELECT count(*) FROM _pipeline_meta_dataset_updates WHERE dataset_name = 'transitions'") == [(0,)]
    if engine.dialect.name == "postgresql":  # the replaced view is kept too, and still reads its own components
        assert "__old__transitions" in insp.get_view_names()
        assert _view_rows(engine, "__old__transitions") == view
        if role:
            assert can_select(engine, role, 'public."__old__transitions"') is False
    ko.cleanup_kept_old(engine)
    assert old_names(engine) == [] and old_names(engine, "analytics_storage") == []
    assert _view_rows(engine) == flat


def test_keep_old_keeps_replaced_components_and_views_consistent(engine, tmp_path):
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(engine)
    assert _publish_transitions(sink, toy) == ["transitions"]
    v1 = _view_rows(engine)
    write_table(
        toy["fact"], "obs_id BIGINT, region VARCHAR, anchor VARCHAR, flow DOUBLE",
        [(1, "R1", "A", 99.0), (2, "R1", "B", 20.25), (3, "R2", "A", None)],
    )
    assert _publish_transitions(sink, toy, keep_old=True) == ["transitions"]
    v2 = _view_rows(engine)
    assert v2 != v1
    assert "__old__fact" in names(engine, "analytics_storage")
    assert "__old__bridge" not in names(engine, "analytics_storage")  # unchanged components are not replaced
    if engine.dialect.name == "postgresql":  # a kept old view follows the kept old component
        assert _view_rows(engine, "__old__transitions") == v1
    ko.rollback_kept_old(engine)
    assert _view_rows(engine) == v1
    ko.rollback_kept_old(engine)
    assert _view_rows(engine) == v2
    ko.cleanup_kept_old(engine)
    assert old_names(engine) == [] and old_names(engine, "analytics_storage") == []
    assert _view_rows(engine) == v2


def test_keep_old_cli(engine, parquet_dir, monkeypatch, capsys):
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    _change(parquet_dir["fake_a"], [(1, "v2", 1.0)])
    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()], keep_old=True)
    monkeypatch.setenv("SQLSINK_TEST_CLI_DSN", engine.url.render_as_string(hide_password=False))
    assert ko.main(["list", "--dsn-env", "SQLSINK_TEST_CLI_DSN"]) == 0
    assert "__old__fake_a (table)" in capsys.readouterr().out
    assert ko.main(["cleanup", "--dsn-env", "SQLSINK_TEST_CLI_DSN", "--lock-timeout", "5s"]) == 0
    assert "cleanup: 1 relation(s)" in capsys.readouterr().out
    assert old_names(engine) == []


@needs_pg
def test_keep_old_refuses_an_identifier_postgres_would_truncate(engine):
    long_name = "x" * 60
    with engine.begin() as conn:
        conn.execute(text(f'CREATE TABLE "{long_name}" (a INTEGER)'))
        with pytest.raises(ValueError, match="identifier limit"):
            ko.park(conn, long_name, None)


# -- 4. analyze -------------------------------------------------------------------------


class _capture_analyze:
    """Records every `ANALYZE ...` statement `engine` executes while this is
    active, so tests can assert exactly which relations were analyzed
    without depending on timing-sensitive `pg_stat_user_tables` columns."""

    def __init__(self, engine):
        self.engine = engine
        self.statements: list[str] = []

    def _listener(self, conn, cursor, statement, parameters, context, executemany):
        if statement.strip().upper().startswith("ANALYZE"):
            self.statements.append(statement.strip())

    def __enter__(self):
        event.listen(self.engine, "before_cursor_execute", self._listener)
        return self

    def __exit__(self, *exc):
        event.remove(self.engine, "before_cursor_execute", self._listener)


def test_resolve_analyze(monkeypatch):
    monkeypatch.delenv("SQLSINK_ANALYZE", raising=False)
    assert resolve_analyze() is True  # unlike keep_old/lock_timeout, this defaults ON
    assert resolve_analyze(True) is True and resolve_analyze(False) is False
    monkeypatch.setenv("SQLSINK_ANALYZE", "0")
    assert resolve_analyze() is False
    assert resolve_analyze(True) is True  # an explicit value wins over the environment
    for on in ("1", "true", "Yes", "ON"):
        monkeypatch.setenv("SQLSINK_ANALYZE", on)
        assert resolve_analyze() is True
    for off in ("0", "false", "No", "OFF"):
        monkeypatch.setenv("SQLSINK_ANALYZE", off)
        assert resolve_analyze() is False
    monkeypatch.setenv("SQLSINK_ANALYZE", "sometimes")
    with pytest.raises(ValueError):
        resolve_analyze()


def test_analyze_is_a_noop_on_duckdb(engine, parquet_dir):
    if engine.dialect.name != "duckdb":
        pytest.skip("duckdb-specific: exercised for real by the postgres tests below")
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    with _capture_analyze(engine) as cap:
        assert publish_tables(engine, [receipt], analyze=True) == ["fake_a"]
    assert cap.statements == []


@needs_pg
def test_analyze_runs_after_commit_with_the_publish_lock_already_released(engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    lock_free_when_analyzed = []
    marker_visible_when_analyzed = []

    def _check(conn, cursor, statement, parameters, context, executemany):
        if not statement.strip().upper().startswith("ANALYZE"):
            return
        other = make_engine(PG_DSN).connect()
        try:
            key = {"k": _lock_key_to_bigint(PUBLISH_LOCK_KEY)}
            got = other.execute(text("SELECT pg_try_advisory_lock(:k)"), key).scalar()
            lock_free_when_analyzed.append(bool(got))
            if got:
                other.execute(text("SELECT pg_advisory_unlock(:k)"), key)
            marker_visible_when_analyzed.append(
                other.execute(
                    text("SELECT update_id FROM _pipeline_meta_table_updates WHERE table_name = 'fake_a'")
                ).scalar()
                is not None
            )
            other.rollback()
        finally:
            other.close()

    event.listen(engine, "before_cursor_execute", _check)
    try:
        assert publish_tables(engine, [receipt]) == ["fake_a"]
    finally:
        event.remove(engine, "before_cursor_execute", _check)

    assert lock_free_when_analyzed == [True]  # the transaction holding the advisory lock had already committed
    assert marker_visible_when_analyzed == [True]  # ...and so had the publish itself, from another session
    assert rows(engine, "SELECT count(*) FROM pg_stats WHERE schemaname = 'public' AND tablename = 'fake_a'")[0][0] > 0


@needs_pg
def test_analyze_after_publish_datasets_also_runs_with_the_lock_already_released(engine, tmp_path):
    """Same proof as `test_analyze_runs_after_commit_with_the_publish_lock_already_released`,
    for the `publish_datasets`/`publish_v2` path (the one production datasets
    actually go through), not just `publish_tables`."""
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(engine)
    manifest = transitions()
    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)

    lock_free_when_analyzed = []
    marker_visible_when_analyzed = []

    def _check(conn, cursor, statement, parameters, context, executemany):
        if not statement.strip().upper().startswith("ANALYZE"):
            return
        other = make_engine(PG_DSN).connect()
        try:
            key = {"k": _lock_key_to_bigint(PUBLISH_LOCK_KEY)}
            got = other.execute(text("SELECT pg_try_advisory_lock(:k)"), key).scalar()
            lock_free_when_analyzed.append(bool(got))
            if got:
                other.execute(text("SELECT pg_advisory_unlock(:k)"), key)
            marker_visible_when_analyzed.append(
                other.execute(
                    text("SELECT update_id FROM _pipeline_meta_dataset_updates WHERE dataset_name = 'transitions'")
                ).scalar()
                is not None
            )
            other.rollback()
        finally:
            other.close()

    event.listen(engine, "before_cursor_execute", _check)
    try:
        assert publish_v2(sink, [manifest], [result]) == ["transitions"]
    finally:
        event.remove(engine, "before_cursor_execute", _check)

    assert lock_free_when_analyzed and all(lock_free_when_analyzed)
    assert marker_visible_when_analyzed and all(marker_visible_when_analyzed)


@needs_pg
def test_analyze_targets_the_published_table_by_name(engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    with _capture_analyze(engine) as cap:
        assert publish_tables(engine, [receipt]) == ["fake_a"]
    assert cap.statements == ['ANALYZE "fake_a"']


@needs_pg
def test_analyze_can_be_disabled_by_kwarg_or_environment(engine, parquet_dir, monkeypatch):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    with _capture_analyze(engine) as cap:
        publish_tables(engine, [receipt], analyze=False)
    assert cap.statements == []
    assert rows(engine, "SELECT count(*) FROM pg_stats WHERE schemaname = 'public' AND tablename = 'fake_a'")[0][0] == 0

    _change(parquet_dir["fake_a"], [(1, "v2", 1.0)])
    receipt2 = stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()
    monkeypatch.setenv("SQLSINK_ANALYZE", "0")
    with _capture_analyze(engine) as cap:
        assert publish_tables(engine, [receipt2]) == ["fake_a"]
    assert cap.statements == []


@needs_pg
def test_analyze_targets_only_the_swapped_v2_components_never_the_plain_view(engine, tmp_path):
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(engine)
    manifest = transitions()  # materialize="view" by default: a plain view holds no data of its own

    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    with _capture_analyze(engine) as cap:
        assert publish_v2(sink, [manifest], [result]) == ["transitions"]
    assert sorted(cap.statements) == sorted(
        f'ANALYZE "analytics_storage"."{n}"' for n in ("fact", "bridge", "sector_paths")
    )
    assert not any("transitions" in s for s in cap.statements)

    # only `fact` changes: only `fact` is re-analyzed, `bridge`/`sector_paths` are untouched
    write_table(
        toy["fact"], "obs_id BIGINT, region VARCHAR, anchor VARCHAR, flow DOUBLE",
        [(1, "R1", "A", 99.0), (2, "R1", "B", 20.25), (3, "R2", "A", None)],
    )
    result2 = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    with _capture_analyze(engine) as cap2:
        assert publish_v2(sink, [manifest], [result2]) == ["transitions"]
    assert cap2.statements == ['ANALYZE "analytics_storage"."fact"']


@needs_pg
def test_analyze_targets_a_materialized_view_dataset_too(engine, tmp_path):
    toy = make_transitions_fixtures(tmp_path / "toy")
    sink = SqlSink(engine)
    manifest = transitions(materialize="materialized")
    result = stage_v2(normalize_v2(manifest, toy, sink=sink), sink)
    with _capture_analyze(engine) as cap:
        assert publish_v2(sink, [manifest], [result]) == ["transitions"]
    assert 'ANALYZE "transitions"' in cap.statements  # the matview holds its own copy of the data


@needs_pg
def test_analyze_targets_the_compact_contour_and_zone_tables_of_a_keyed_v1_dataset(engine, tmp_path):
    from fixtures.make_fixtures import make_multipart_geometry_fixtures
    from sqlsink.manifest import DatasetMaterialization
    from sqlsink.sink import materialize as materialize_v1

    manifest = DatasetMaterialization(
        name="zones",
        geometry_column="polygon_coords",
        contour_table="contours",
        fact_join_columns=("zone_id",),
        contour_join_columns=("zone_id",),
        output_columns=("zone_id", "metric", "polygon_coords"),
        fact_source="facts",
        keyed=True,
    )
    paths = make_multipart_geometry_fixtures(tmp_path / "parquet")
    with _capture_analyze(engine) as cap:
        published, _ = materialize_v1(manifest, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    assert published == ["zones"]
    assert sorted(cap.statements) == sorted(
        f'ANALYZE "analytics_storage"."{n}"' for n in (meta_mod.compact_table_name("zones"), "contours", "contours_zone")
    )
    assert 'ANALYZE "zones"' not in cap.statements  # the public compatibility relation is a view, never ANALYZEd
