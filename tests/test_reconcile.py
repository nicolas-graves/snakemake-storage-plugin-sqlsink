"""PostgreSQL-only coverage for explicit authoritative reconciliation."""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text

from sqlsink import DesiredState, ReconciliationError
from sqlsink.reconcile import desired_state
from sqlsink.publish import publish_tables
from sqlsink.sink_postgres import SqlSink
from sqlsink.stage import stage_table


def _postgres(engine):
    if engine.dialect.name != "postgresql":
        pytest.skip("authoritative reconciliation is PostgreSQL-first")


def _publish(engine, parquet_dir, *names):
    publish_tables(
        engine,
        [stage_table(engine, name, str(parquet_dir[name])).to_dict() for name in names],
    )


def test_partial_mapping_is_not_accepted_as_complete_desired_state():
    with pytest.raises(ReconciliationError, match="complete 'tables'"):
        desired_state({"datasets": []})
    assert desired_state({"datasets": [], "tables": []}) == DesiredState()


def test_reconcile_is_report_first_and_requires_authority(engine, parquet_dir):
    _postgres(engine)
    _publish(engine, parquet_dir, "fake_a", "fake_b")
    sink = SqlSink(engine)

    report = sink.reconcile(DesiredState(tables=("fake_a",)))
    assert {o.name for o in report.obsolete} == {"fake_b"}
    with pytest.raises(ReconciliationError, match="authoritative=True"):
        report.prune()
    assert inspect(engine).has_table("fake_b", schema="public")

    report = sink.reconcile(DesiredState(tables=("fake_a",)), authoritative=True)
    assert report.prune() is report  # dry-run is still mutation-free
    assert inspect(engine).has_table("fake_b", schema="public")
    report.prune(execute=True)
    assert not inspect(engine).has_table("fake_b", schema="public")
    assert sink.reconcile(DesiredState(tables=("fake_a",)), authoritative=True).clean


def test_marker_only_catalog_only_and_schema_boundaries(engine, parquet_dir):
    _postgres(engine)
    _publish(engine, parquet_dir, "fake_a", "fake_b")
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE "public"."fake_b"'))  # stale marker only
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS "analytics_storage"'))
        conn.execute(text('CREATE TABLE "analytics_storage"."orphan_component" (x integer)'))
        conn.execute(text('CREATE SCHEMA "other_owner"'))
        conn.execute(text('CREATE TABLE "other_owner"."untouched" (x integer)'))
        conn.execute(text('CREATE TABLE "public"."unmanaged" (x integer)'))

    report = SqlSink(engine).reconcile(DesiredState(tables=("fake_a",)), authoritative=True)
    assert ("plain_marker", "fake_b") in {(o.kind, o.name) for o in report.obsolete}
    assert ("component", "orphan_component") in {(o.kind, o.name) for o in report.obsolete}
    assert "unmanaged" not in {o.name for o in report.obsolete}
    assert "untouched" not in {o.name for o in report.obsolete}

    report.prune(execute=True)
    insp = inspect(engine)
    assert not insp.has_table("orphan_component", schema="analytics_storage")
    assert insp.has_table("unmanaged", schema="public")
    assert insp.has_table("untouched", schema="other_owner")


def test_prune_rolls_back_objects_and_markers_together(engine, parquet_dir):
    _postgres(engine)
    _publish(engine, parquet_dir, "fake_a", "fake_b")
    with engine.begin() as conn:
        conn.execute(text('CREATE VIEW "public"."outside" AS SELECT * FROM "public"."fake_b"'))

    report = SqlSink(engine).reconcile(DesiredState(), authoritative=True)
    with pytest.raises(Exception):
        report.prune(execute=True)

    # fake_a was visited before fake_b, but PostgreSQL transactional DDL and
    # marker deletion make the failure all-or-nothing.
    assert inspect(engine).has_table("fake_a", schema="public")
    assert inspect(engine).has_table("fake_b", schema="public")
    again = SqlSink(engine).reconcile(DesiredState(), authoritative=True)
    assert {"fake_a", "fake_b"} <= {o.name for o in again.obsolete}
