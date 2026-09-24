"""Dagster's advantages: an I/O-manager boundary, asset freshness keys,
idempotent re-materialization, partitions, asset metadata."""

from __future__ import annotations

import dataclasses
import inspect as pyinspect

import pytest
from sqlalchemy import inspect

from compat_support import MANIFEST, OTHER, PARTS_V2, capability, has_feature, joined_rows, original_rows, rewrite_contours
from sql_incremental import fingerprint
from sql_incremental.sink import make_sink, materialize, normalize, publish, stage
from sql_incremental.sink_postgres import SqlSink


def _asset(sink, paths):
    """The 'asset body': it never names a storage backend."""
    return materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)


@capability("dagster", "io_manager_swap", note="swapping the store is changing the sink")
def test_the_same_asset_runs_on_either_sink_with_identical_results(engine, tmp_path, paths):
    other = make_sink({"type": "duckdb", "path": str(tmp_path / "other.duckdb")})
    sinks = [SqlSink(engine), other]
    for sink in sinks:
        _asset(sink, paths)
    assert joined_rows(sinks[0]) == joined_rows(sinks[1]) == original_rows(paths)


@capability("dagster", "asset_freshness", note="materialization key changes iff inputs change")
def test_update_id_changes_with_facts_contours_manifest_and_loader(paths):
    def uid(manifest=MANIFEST, **kw):
        return fingerprint.compute_dataset_update_id_for_files(
            manifest, str(paths["facts"]), str(paths["contours"]), **kw
        )[0]

    base = uid()
    assert uid() == base  # deterministic
    assert uid(dataclasses.replace(MANIFEST, output_columns=("zone_id", "polygon_coords", "metric"))) != base
    assert uid(loader_version=fingerprint.LOADER_VERSION + 1) != base
    rewrite_contours(paths, PARTS_V2)
    assert uid() != base  # a contour change alone


@capability("dagster", "idempotent_rematerialize")
def test_rematerializing_identical_inputs_changes_nothing(sink, paths):
    _asset(sink, paths)
    marker = sink.current_update_id("zones")
    published, result = _asset(sink, paths)
    assert published == [] and result.dataset.status == "current"
    assert sink.current_update_id("zones") == marker


@capability("dagster", "asset_metadata", note="row counts and staging time on every materialization")
def test_receipts_expose_row_count_and_timestamps(sink, paths):
    _, result = _asset(sink, paths)
    receipt = result.dataset.to_dict()
    assert receipt["row_count"] == 2 and receipt["staged_at"]
    assert receipt["update_id"] == sink.current_update_id("zones")


@capability("dagster", "multi_object_asset", "beyond", note="Dagster models one object per asset")
def test_one_logical_dataset_is_several_physical_objects_published_together(engine, tmp_path, paths):
    keyed = dataclasses.replace(MANIFEST, keyed=True)
    materialize(keyed, str(paths["facts"]), str(paths["contours"]), SqlSink(engine))
    tables = inspect(engine).get_table_names(schema=keyed.compact_schema)
    assert len(tables) >= 3  # zone, zone_part, facts ...
    assert inspect(engine).get_view_names() == ["zones"]  # ... behind one logical name


def _crash_on_second_marker(sink, monkeypatch):
    """Make the sink die between the commit of the first and second dataset."""
    calls = []
    from sql_incremental import publish as publish_mod

    real = publish_mod._upsert_dataset_marker

    def boom(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("crash between the two commits")
        return real(*a, **k)

    monkeypatch.setattr(publish_mod, "_upsert_dataset_marker", boom)


def _publish_two_with_a_crash(sink, paths, monkeypatch):
    results = [stage(normalize(m, str(paths["facts"]), str(paths["contours"])), sink) for m in (MANIFEST, OTHER)]
    _crash_on_second_marker(sink, monkeypatch)
    with pytest.raises(RuntimeError):
        publish(sink, [MANIFEST, OTHER], results)
    return sink.current_update_id("zones"), sink.current_update_id("others")


@capability("dagster", "multi_asset_atomic_publish", "beyond", note="several datasets flip together or not at all ")
def test_a_crash_while_publishing_two_datasets_publishes_neither(engine, paths, monkeypatch):
    assert _publish_two_with_a_crash(SqlSink(engine), paths, monkeypatch) == (None, None)


@capability(
    "dagster",
    "partitions",
    "gap",
    note="close: per-key incremental staging in the sinks (large); adopt: not possible, partitions need Dagster's run/asset model",
)
def test_a_single_partition_can_be_materialized():
    assert "partition" in pyinspect.signature(materialize).parameters
