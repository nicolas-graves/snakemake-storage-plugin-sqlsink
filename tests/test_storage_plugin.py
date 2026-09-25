"""Unit tests for the private storage plugin, exercised directly against a
StorageObject instance (no full Snakemake DAG needed for this level).
"""

from __future__ import annotations

import logging

import pytest

from sqlsink.publish import publish_tables
from sqlsink.stage import fetch_marker, stage_table

pytest.importorskip("snakemake_storage_plugin_sqlsink")

from snakemake_storage_plugin_sqlsink import (  # noqa: E402
    StorageObject,
    StorageProvider,
    StorageProviderSettings,
)


@pytest.fixture
def provider(engine, parquet_dir, tmp_path):
    p = StorageProvider(
        local_prefix=tmp_path / "local",
        logger=logging.getLogger("test"),
        settings=StorageProviderSettings(dsn="duckdb:///:memory:", parquet_dir=str(next(iter(parquet_dir.values())).parent)),
    )
    p.engine = engine  # reuse the test's in-memory engine/connection pool
    return p


def _obj(provider, table_name):
    return StorageObject(query=table_name, keep_local=True, retrieve=True, provider=provider)


def test_exists_false_before_any_publish(provider):
    assert _obj(provider, "fake_a").exists() is False


def test_exists_true_after_publish_and_mtime_reflects_marker(provider, engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])

    obj = _obj(provider, "fake_a")
    assert obj.exists() is True
    assert obj.mtime() > 0


def test_exists_becomes_false_again_after_parquet_changes(provider, engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])
    assert _obj(provider, "fake_a").exists() is True

    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 'changed', 1.0)")
    con.execute(f"COPY t TO '{parquet_dir['fake_a']}' (FORMAT PARQUET)")
    con.close()

    assert _obj(provider, "fake_a").exists() is False


def test_retrieve_object_writes_manifest_for_current_table(provider, engine, parquet_dir):
    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])

    obj = _obj(provider, "fake_a")
    obj.retrieve_object()

    import json

    manifest = json.loads(obj.local_path().read_text())
    assert manifest["table"] == "fake_a"
    assert manifest["status"] == "current"


def test_exists_false_when_the_published_table_was_dropped_but_its_marker_remains(provider, engine, parquet_dir):
    from sqlalchemy import text

    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    publish_tables(engine, [receipt.to_dict()])
    with engine.begin() as conn:
        conn.execute(text('DROP TABLE "fake_a"'))

    assert _obj(provider, "fake_a").exists() is False
    assert stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).status == "staged"


def test_exists_false_when_the_staging_table_was_dropped_but_its_marker_remains(provider, engine, parquet_dir):
    from sqlalchemy import text

    from sqlsink.metadata import staging_name

    stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    assert _obj(provider, "fake_a").exists() is True  # staged, awaiting publish
    with engine.begin() as conn:
        conn.execute(text(f'DROP TABLE "{staging_name("fake_a")}"'))

    assert _obj(provider, "fake_a").exists() is False


# -- published/{table} ----------------------------------------------------


def _restage_changed(engine, parquet_dir):
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE t (id BIGINT, name VARCHAR, score DOUBLE)")
    con.execute("INSERT INTO t VALUES (1, 'changed', 1.0)")
    con.execute(f"COPY t TO '{parquet_dir['fake_a']}' (FORMAT PARQUET)")
    con.close()
    return stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))


def test_published_object_lifecycle(provider, engine, parquet_dir):
    published = lambda: _obj(provider, "published/fake_a")  # noqa: E731
    assert published().exists() is False  # nothing published

    receipt = stage_table(engine, "fake_a", str(parquet_dir["fake_a"]))
    assert published().exists() is False  # staged only
    publish_tables(engine, [receipt.to_dict()])
    assert published().exists() is True
    assert published().mtime() > 0

    # Restage with new content: the published marker no longer matches the staged one.
    new = _restage_changed(engine, parquet_dir)
    assert published().exists() is False
    publish_tables(engine, [new.to_dict()])
    assert published().exists() is True


def test_published_receipt_carries_marker_content(provider, engine, parquet_dir):
    import json

    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    obj = _obj(provider, "published/fake_a")
    obj.retrieve_object()
    receipt = json.loads(obj.local_path().read_text())
    assert receipt["table"] == "fake_a"
    assert receipt["marker"]["row_count"] == 3
    assert "published_at" not in receipt["marker"]  # byte-stable across refreshes
    # Distinct local path from the staged object of the same table.
    assert obj.local_path() != _obj(provider, "fake_a").local_path()


def test_is_valid_query_accepts_kinds_and_rejects_paths():
    ok = StorageProvider.is_valid_query
    for q in ("t", "published/t", "dataset/d", "grants/r", "published/{table}"):
        assert ok(q).valid, q
    for q in ("a/b", "published/a/b", "published/", "s3://x"):
        assert not ok(q).valid, q


# -- fingerprint ----------------------------------------------------------


def test_fingerprint_is_byte_stable_and_tracks_changes(engine, parquet_dir):
    from sqlsink.fingerprint import fingerprint_json

    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    first = fingerprint_json(engine, ["fake_a", "missing_table"])
    blind = fingerprint_json(engine, ["fake_a"], count_rows=False)
    assert fingerprint_json(engine, ["missing_table", "fake_a"]) == first
    assert '"exists": false' in first

    with engine.begin() as conn:
        from sqlalchemy import text

        conn.execute(text("INSERT INTO fake_a VALUES (9, 'z', 0.0)"))
    assert fingerprint_json(engine, ["fake_a", "missing_table"]) != first
    # Without the row count, an out-of-band INSERT is the accepted blind spot (DuckDB has no size probe).
    if engine.dialect.name == "duckdb":
        assert fingerprint_json(engine, ["fake_a"], count_rows=False) == blind


# -- grants ---------------------------------------------------------------


def test_grants_object_receipt_and_dialect_guard(provider, engine, parquet_dir, tmp_path, monkeypatch):
    import json

    from sqlsink import grants

    publish_tables(engine, [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()])
    gf = tmp_path / "grants.yaml"
    gf.write_text("runtime:\n  - fake_b\n  - fake_a\n")
    provider.settings.grants_file = str(gf)

    obj = _obj(provider, "grants/runtime")
    obj.retrieve_object()
    receipt = json.loads(obj.local_path().read_text())
    assert receipt["relations"] == ["fake_a", "fake_b"]
    assert receipt["markers"]["fake_a"]["update_id"] and receipt["markers"]["fake_b"] is None
    assert "published_at" not in receipt["markers"]["fake_a"]
    assert obj.mtime() > 0

    if engine.dialect.name != "postgresql":
        with pytest.raises(NotImplementedError):
            obj.exists()
        with pytest.raises(NotImplementedError):
            grants.grant_runtime(engine, "runtime", ["fake_a"])
    else:
        monkeypatch.setattr(grants, "role_has_select", lambda *a: True)
        assert obj.exists() is True


# -- provider settings ----------------------------------------------------


def _provider(tmp_path, engine=None, **kw):
    return StorageProvider(
        local_prefix=tmp_path / "local", logger=logging.getLogger("test"), settings=StorageProviderSettings(**kw)
    )


def test_dsn_from_file_and_env(tmp_path, monkeypatch):
    from snakemake_storage_plugin_sqlsink import resolve_dsn

    f = tmp_path / "dsn"
    f.write_text(f"duckdb:///{tmp_path / 'a.duckdb'}\n")
    assert resolve_dsn(StorageProviderSettings(dsn_file=str(f))).endswith("a.duckdb")
    monkeypatch.setenv("SQLSINK_TEST_DSN", "duckdb:///:memory:")
    assert resolve_dsn(StorageProviderSettings(dsn_env="SQLSINK_TEST_DSN")) == "duckdb:///:memory:"
    with pytest.raises(ValueError):
        resolve_dsn(StorageProviderSettings())
    p = _provider(tmp_path, dsn_file=str(f))
    assert p.reachable


def test_on_unreachable(tmp_path):
    bad = "duckdb:////nonexistent-dir/x.duckdb"
    with pytest.raises(Exception):
        _provider(tmp_path, dsn=bad)
    p = _provider(tmp_path, dsn=bad, on_unreachable="treat-missing")
    assert p.reachable is False
    obj = StorageObject(query="t", keep_local=True, retrieve=True, provider=p)
    assert obj.exists() is False and obj.mtime() == 0.0
    with pytest.raises(ValueError):
        _provider(tmp_path, dsn=bad, on_unreachable="nope")


# -- dataset/{name} -------------------------------------------------------


def test_dataset_object(provider, engine, tmp_path):
    pytest.importorskip("pyarrow")
    import json

    from fixtures.make_fixtures import make_multipart_geometry_fixtures
    from sqlsink.materialize import stage_contours, stage_dataset
    from sqlsink.publish import publish_datasets
    from test_materialize_dataset import MANIFEST

    paths = make_multipart_geometry_fixtures(tmp_path / "pq")
    provider.settings.parquet_dir = str(tmp_path / "pq")
    # MANIFEST's default contour_source name.
    (tmp_path / "pq" / "zone_emploi_contours.parquet").write_bytes(paths["contours"].read_bytes())
    mf = tmp_path / "manifests.yaml"
    mf.write_text(
        "datasets:\n  - {name: zones, fact_source: facts, contour_table: contours, geometry_column: polygon_coords,\n"
        "     fact_join_columns: [zone_id], contour_join_columns: [zone_id], output_columns: [zone_id, metric, polygon_coords]}\n"
    )
    provider.settings.manifests = str(mf)
    obj = _obj(provider, "dataset/zones")
    assert obj.exists() is False

    c = stage_contours(engine, MANIFEST, str(paths["contours"])).to_dict()
    d = stage_dataset(engine, MANIFEST, str(paths["facts"]), str(paths["contours"])).to_dict()
    publish_datasets(engine, [MANIFEST], [d], [c])
    assert obj.exists() is True and obj.mtime() > 0
    obj.retrieve_object()
    receipt = json.loads(obj.local_path().read_text())
    assert receipt["dataset"] == "zones" and receipt["contour"]["kind"] == "contour"


def test_refresh_publish_stamps_every_table_identically_and_keeps_receipts_stable(provider, engine, parquet_dir):
    """The reason for `refresh`: outputs of one publish job must share one mtime
    while their receipts do not change."""
    names = ["fake_a", "fake_b"]
    receipts = [stage_table(engine, n, str(parquet_dir[n])).to_dict() for n in names]
    publish_tables(engine, receipts, refresh=True)

    def snapshot():
        out = {}
        for n in names:
            obj = _obj(provider, f"published/{n}")
            obj.retrieve_object()
            out[n] = (obj.mtime(), obj.local_path().read_text())
        return out

    first = snapshot()
    assert first["fake_a"][0] == first["fake_b"][0] > 0

    # Nothing changed: every receipt is "current", nothing is swapped, yet all
    # markers are stamped again, together.
    again = [stage_table(engine, n, str(parquet_dir[n])).to_dict() for n in names]
    assert all(r["status"] == "current" for r in again)
    assert publish_tables(engine, again, refresh=True) == []
    second = snapshot()
    assert second["fake_a"][0] == second["fake_b"][0] > first["fake_a"][0]
    assert {n: second[n][1] for n in names} == {n: first[n][1] for n in names}


def test_publish_without_refresh_leaves_current_markers_alone(engine, parquet_dir):
    receipts = [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()]
    publish_tables(engine, receipts)
    before = fetch_marker(engine, "fake_a")
    again = [stage_table(engine, "fake_a", str(parquet_dir["fake_a"])).to_dict()]
    assert publish_tables(engine, again) == []
    assert fetch_marker(engine, "fake_a") == before
