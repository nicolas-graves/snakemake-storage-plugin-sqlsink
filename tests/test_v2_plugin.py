"""Storage plugin objects over v2 datasets: `dataset/{name}` freshness and
receipt, and `dataset:<name>` entries in a grants file."""

from __future__ import annotations

import json
import logging

import pytest
import yaml

pytest.importorskip("snakemake_storage_plugin_sqlsink")

from snakemake_storage_plugin_sqlsink import (  # noqa: E402
    StorageObject,
    StorageProvider,
    StorageProviderSettings,
)

from fixtures.make_fixtures import TRANSITIONS_VIEW_SQL, make_transitions_fixtures, write_table  # noqa: E402
from sqlsink.manifest import load_manifests  # noqa: E402
from sqlsink.sink import materialize_v2, sources_from_dir  # noqa: E402
from sqlsink.sink_postgres import SqlSink  # noqa: E402

MANIFESTS = {
    "components": [
        {"name": "bridge", "kind": "bridge", "primary_key": ["anchor", "f21"], "priority": "priority"},
    ],
    "datasets": [
        {
            "name": "transitions",
            "components": [
                "bridge",
                {"name": "fact", "kind": "fact", "primary_key": ["obs_id"]},
                {"name": "sector_paths", "kind": "dimension", "primary_key": ["region", "f21", "path"]},
            ],
            "view_sql": TRANSITIONS_VIEW_SQL,
        }
    ],
}


@pytest.fixture
def provider(engine, tmp_path):
    pq = tmp_path / "pq"
    make_transitions_fixtures(pq)
    manifests = tmp_path / "manifests.yaml"
    manifests.write_text(yaml.safe_dump(MANIFESTS))
    p = StorageProvider(
        local_prefix=tmp_path / "local",
        logger=logging.getLogger("test"),
        settings=StorageProviderSettings(dsn="duckdb:///:memory:", parquet_dir=str(pq), manifests=str(manifests)),
    )
    p.engine = engine
    return p


def _obj(provider, query):
    return StorageObject(query=query, keep_local=True, retrieve=True, provider=provider)


def test_dataset_object_tracks_source_files_components_and_receipt(provider, engine):
    obj = _obj(provider, "dataset/transitions")
    assert obj.exists() is False
    manifest = provider.manifests()["transitions"]
    materialize_v2(manifest, sources_from_dir(manifest, provider.settings.parquet_dir), SqlSink(engine))
    assert obj.exists() is True and obj.mtime() > 0

    obj.retrieve_object()
    receipt = json.loads(obj.local_path().read_text())
    assert receipt["dataset"] == "transitions"
    assert sorted(receipt["components"]) == ["bridge", "fact", "sector_paths"]
    assert receipt["components"]["bridge"]["kind"] == "component"
    assert "published_at" not in receipt["components"]["bridge"]

    # A changed component source makes the published dataset stale.
    write_table(
        provider.settings.parquet_dir + "/bridge.parquet",
        "anchor VARCHAR, f21 VARCHAR, priority INTEGER",
        [("A", "F2", 9)],
    )
    assert obj.exists() is False


def test_grants_file_can_name_a_dataset(provider, engine, tmp_path):
    grants = tmp_path / "grants.yaml"
    grants.write_text("runtime:\n  - dataset:transitions\n  - other_table\n")
    provider.settings.grants_file = str(grants)
    assert provider.grant_relations("runtime") == [
        "analytics_storage.bridge",
        "analytics_storage.fact",
        "analytics_storage.sector_paths",
        "other_table",
        "public.transitions",
    ]
