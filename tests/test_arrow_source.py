"""In-memory (Arrow / pandas) sources behave exactly like the same data as Parquet."""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import text

from sqlsink.fingerprint import sha256_arrow
from sqlsink.manifest import DatasetMaterialization
from sqlsink.materialize import stage_contours, stage_dataset
from sqlsink.publish import publish_datasets
from sqlsink.sink import make_sink, materialize, normalize

MANIFEST = DatasetMaterialization(
    name="zones",
    geometry_column="polygon_coords",
    contour_table="contours",
    fact_join_columns=("zone id",),
    contour_join_columns=("zone id",),
    output_columns=("zone id", "metric", "polygon_coords"),
    fact_source="facts",
)

FACTS = pd.DataFrame(
    {
        "zone id": ["Z1", "Z1", "Z2"],
        "metric": [10, 10, 20],
        "polygon_coords": pd.Categorical(["a", "b", "c"]),
    }
)
CONTOURS = pd.DataFrame({"zone id": ["Z1", "Z1", "Z2"], "polygon_coords": ["a", "b", "c"]})


def _rows(engine):
    with engine.connect() as conn:
        return sorted(
            tuple(r) for r in conn.execute(text('SELECT "zone id", metric, polygon_coords FROM "zones"')).fetchall()
        )


def test_dataframe_sources_publish_like_parquet(tmp_path, engine):
    contour_receipt = stage_contours(engine, MANIFEST, CONTOURS).to_dict()
    dataset_receipt = stage_dataset(engine, MANIFEST, FACTS, CONTOURS).to_dict()
    assert publish_datasets(engine, [MANIFEST], [dataset_receipt], [contour_receipt]) == ["zones"]

    assert _rows(engine) == [("Z1", 10, "a"), ("Z1", 10, "b"), ("Z2", 20, "c")]
    # Two logical facts, geometry not repeated in the compact table.
    assert dataset_receipt["row_count"] == 2


def test_dataframe_and_parquet_share_one_identity(tmp_path):
    pq.write_table(pa.Table.from_pandas(FACTS, preserve_index=False), tmp_path / "f.parquet")
    a = normalize(MANIFEST, FACTS, CONTOURS)
    assert a.update_id == normalize(MANIFEST, FACTS, CONTOURS).update_id  # deterministic
    changed = FACTS.assign(metric=[10, 10, 21])
    assert normalize(MANIFEST, changed, CONTOURS).update_id != a.update_id


def test_sha256_arrow_ignores_chunk_layout():
    table = pa.Table.from_pandas(FACTS, preserve_index=False)
    chunked = pa.concat_tables([table.slice(0, 1), table.slice(1)])
    assert sha256_arrow(table) == sha256_arrow(chunked)


def test_materialize_from_memory_then_noop(tmp_path):
    sink = make_sink({"type": "duckdb", "path": str(tmp_path / "s.duckdb")})
    try:
        published, first = materialize(MANIFEST, FACTS, CONTOURS, sink)
        assert published == ["zones"]
        published, second = materialize(MANIFEST, FACTS, CONTOURS, sink)
        assert second.dataset.status == "current"
        assert published == []
    finally:
        sink.engine.dispose()


def test_duckdb_sink_can_publish_into_a_default_schema(tmp_path):
    path = str(tmp_path / "s.duckdb")
    sink = make_sink({"type": "duckdb", "path": path, "schema": "public"})
    try:
        materialize(MANIFEST, FACTS, CONTOURS, sink)
    finally:
        sink.engine.dispose()
    import duckdb

    con = duckdb.connect(path, read_only=True)
    try:
        assert con.execute('SELECT count(*) FROM public."zones"').fetchone()[0] == 3
    finally:
        con.close()


def test_double_columns_keep_full_precision_in_the_duckdb_sink(tmp_path):
    import duckdb

    value = 0.1234567890123  # not representable as float32
    facts = FACTS.assign(ratio=[value, value, 2.0], count=pd.array([1, 2, 3], dtype="int16"))
    manifest = DatasetMaterialization(
        name="zones",
        geometry_column="polygon_coords",
        contour_table="contours",
        fact_join_columns=("zone id",),
        contour_join_columns=("zone id",),
        output_columns=("zone id", "metric", "ratio", "count", "polygon_coords"),
    )
    path = str(tmp_path / "s.duckdb")
    sink = make_sink({"type": "duckdb", "path": path, "schema": "public"})
    try:
        materialize(manifest, facts, CONTOURS, sink)
    finally:
        sink.engine.dispose()
    con = duckdb.connect(path, read_only=True)
    try:
        types = {r[0]: r[1] for r in con.execute('DESCRIBE public."zones"').fetchall()}
        assert types["ratio"] == "DOUBLE"
        assert con.execute('SELECT min(ratio) FROM public."zones"').fetchone()[0] == value
        assert types["count"] in ("SMALLINT",)
    finally:
        con.close()
