"""Security hardening: names that come from data (Parquet column names) are
always quoted, names that come from config are validated, the DuckDB postgres
scanner connects with the DSN's TLS settings and never echoes its password, and
values read back from the catalog are not trusted as SQL. Runs on DuckDB, and
on PostgreSQL with SNAKEMAKE_SQL_TEST_PG_DSN."""

from __future__ import annotations

import json
import os

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url

from sqlsink import keep_old as ko
from sqlsink.manifest import ManifestError, load_manifest, load_manifests
from sqlsink.publish import publish_tables
from sqlsink.sink import materialize_v2
from sqlsink.sink_postgres import SqlSink, pg_attach, pg_conninfo
from sqlsink.stage import _read_parquet_schema_and_stats, stage_table

PG_DSN = os.environ.get("SNAKEMAKE_SQL_TEST_PG_DSN")
needs_pg = pytest.mark.skipif(not PG_DSN, reason="needs SNAKEMAKE_SQL_TEST_PG_DSN (a disposable PostgreSQL)")

QUOTED = 'say "hi"'


def _injection(out) -> str:
    """A column name that, spliced unquoted into the null-count query, ends it
    and runs a `COPY ... TO` writing `out`."""
    return f"x\" IS NULL) AS a FROM (SELECT 1 AS ok, 1 AS x); COPY (SELECT 42) TO '{out}'; SELECT 1, 1 --"


def test_parquet_column_names_cannot_inject_sql_into_staging(tmp_path):
    out = tmp_path / "written_by_injection.csv"
    evil = _injection(out)
    path = tmp_path / "evil.parquet"
    pq.write_table(pa.table({"ok": [1, 2], evil: [1, None]}), path)

    columns, row_count, null_counts = _read_parquet_schema_and_stats(str(path))

    assert not out.exists()
    assert [c[0] for c in columns] == ["ok", evil]
    assert row_count == 2
    assert null_counts == {"ok": 0, evil: 1}


def test_stage_and_publish_a_table_whose_column_name_has_a_quote(engine, tmp_path):
    path = tmp_path / "quoted.parquet"
    pq.write_table(pa.table({"id": [1, 2], QUOTED: ["a", None]}), path)

    receipt = stage_table(engine, "quoted", str(path))
    assert receipt.null_counts == {"id": 0, QUOTED: 1}
    assert publish_tables(engine, [receipt.to_dict()]) == ["quoted"]

    assert [c["name"] for c in inspect(engine).get_columns("quoted")] == ["id", QUOTED]
    with engine.connect() as conn:
        assert sorted(conn.execute(text('SELECT id, "say ""hi""" FROM "quoted"')).all(), key=lambda r: r[0]) == [
            (1, "a"),
            (2, None),
        ]


def test_v2_component_whose_column_name_has_a_quote(engine, tmp_path):
    path = tmp_path / "fact.parquet"
    pq.write_table(pa.table({"id": [1, 2], QUOTED: ["a", "b"]}), path)
    manifest = load_manifest(
        {
            "name": "quoted_ds",
            "components": [{"name": "fact", "kind": "fact", "primary_key": ["id"]}],
            "view_sql": "SELECT * FROM {fact}",
        }
    )

    published, _ = materialize_v2(manifest, {"fact": str(path)}, SqlSink(engine))

    assert published == ["quoted_ds"]
    with engine.connect() as conn:
        got = conn.execute(text('SELECT id, "say ""hi""" FROM "quoted_ds" ORDER BY id')).all()
    assert [tuple(r) for r in got] == [(1, "a"), (2, "b")]


# -- DuckDB postgres scanner ---------------------------------------------------


def test_pg_conninfo_forwards_the_dsn_tls_options():
    url = make_url("postgresql+psycopg://u:p'w\\x@db.example:5433/an?sslmode=verify-full&sslrootcert=/etc/ca.pem")
    conninfo = pg_conninfo(url)
    assert "sslmode='verify-full'" in conninfo
    assert "sslrootcert='/etc/ca.pem'" in conninfo
    assert "password='p\\'w\\\\x'" in conninfo
    assert "host='db.example'" in conninfo and "port='5433'" in conninfo


def test_pg_conninfo_flattens_repeated_query_keys():
    # libpq takes a comma-separated host list for multi-host DSNs.
    url = make_url("postgresql+psycopg://u@/an?host=h1&host=h2")
    assert "host='h1,h2'" in pg_conninfo(url)


@needs_pg
def test_pg_attach_error_does_not_leak_the_password():
    url = make_url(PG_DSN).set(password="s3cr3t-not-the-password")
    con = duckdb.connect()
    with pytest.raises(duckdb.Error) as raised:
        pg_attach(con, url, alias="leak")
    err = raised.value
    chain = [err]
    while chain[-1].__cause__ or chain[-1].__context__:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert all("s3cr3t-not-the-password" not in str(e) for e in chain)
    assert "password='***'" in str(err)


@needs_pg
def test_pg_attach_honours_sslmode():
    # The disposable test server has no TLS: a DSN that requires it must fail
    # through the scanner too, instead of silently connecting in clear text.
    url = make_url(PG_DSN).update_query_dict({"sslmode": "require"})
    con = duckdb.connect()
    with pytest.raises(duckdb.Error, match="SSL"):
        pg_attach(con, url, alias="tls")


# -- names from config -----------------------------------------------------------


def test_storage_query_with_trailing_newline_is_rejected():
    plugin = pytest.importorskip("snakemake_storage_plugin_sqlsink")
    ok = plugin.StorageProvider.is_valid_query
    assert ok("t").valid and ok("published/t").valid
    assert not ok("t\n").valid
    assert not ok("published/t\n").valid
    assert not ok("grants/r\n").valid


def _v1(**overrides):
    spec = {
        "name": "ds",
        "fact_source": "ds",
        "contour_source": "zones",
        "contour_table": "zones",
        "geometry_column": "geom",
        "fact_join_columns": ["Code ZE"],
        "contour_join_columns": ["Code ZE"],
        "output_columns": ["Code ZE", "Zone d'emploi", "Famille (FAP 2021)", "geom"],
    }
    spec.update(overrides)
    return spec


def _v2(**overrides):
    spec = {
        "name": "ds",
        "components": [{"name": "fact", "kind": "fact", "primary_key": ["id"]}],
        "view_sql": "SELECT * FROM {fact}",
    }
    spec.update(overrides)
    return spec


def test_manifest_column_names_stay_free_form():
    # Columns are always quoted, never validated: real ones carry spaces,
    # apostrophes, accents and non-breaking spaces.
    load_manifest(_v1())


@pytest.mark.parametrize(
    "spec",
    [
        _v1(name='ds"; DROP TABLE x; --'),
        _v1(name="ds\n"),
        _v1(contour_table="zones; x"),
        _v1(compact_schema='s"x'),
        _v2(name="a b"),
        _v2(storage_schema="s\n"),
        _v2(components=[{"name": "fact", "kind": "fact", "primary_key": ["id"], "schema": 's"x'}]),
        _v2(
            components=[
                {"name": "fact", "kind": "fact", "primary_key": ["id"]},
                {"name": "dim\n", "kind": "dimension", "primary_key": ["id"]},
            ]
        ),
        _v2(components=[{"name": "fact", "kind": "fact", "primary_key": ["id"], "source_table": "t;x"}]),
    ],
)
def test_manifest_rejects_unsafe_relation_and_schema_names(spec):
    with pytest.raises((ValueError, ManifestError)):
        load_manifest(spec)


def test_the_shipped_config_passes_name_validation():
    import yaml
    from pathlib import Path

    config = yaml.safe_load((Path(__file__).parent.parent / "config" / "config.yaml").read_text())
    assert load_manifests({"datasets": config["datasets"]})


# -- values read back from the catalog --------------------------------------------


class _RecordingConn:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, statement, *args, **kwargs):
        self.statements.append(str(statement))


def test_restore_grants_refuses_a_privilege_that_is_not_a_privilege():
    conn = _RecordingConn()
    comment = ko.TAG + json.dumps({"kind": "table", "grants": [["reader", "SELECT ON x TO y; DROP TABLE z; --"]]})
    with pytest.raises(ValueError, match="privilege"):
        ko._restore_grants(conn, '"public"."t"', comment)
    assert conn.statements == []


def test_restore_grants_replays_known_privileges():
    conn = _RecordingConn()
    comment = ko.TAG + json.dumps({"kind": "table", "grants": [["reader", "SELECT"], ["PUBLIC", "select"]]})
    ko._restore_grants(conn, '"public"."t"', comment)
    assert conn.statements == [
        'GRANT SELECT ON TABLE "public"."t" TO "reader"',
        'GRANT SELECT ON TABLE "public"."t" TO PUBLIC',
    ]
