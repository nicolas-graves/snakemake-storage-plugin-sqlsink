"""Per-table incremental publishing of Parquet outputs into the SQL database.

Replaces the old monolithic "load everything" rule. `stage_table`'s output
is backed by the private `sql-incremental` storage plugin
(`workflow/sql_storage_plugin/`): Snakemake itself decides, via the plugin's
`exists()`/`mtime()`, whether a table's Parquet changed since it was last
published, and **skips the rule entirely** for tables that are already
current -- it no longer runs just to write a no-op receipt.

`publish_tables` still depends on every table's receipt and publishes all
staged tables in a single all-or-nothing transaction; the storage plugin
has no notion of a multi-object transaction, so this final rule is
unchanged from the pre-plugin design. For a table Snakemake decided not to
re-stage this run, the plugin's `retrieve_object()` reconstructs an
equivalent "status": "current" receipt on demand (from the marker table),
so `publish_tables` sees a receipt for every table regardless of whether
`stage_table` actually ran for it this session.

Downstream rules (Superset registration, equivalence audit, smoke tests)
should depend on `results/.db_published`, not on any individual table's
receipt.

This module is written against SQLAlchemy Core throughout (see
`workflow/scripts/sql_incremental/`), so `config["db"]["dsn"]` can point at
any SQLAlchemy-supported engine, not just PostgreSQL.
"""

TABLES = config["tables"]

# Logical datasets with a normalized PostgreSQL materialization (compact
# fact table + shared contour table + public compatibility view), per
# docs/postgresql-geometry-normalization.md. Empty until Phase 1
# characterization has proven the multiset-equivalence of a manifest's
# derived compact facts against the real, already-joined Parquet -- see
# config/config.yaml for the manifest shape and the pending TODOs.
DATASETS = config.get("datasets", [])
DATASETS_BY_NAME = {d["name"]: d for d in DATASETS}


storage sql_incremental:
    provider="sql-incremental",
    dsn=config["db"]["dsn"],
    parquet_dir="results/parquet",


# No separate "setup_db_meta" rule: the marker table has to exist before
# Snakemake can even build the DAG (it calls exists()/mtime() on every
# storage object up front), so the storage plugin's own StorageProvider
# creates it eagerly on construction (see workflow/sql_storage_plugin/).


rule stage_table:
    input:
        parquet="results/parquet/{table}.parquet",
    output:
        receipt=storage.sql_incremental("{table}"),
    params:
        dsn=config["db"]["dsn"],
        table="{table}",
    script:
        "../scripts/stage_table.py"


rule publish_tables:
    input:
        receipts=storage.sql_incremental(expand("{table}", table=TABLES)),
    output:
        touch("results/.tables_published"),
    params:
        dsn=config["db"]["dsn"],
    script:
        "../scripts/publish_tables.py"


# Datasets go through one materialization API with two SQL sinks
# (`sql_incremental.sink`): "postgres" (production) and "duckdb" (a local
# database file, `duckdb_sink_path`); both hold a compact table + shared
# contour table + compatibility view. `sinks:` in the config selects which
# ones run; default is postgres only. The rules below are identical for
# every sink -- only the sink spec in `params` differs. Parquet is not a
# sink: `export_dataset` (opt-in, not part of any default target) writes the
# published version of a dataset to Parquet.
#
# DuckDB allows one writer process per file: run with `--resources
# duckdb_writer=1` so staging jobs on a DuckDB sink do not overlap.
#
# Dataset staging is not yet storage-plugin-backed (unlike stage_table
# above): a dataset's freshness depends on two Parquets plus a manifest
# hash, not one file, and Snakemake's storage-object interface has no
# built-in notion of that composite identity. `sink.stage` computes a
# correct update_id from all three, so this rule simply always runs and lets
# it decide "current" vs "staged" -- the same no-op-but-cheap fallback
# stage_table had before the storage plugin existed. Making the freshness
# check itself skip the rule (like the per-table plugin does) is a
# reasonable follow-up, not required for correctness.
SINK_SPECS = {
    "postgres": {"type": "postgres", "dsn": config["db"]["dsn"]},
    "duckdb": {"type": "duckdb", "path": config.get("duckdb_sink_path", "results/sink.duckdb")},
}
SINKS = config.get("sinks", ["postgres"])
_unknown_sinks = set(SINKS) - set(SINK_SPECS)
if _unknown_sinks:
    raise ValueError(f"unknown sink(s) in config `sinks`: {sorted(_unknown_sinks)}")


rule stage_dataset:
    input:
        fact_parquet=lambda wc: f"results/parquet/{DATASETS_BY_NAME[wc.dataset]['fact_source']}.parquet",
        contour_parquet=lambda wc: f"results/parquet/{DATASETS_BY_NAME[wc.dataset]['contour_source']}.parquet",
    output:
        dataset_receipt="results/dataset_receipts/{sink}/{dataset}.json",
        contour_receipt="results/contour_receipts/{sink}/{dataset}.json",
    wildcard_constraints:
        sink="|".join(SINK_SPECS),
    params:
        sink=lambda wc: SINK_SPECS[wc.sink],
        manifest=lambda wc: DATASETS_BY_NAME[wc.dataset],
    threads: 2
    resources:
        duckdb_writer=1,
    script:
        "../scripts/stage_dataset.py"


rule publish_datasets:
    input:
        dataset_receipts=lambda wc: expand(
            "results/dataset_receipts/{sink}/{dataset}.json", sink=wc.sink, dataset=DATASETS_BY_NAME
        ),
        contour_receipts=lambda wc: expand(
            "results/contour_receipts/{sink}/{dataset}.json", sink=wc.sink, dataset=DATASETS_BY_NAME
        ),
    output:
        touch("results/.datasets_published.{sink}"),
    wildcard_constraints:
        sink="|".join(SINK_SPECS),
    params:
        sink=lambda wc: SINK_SPECS[wc.sink],
        manifests=DATASETS,
    script:
        "../scripts/publish_datasets.py"


# Opt-in Parquet export of a published dataset, e.g.
#   snakemake results/exports/postgres/<dataset>.parquet
# It reads the published relation of the sink, never the source Parquet, and
# depends on the sink's publish so it can only run on a published version.
rule export_dataset:
    input:
        published="results/.datasets_published.{sink}",
    output:
        parquet="results/exports/{sink}/{dataset}.parquet",
    wildcard_constraints:
        sink="|".join(SINK_SPECS),
    params:
        sink=lambda wc: SINK_SPECS[wc.sink],
        manifest=lambda wc: DATASETS_BY_NAME[wc.dataset],
    threads: 2
    script:
        "../scripts/export_dataset.py"
