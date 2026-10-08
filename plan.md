# Batched read-only planning checks

## Implemented

PostgreSQL planning reads staged and published markers, relation kinds, component
update IDs, and requested SELECT ACLs in one read-only, repeatable-read
transaction. Storage inventory shares that snapshot for its cache pass. Direct
`exists()` and `mtime()` checks read fresh state. The consuming workflow checks
all dataset publications from one snapshot per DAG build. Missing marker tables
are treated as missing state without creating them.

## Local verification, 2026-10-08

- Guix local suite: 179 passed, 39 skipped.
- Disposable PostgreSQL focused suite: 55 passed, covering marker and relation
  freshness, wrong relation kinds, changed component updates, and revoked
  SELECT grants.
- Snakemake end-to-end suite with DuckDB 1.5.3 wheel under Guix Python:
  8 passed, including changed-input republishing and Parquet export.
- The Guix-built DuckDB reports version v0.0.1 and cannot download its
  PostgreSQL scanner extension; the wheel was used for scanner-dependent tests.

## Real-world trial

The original analytics tunnel, profile and representative Parquet are not
available in this workspace. Run the remote `superset_all` dry run with the
same data and tunnel, compare planned jobs and receipt decisions with the
previous run, and record database statement count and elapsed DAG time. This
remote comparison was explicitly deferred for the local completion.
