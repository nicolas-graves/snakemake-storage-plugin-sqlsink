# Binary PostgreSQL staging from DuckDB

## Goal

Speed up staging of Parquet-backed components into remote PostgreSQL, especially
`fact_transitions_entrantes_fap2021`, without changing dataset contents, receipts,
locking, or publish behavior.

The current path in `workflow/scripts/sqlsink/sink_v2.py` reads a Parquet source
through DuckDB, validates its key, creates a constrained staging table, and calls
`engine.bulk_load_streaming`. On PostgreSQL that function converts DuckDB rows to
CSV text and sends them through psycopg `COPY FROM STDIN`. The 12 MB compressed
fact Parquet generated a much larger stream through the workstation's SSH tunnel;
the staging process spent most of its time waiting on the socket.

DuckDB's PostgreSQL extension supports `COPY attached_pg.table FROM 'file.parquet'`
using PostgreSQL binary wire encoding. It also supports writing
`FORMAT postgres_binary` to a file. Compare both with the current CSV path before
choosing the implementation: binary format avoids CSV conversion, but neither
binary path guarantees a stream as small as compressed Parquet.

## Implementation

1. **Establish a reproducible baseline.** Confirm that the PostgreSQL extension
   can be installed and loaded with the pinned DuckDB 1.5.3 environment; make
   it available through the development environment if runtime downloads are
   unreliable. Use the pinned version and the
   same analytics tunnel to stage `fact_transitions_entrantes_fap2021` into a
   disposable staging table. Record wall time, CPU time, rows, bytes sent on the
   tunnel, and output equivalence. Repeat enough times to separate network
   variation from the format change. Do not benchmark against published tables.

2. **Prototype DuckDB's writable PostgreSQL attachment.** Extend the existing
   temporary-secret connection helper in `sink_postgres.py` to allow a writable
   attachment while retaining libpq TLS options and keeping credentials out of
   SQL text and logs. After `check_component_source` and `_constrained_table`,
   load the already-created staging table with DuckDB's Parquet-to-PostgreSQL
   `COPY`. Check that its schema, column order, NULLs, Unicode, numeric and date
   types, and primary-key enforcement match the current loader. Keep the
   component advisory lock around the operation.

3. **Resolve transaction ownership before replacing the loader.** Today table
   replacement, COPY, and `staged_component_updates` are in one SQLAlchemy
   transaction. A writable DuckDB attachment opens another PostgreSQL
   connection; test rollback on a failed COPY, process termination, and a
   concurrent stage of the same component. If the direct attachment cannot
   preserve the required atomic marker/table state, use DuckDB's
   `FORMAT postgres_binary` output and stream it to psycopg `COPY ... FORMAT
   binary` inside the existing transaction. A bounded temporary binary file is
   acceptable for this path; remove it on success and failure. Do not publish
   a staging table without its matching marker.

4. **Change the selected PostgreSQL loader in place.** Keep the DuckDB sink's
   current Arrow path. Cover the PostgreSQL call sites in both v1
   facts/contours and v2 components; some v1 inputs are query results rather
   than plain Parquet files, so they may need a DuckDB binary export rather
   than direct `COPY FROM` a source file. Preserve the
   current published/staged marker checks and idempotent retries. Avoid a
   permanent CSV compatibility path or a runtime format switch.

5. **Verify and pin.** Add PostgreSQL integration tests for non-ASCII text,
   empty strings versus NULL, quotes/newlines, booleans, decimals, timestamps,
   duplicate and NULL keys, failed loads, interrupted/retried staging, and
   concurrent stages. Compare row counts and values with the current loader
   before removing it. Update the pipeline's `dev-manifest.scm` sqlsink source
   pin to the tested commit, then rerun the representative remote stage and
   record the same timing and network measurements as the baseline.

## Acceptance criteria

- The staged relation and receipt match the current implementation, and no
  partially loaded table can be treated as current after a failure.
- Existing publish and idempotency tests pass; the local DuckDB sink is unchanged.
- The representative remote stage shows a material, repeatable improvement in
  elapsed time or tunnel bytes. If binary transfer does not improve the measured
  bottleneck, stop before replacing the loader and investigate staging on the
  remote host with compressed Parquet instead.

Reference: [DuckDB PostgreSQL extension](https://duckdb.org/docs/current/core_extensions/postgres/overview).

## Local experiment, 2026-10-08

The pinned DuckDB 1.5.3 wheel successfully installed and loaded its PostgreSQL
extension, and `FORMAT postgres_binary` produced a valid COPY stream. Against a
disposable local PostgreSQL 17 instance, binary COPY preserved Unicode,
embedded quotes/newlines, empty strings versus NULL, booleans, decimals, dates,
timestamps and column order. Duplicate and NULL primary keys rolled back the
staging table and marker transaction. A writable DuckDB attachment could copy
into a committed table, but could not see a table created in the active
SQLAlchemy staging transaction; it therefore cannot replace that transaction's
loader without changing stage/marker atomicity.

For a synthetic 200,000-row query (integer, short Unicode text, boolean and
decimal), three local runs measured CSV at 0.633/0.617/0.604 seconds and
8,366,780 payload bytes, versus binary at 0.351/0.088/0.090 seconds and
9,688,711 payload bytes. The first binary run includes extension/export warmup.
These are client payload sizes, not SSH tunnel byte counts. Binary was faster
on loopback but sent 15.8% more bytes for this shape. Production staging
therefore stays on CSV pending the representative same-tunnel benchmark.

## Remote benchmark, 2026-10-08

The representative `fact_transitions_entrantes_fap2021.parquet` (995,075 rows,
about 12 MB compressed) was loaded through the analytics SSH tunnel into
disposable PostgreSQL tables. Each trial ran in a transaction that was rolled
back. Row counts and row checksums matched across all four trials.

| Order | Format | COPY elapsed | Client CPU | COPY payload |
| --- | --- | ---: | ---: | ---: |
| 1 | CSV | 439.718 s | 14.654 s | 242,179,515 bytes |
| 2 | Binary | 419.222 s | 1.719 s | 243,917,630 bytes |
| 3 | Binary | 406.858 s | 2.359 s | 243,917,630 bytes |
| 4 | CSV | 276.626 s | 26.976 s | 242,179,515 bytes |

Payload counts are bytes written by the client to PostgreSQL COPY, not SSH
tunnel byte counters. The large change between the two CSV timings shows that
elapsed time varied substantially during this benchmark. Binary lowered client
CPU use but did not show a repeatable elapsed-time improvement and sent 0.7%
more payload. The acceptance gate was not met, so the experimental loader and
Guix extension changes were reverted. The released SQLsink 0.1.1 loader still
uses CSV; no binary release or source pin was made.

Guix's Python DuckDB recipe fetches the v1.5.3 Python and DuckDB source tags,
but its built library reports `v0.0.1` because the source archive lacks Git
tag metadata. A first packaging override did not change the embedded version.
A corrected CMake argument was not validated after the remote benchmark failed
the acceptance gate.

## SSH compression benchmark, 2026-10-08

The same 995,075-row Parquet source was loaded through four fresh analytics SSH
tunnels in off/on/on/off order. Each tunnel had a unique control socket and an
explicit SSH `Compression` setting. Each COPY used the current CSV loader into
a disposable PostgreSQL table inside a rolled-back transaction. Row counts and
row checksums matched in all four trials.

| Order | SSH compression | COPY elapsed | Client CPU | SSH bytes acknowledged |
| --- | --- | ---: | ---: | ---: |
| 1 | Off | 257.773 s | 27.901 s | 242,790,019 |
| 2 | On | 211.902 s | 27.740 s | 242,907,619 |
| 3 | On | 196.283 s | 12.257 s | 242,907,647 |
| 4 | Off | 198.426 s | 9.367 s | 242,789,647 |

The byte counts are TCP `bytes_acked` on the SSH connection to the VPS,
sampled with `ss -tinp`; they include SSH and TCP framing but exclude
retransmissions. The CSV COPY payload was 242,179,515 bytes in each run.
Compression did not reduce wire traffic (the compressed runs sent about 0.05%
more acknowledged bytes). The final uncompressed run matched the compressed
timings, so the earlier elapsed-time change is consistent with tunnel or server
variation rather than a repeatable compression benefit. PostgreSQL uses TLS
inside this tunnel, which likely leaves the SSH layer encrypted data with
little compressible structure. Keep SSH compression off and investigate moving
the 12 MB Parquet file to a runner near PostgreSQL instead.
