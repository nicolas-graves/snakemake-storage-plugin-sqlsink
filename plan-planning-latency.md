# Planning latency over a slow tunnel: one engine, one snapshot, two round trips

## Status (2026-10-08)

Steps 1 and 2 are implemented (`f6bb742`). Remote dry-run of cgdd-sevs-ecf
`superset_all`: 1 connection and 5 statements instead of 2 and ~23, 5 s wall on
a warm tunnel (25-57 s before); the old and new readers return identical state
on the live catalog. Not measured on a cold tunnel. Step 3 (connect gap) and
step 4 (tunnel) are open.

## Context

`snakemake -n superset_all --configfile superset/remote.yaml` in cgdd-sevs-ecf
(sqlsink pinned at `28f9d4b`, i.e. already including "reuse planning snapshot for
direct checks") spends 25-57 s of wall time for ~5 s of CPU, all in PostgreSQL
round trips through an ssh tunnel to OVH (`127.0.0.1:<port>`). Measured on
2026-10-08 with SQLAlchemy `engine_connect` / `before|after_cursor_execute`
hooks, in order:

| # | Where | What it does | Observed |
|---|-------|--------------|----------|
| 1 | `StorageProvider.__post_init__` (engine A) | handshake + `SELECT 1` | connect 7 s, 11 s until first checkout, query 2.5 s |
| 2 | `sqlsink.smk::_sqlsink_planning_state` (engine B, **its own engine**) | handshake + `read_planning_state` | connect 6 s, 9 s until checkout, then 11 statements at 0.6-2.8 s |
| 3 | provider `direct_state()` (engine A, pooled connection reused) | `read_planning_state` again, same content as #2 | first statement 2.8 s, then 11 statements at 0.05-0.15 s |

Per-statement latency is bimodal: 0.6-2.8 s on a cold tunnel, 0.05 s warm. So the
cost is (number of handshakes) x (cold connect) + (number of statements) x
(cold RTT). Today: 2 handshakes, ~23 statements.

Two corrections to the first read of this:
- Dropping the `SELECT 1` probe saves little: engine A's pooled connection is
  reused by `direct_state()`, so the probe pays the handshake that #3 would pay
  anyway. Not a goal.
- The duplicate read is caused by the consumer (`sqlsink.smk` builds engine B),
  not by the provider. The provider already caches `_direct_state`.

## Goals

1. One handshake and one snapshot per DAG build, shared by the provider's
   `direct_state()` and the workflow's dataset/grants predicates.
2. `read_planning_state` costs a constant, small number of round trips (target 2),
   independent of the number of marker tables.
3. No change in planning decisions (same `PlanningState`), no writes, same
   `on_unreachable` semantics.

## Steps

### 1. Process-level shared snapshot (`planning.py`)

Add `shared_planning_state(engine_or_url, *, roles)` plus `invalidate_shared_planning_state()`,
holding one `PlanningState` per database URL for the process.
- Roles are additive: if a later caller asks for roles the cached snapshot did not
  read, re-read with the union once, never per call.
- `StorageProvider.direct_state()` and `invalidate_planning_state()` delegate to it
  (so a storage write/retrieval still drops it, as in `28f9d4b`).
- `sqlsink.smk::_sqlsink_planning_state` becomes a thin call to it and drops its
  `make_engine`/`dispose`; its `onstart` `cache_clear()` calls
  `invalidate_shared_planning_state()` instead (checkpoint re-planning must still
  re-read).
- Keep the "callers must not cache across staging or publication" contract: the
  shared snapshot is invalidated by the provider's existing write hooks and by
  `onstart`; document it in the module docstring (currently says never cache).
- Open question: sharing across the engine boundary needs the engine from the
  provider. Simplest is URL-keyed engine reuse via a small `get_engine(url)` in
  `engine.py` used by both; confirm that the credentials resolution yields an
  identical URL string in provider and workflow (same `SQLSINK_CREDENTIALS`).

Saves: one handshake (~15 s cold) and 11 statements.

### 2. Collapse the catalog reads into two round trips (`planning.py`)

Today: `SET TRANSACTION`, `SET LOCAL statement_timeout`, `pg_tables`, then one
`SELECT` per marker table (4 + dataset_components), relations, grants = 11.
- Move isolation and timeout out of statements: transaction isolation through
  `execution_options(isolation_level="REPEATABLE READ")` and `read_only`, timeout
  through the connection `options=-c statement_timeout=15s` (psycopg connect arg).
  Check the 15 s timeout semantics are preserved (it was `SET LOCAL`, scoped to
  the transaction).
- Round trip 1: one `UNION ALL` query tagged by kind returning the existing
  `_pipeline_meta_*` tables, relations and grants as rows.
- Round trip 2: one `UNION ALL` over the marker tables that exist, rows as
  `to_jsonb`, tagged by kind (statement built from round trip 1, as the code does
  today with `existing`).
- Fallback if the single-statement form is awkward with the SQLAlchemy Core
  tables: psycopg pipeline mode over the same statements (one network flush).
  Prefer the UNION form; it also works through pgbouncer in transaction mode.

Saves: ~9 statements per read; with step 1, the read happens once.

### 3. Measure the connect gap (investigation, may add nothing)

There is a 9-11 s gap between "connected" and the first `engine_connect` on each
fresh connection. Hypothesis: SQLAlchemy dialect first-connect initialization
(server version, `standard_conforming_strings`, isolation level probes). Verify by
logging at the psycopg level. If confirmed, options: reuse the already
initialized dialect/engine (step 1 already does for the second consumer), or run
planning through a raw psycopg connection and skip SQLAlchemy initialization.
Decide after measuring; do not do this speculatively.

### 4. Tunnel (outside this repo)

Per-statement 0.6-2.8 s on a cold tunnel is the multiplier. Check separately
whether the ssh hop is the problem (keepalive, `ControlMaster`, compression
settings from the existing compression benchmark). Tracked in cgdd-sevs-ecf, not
here.

## Tests

- `tests/test_planning.py`: with a statement counter (SQLAlchemy event, as in the
  session's trace), assert `read_planning_state` issues <= 2 statements (plus
  fixed transaction control) for 0, 1 and all marker tables present, on the
  disposable PostgreSQL suite.
- Equivalence: new reader vs. current reader return equal `PlanningState` over the
  existing freshness fixtures (marker freshness, wrong relation kind, changed
  component update, revoked SELECT).
- `tests/test_storage_plugin.py`: provider `direct_state()` and the shared
  snapshot return the same object until `invalidate_planning_state()`; a role
  requested later triggers exactly one extra read.
- Workflow-level: a counter around a `snakemake -n` of the test workflow showing
  one `read_planning_state` per DAG build.

## Verification in cgdd-sevs-ecf

Re-run the instrumented dry run (sqlalchemy hooks, `launch2.py`, kept in that
session's scratchpad; recreate from the table above) with
`--configfile superset/remote.yaml`. Success: one handshake, <= ~4 statements
total in planning, DAG time on a cold tunnel under ~15 s; planned jobs identical
to a run on the previous pin. Then bump the `dev-manifest.scm` commit and hash.
This is also the real-world trial `plan.md` deferred.

## Order

1 (biggest, simplest, no SQL changes) -> 2 -> 3 (only if the gap is still
dominant) -> bump pin. Step 4 in parallel.
