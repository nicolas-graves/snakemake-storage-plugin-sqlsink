# Plan: authoritative pruning in sqlsink

## Summary

Move generic reconciliation and pruning into sqlsink. Pruning remains explicit and
report-first; it must never run implicitly during partial publication.

## Implementation

- Add `sink.reconcile(desired, authoritative=False)`:
  - inventory plugin-managed public relations, private components and marker rows;
  - return desired, obsolete and missing objects without mutation;
  - reject pruning unless `authoritative=True`.
- Add `report.prune(execute=False)`:
  - default to dry-run;
  - with `execute=True`, acquire the publication advisory lock and transactionally
    remove obsolete loader-owned objects and markers;
  - restrict deletion to configured schemas and preserve shared components still
    referenced by desired datasets.
- Keep `publish()` unchanged: it continues replacing same-name generations only.
- Replace the downstream repository's pruning implementation with a thin wrapper
  that loads the complete manifest and plain-table registry, calls
  `reconcile(..., authoritative=True)`, and exposes `--execute`.
- Refuse pruning from partial publication scopes.
- Document the new plugin API and the distinction between publication and
  authoritative reconciliation.

## Tests

- Plugin tests: dry-run, explicit execution, ownership/schema boundaries,
  transactional rollback, stale marker-only and catalog-only objects, shared
  components, and partial-scope rejection.
- Downstream tests: complete desired-state construction and wrapper argument handling.
- Run both plugin and focused downstream suites.
- Run the downstream command report-only against production and require an empty report.

## Assumptions

- PostgreSQL pruning is implemented first; DuckDB reconciliation is out of scope.
- Automatic means a reusable plugin operation, not an unconditional side effect of
  publication.
- The existing production cleanup is complete, so migration requires no further
  database deletion.
