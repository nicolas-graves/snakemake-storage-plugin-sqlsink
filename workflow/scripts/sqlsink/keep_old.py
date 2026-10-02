"""Optional rollback window for the publish swap: `keep_old`.

By default a publish drops what it replaces right after the commit
(`<name>__old__<ts>`, best effort). With `keep_old` (argument of
`publish_tables` / `publish_datasets`, or `SQLSINK_KEEP_OLD=1`) every replaced
relation is instead renamed, inside the publish transaction, to the well-defined
name `__old__<name>` in the SAME schema:

    flat table `t`           -> table `__old__t`     (then `t` is the new view)
    plain table `t`          -> table `__old__t`
    component `dim_region`   -> `analytics_storage.__old__dim_region`
    view / materialized view -> `__old__<name>`      (PostgreSQL only; see below)

Nothing is dropped, so the previous state stays readable until `cleanup_kept_old`
removes it. `rollback_kept_old` swaps every kept relation back in one transaction.

Rules that make this well defined:

* One generation. A publish that would replace a relation whose `__old__` name is
  already taken raises `KeptOldExists` (before changing anything): verify, then
  clean up, then publish again. Cleanup and rollback act on ALL kept relations,
  so a kept relation left over from an older publish would otherwise be mixed
  into a rollback of a newer one.
* Grants never widen. A kept relation is unreadable by anyone but its owner: the
  privileges it carried (typically SELECT for the runtime role) are revoked when
  it is set aside and recorded, together with a tag, in the relation's comment
  (`sqlsink:keep_old {...}`), so `rollback_kept_old` can give them back to the
  relation it puts back into service. Only relations carrying that tag are ever
  listed, rolled back or dropped by this module.
* Views follow objects, not names, on PostgreSQL: a kept old view keeps reading
  the kept old components it was built over, so the whole previous state is
  consistent. DuckDB resolves names at query time, so there a replaced view is
  simply dropped and only tables are kept.
* Markers are not rolled back: `rollback_kept_old` deletes the markers of the
  relations it restores (their content is no longer what the markers vouch for),
  so the next run re-stages and republishes them instead of trusting them.
  Relations that the publish created new (nothing was replaced) are left alone.
* Everything runs in one transaction, under the publish advisory lock and the
  publish `lock_timeout`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy import delete, inspect, text

from . import metadata as meta_mod
from .engine import advisory_lock, apply_lock_timeout
from .sqlident import quote_ident as _quote, quote_literal as _literal

log = logging.getLogger(__name__)

OLD_PREFIX = "__old__"
TMP_PREFIX = "__tmp__"
TAG = "sqlsink:keep_old "
PG_MAX_IDENTIFIER = 63

_KEYWORD = {"table": "TABLE", "view": "VIEW", "matview": "MATERIALIZED VIEW"}
# Privileges a relation's ACL can hold (`aclexplode`), i.e. every value
# `_grants` can record. Anything else read back from a comment is refused.
_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER", "MAINTAIN"})


class KeptOldExists(RuntimeError):
    """A relation to be set aside already has a kept `__old__` version."""


@dataclass(frozen=True)
class KeptRelation:
    schema: str
    name: str  # the name without the `__old__` prefix
    kind: str  # "table" | "view" | "matview"

    @property
    def old_name(self) -> str:
        return old_name(self.name)

    def ref(self) -> str:
        return _ref(self.old_name, self.schema)


def old_name(name: str) -> str:
    return f"{OLD_PREFIX}{name}"


def _ref(name: str, schema: str | None) -> str:
    return f"{_quote(schema)}.{_quote(name)}" if schema else _quote(name)


def kind_of(conn, name: str, schema: str | None) -> str | None:
    """"table", "view", "matview" or None, in `schema` (None: the default one)."""
    insp = inspect(conn)
    if name in insp.get_view_names(schema=schema):
        return "view"
    if conn.engine.dialect.name == "postgresql" and name in insp.get_materialized_view_names(schema=schema):
        return "matview"
    if insp.has_table(name, schema=schema):
        return "table"
    return None


def _grants(conn, name: str, schema: str | None) -> list[list[str]]:
    """[[grantee, privilege], ...] the relation carries besides its owner's own
    (`grantee` is "PUBLIC" for a grant to everyone). PostgreSQL only."""
    rows = conn.execute(
        text(
            "SELECT COALESCE(r.rolname, 'PUBLIC'), a.privilege_type "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "CROSS JOIN LATERAL aclexplode(c.relacl) a LEFT JOIN pg_roles r ON r.oid = a.grantee "
            "WHERE n.nspname = COALESCE(:schema, current_schema()) AND c.relname = :name AND a.grantee <> c.relowner "
            "ORDER BY 1, 2"
        ),
        {"schema": schema, "name": name},
    ).all()
    return [[r[0], r[1]] for r in rows]


def _comment(conn, name: str, schema: str | None, kind: str) -> str | None:
    if conn.engine.dialect.name == "postgresql":
        return conn.execute(
            text(
                "SELECT obj_description(c.oid, 'pg_class') FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = COALESCE(:schema, current_schema()) AND c.relname = :name"
            ),
            {"schema": schema, "name": name},
        ).scalar()
    view = "duckdb_views()" if kind == "view" else "duckdb_tables()"
    column = "view_name" if kind == "view" else "table_name"
    return conn.execute(
        text(
            f"SELECT comment FROM {view} WHERE {column} = :name AND schema_name = COALESCE(:schema, current_schema()) "
            "AND database_name = current_database()"
        ),
        {"schema": schema, "name": name},
    ).scalar()


def _set_comment(conn, ref: str, kind: str, value: str | None) -> None:
    kw = _KEYWORD[kind]
    conn.execute(text(f"COMMENT ON {kw} {ref} IS {'NULL' if value is None else _literal(value)}"))


def park(conn, name: str, schema: str | None, kind: str | None = None) -> bool:
    """Set the live relation `name` aside as `__old__<name>` (same schema), in
    the caller's transaction. Returns True if a relation was kept, False if
    there was nothing to keep (absent, or a DuckDB view, which is dropped).
    Raises `KeptOldExists` if `__old__<name>` is taken."""
    dialect = conn.engine.dialect.name
    kind = kind or kind_of(conn, name, schema)
    if kind is None:
        return False
    live, aside = _ref(name, schema), old_name(name)
    if dialect == "postgresql" and len(aside) > PG_MAX_IDENTIFIER:
        raise ValueError(f"cannot keep {name!r} aside: {aside!r} exceeds PostgreSQL's {PG_MAX_IDENTIFIER}-byte identifier limit")
    if kind_of(conn, aside, schema) is not None:
        raise KeptOldExists(
            f"{_ref(aside, schema)} already exists: verify or roll back the previous publish and run "
            "cleanup_kept_old before publishing with keep_old again"
        )
    if kind != "table" and dialect != "postgresql":
        conn.execute(text(f"DROP VIEW {live}"))  # DuckDB views bind by name: nothing to keep
        return False
    grants = _grants(conn, name, schema) if dialect == "postgresql" else []
    conn.execute(text(f"ALTER {_KEYWORD[kind]} {live} RENAME TO {_quote(aside)}"))
    aside_ref = _ref(aside, schema)
    if grants:
        for grantee in sorted({g[0] for g in grants}):
            target = "PUBLIC" if grantee == "PUBLIC" else _quote(grantee)
            conn.execute(text(f"REVOKE ALL ON TABLE {aside_ref} FROM {target}"))
    _set_comment(conn, aside_ref, kind, TAG + json.dumps({"kind": kind, "grants": grants}, sort_keys=True))
    return True


def _restore_grants(conn, ref: str, comment: str | None) -> None:
    """Replay the grants recorded in a kept relation's comment. The comment is
    read back from the catalog, so it is data: every privilege is checked
    against `_PRIVILEGES` (before any is granted) rather than spliced in."""
    if not comment or not comment.startswith(TAG):
        return
    grants = [(grantee, str(privilege).upper()) for grantee, privilege in json.loads(comment[len(TAG):]).get("grants", [])]
    unknown = sorted({p for _, p in grants} - _PRIVILEGES)
    if unknown:
        raise ValueError(f"refusing to restore grants on {ref}: unknown privilege(s) {unknown} in its keep_old comment")
    for grantee, privilege in grants:
        target = "PUBLIC" if grantee == "PUBLIC" else _quote(grantee)
        conn.execute(text(f"GRANT {privilege} ON TABLE {ref} TO {target}"))


def list_kept(engine, schemas: list[str] | None = None) -> list[KeptRelation]:
    """Read-only: the relations kept aside by a `keep_old` publish (or by a
    rollback), i.e. `__old__*` relations carrying the sqlsink tag."""
    with engine.connect() as conn:
        return _list(conn, schemas)


def _list(conn, schemas: list[str] | None) -> list[KeptRelation]:
    tag = TAG.strip() + "%"
    if conn.engine.dialect.name == "postgresql":
        rows = conn.execute(
            text(
                "SELECT n.nspname, c.relname, c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE c.relname LIKE '\\_\\_old\\_\\_%' AND c.relkind IN ('r', 'p', 'v', 'm') "
                "AND n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg\\_toast%' "
                "AND obj_description(c.oid, 'pg_class') LIKE :tag ORDER BY 1, 2"
            ),
            {"tag": tag},
        ).all()
        kinds = {"r": "table", "p": "table", "v": "view", "m": "matview"}
        found = [(r[0], r[1], kinds[r[2]]) for r in rows]
    else:
        rows = conn.execute(
            text(
                "SELECT schema_name, table_name, 'table' FROM duckdb_tables() WHERE database_name = current_database() "
                "AND table_name LIKE '\\_\\_old\\_\\_%' ESCAPE '\\' AND comment LIKE :tag "
                "UNION ALL SELECT schema_name, view_name, 'view' FROM duckdb_views() WHERE database_name = current_database() "
                "AND NOT internal AND view_name LIKE '\\_\\_old\\_\\_%' ESCAPE '\\' AND comment LIKE :tag ORDER BY 1, 2"
            ),
            {"tag": tag},
        ).all()
        found = [(r[0], r[1], r[2]) for r in rows]
    return [KeptRelation(s, n[len(OLD_PREFIX):], k) for s, n, k in found if not schemas or s in schemas]


def _select(kept: list[KeptRelation], names) -> list[KeptRelation]:
    if names is None:
        return kept
    wanted = set(names)
    return [k for k in kept if k.name in wanted or f"{k.schema}.{k.name}" in wanted]


def cleanup_kept_old(engine, *, names=None, schemas=None, lock_timeout=None) -> list[str]:
    """Drop the kept `__old__*` relations (all, or the ones named in `names`:
    bare or `schema.name` of the ORIGINAL relation). One transaction, under the
    publish lock. Returns the dropped relations as `schema.__old__name`."""
    from .publish import PUBLISH_LOCK_KEY

    dropped: list[str] = []
    with engine.begin() as conn:
        apply_lock_timeout(conn, lock_timeout)
        with advisory_lock(conn, PUBLISH_LOCK_KEY, transactional=True):
            kept = _select(_list(conn, schemas), names)
            postgres = engine.dialect.name == "postgresql"
            # Views first: they read the kept tables. PostgreSQL drops a group
            # in one statement, which orders dependencies among the tables.
            for kind in ("view", "matview", "table"):
                group = [k for k in kept if k.kind == kind]
                if not group:
                    continue
                if postgres:
                    conn.execute(text(f"DROP {_KEYWORD[kind]} {', '.join(k.ref() for k in group)}"))
                else:
                    for k in group:
                        conn.execute(text(f"DROP {_KEYWORD[kind]} {k.ref()}"))
                dropped.extend(k.ref().replace('"', "") for k in group)
    return dropped


def rollback_kept_old(engine, *, names=None, schemas=None, lock_timeout=None) -> list[str]:
    """Swap every kept relation back in, atomically. For each kept `__old__x`:
    the live `x` (if any) is set aside as `__old__x` in turn (so the rollback can
    itself be undone by calling this again) and `__old__x` becomes `x`, getting
    back the grants it had. The markers of the restored relations are deleted
    (see the module docstring). Returns the restored relations as `schema.name`."""
    from .publish import PUBLISH_LOCK_KEY

    restored: list[str] = []
    with engine.begin() as conn:
        apply_lock_timeout(conn, lock_timeout)
        with advisory_lock(conn, PUBLISH_LOCK_KEY, transactional=True):
            kept = _select(_list(conn, schemas), names)
            for k in kept:
                aside_ref, tmp = k.ref(), _quote(TMP_PREFIX + k.name)
                comment = _comment(conn, k.old_name, k.schema, k.kind)
                conn.execute(text(f"ALTER {_KEYWORD[k.kind]} {aside_ref} RENAME TO {tmp}"))
                current = kind_of(conn, k.name, k.schema)
                if current is not None:
                    park(conn, k.name, k.schema, current)
                tmp_ref = _ref(TMP_PREFIX + k.name, k.schema)
                conn.execute(text(f"ALTER {_KEYWORD[k.kind]} {tmp_ref} RENAME TO {_quote(k.name)}"))
                live_ref = _ref(k.name, k.schema)
                _set_comment(conn, live_ref, k.kind, None)
                if conn.engine.dialect.name == "postgresql":
                    _restore_grants(conn, live_ref, comment)
                restored.append(f"{k.schema}.{k.name}")
            _forget_markers(conn, [k.name for k in kept])
    return restored


def _forget_markers(conn, names: list[str]) -> None:
    """Delete the publish markers (and component links) of relations whose
    content was just replaced by an older version."""
    if not names:
        return
    for table, key in (
        (meta_mod.analytics_table_updates, "table_name"),
        (meta_mod.analytics_dataset_updates, "dataset_name"),
        (meta_mod.component_updates, "component_name"),
        (meta_mod.contour_updates, "contour_table"),
        (meta_mod.dataset_components, "dataset_name"),
    ):
        if inspect(conn).has_table(table.name):
            conn.execute(delete(table).where(table.c[key].in_(names)))


def main(argv: list[str] | None = None) -> int:
    """`python -m sqlsink.keep_old {list,cleanup,rollback}`: operate on the
    relations a `keep_old` publish kept aside. The DSN comes from the
    environment variable named by `--dsn-env` (default `SQLSINK_DSN`)."""
    import argparse
    import os

    from .engine import make_engine

    parser = argparse.ArgumentParser(prog="python -m sqlsink.keep_old", description=main.__doc__)
    parser.add_argument("action", choices=["list", "cleanup", "rollback"])
    parser.add_argument("--dsn-env", default="SQLSINK_DSN")
    parser.add_argument("--name", action="append", help="only this original relation (bare or schema.name); repeatable")
    parser.add_argument("--schema", action="append", help="only kept relations in this schema; repeatable")
    parser.add_argument("--lock-timeout", help="PostgreSQL lock_timeout, e.g. 45s (default: SQLSINK_LOCK_TIMEOUT)")
    args = parser.parse_args(argv)
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        parser.error(f"environment variable {args.dsn_env} is not set")
    engine = make_engine(dsn)
    try:
        if args.action == "list":
            result = [f"{k.schema}.{k.old_name} ({k.kind})" for k in _select(list_kept(engine, args.schema), args.name)]
        elif args.action == "cleanup":
            result = cleanup_kept_old(engine, names=args.name, schemas=args.schema, lock_timeout=args.lock_timeout)
        else:
            result = rollback_kept_old(engine, names=args.name, schemas=args.schema, lock_timeout=args.lock_timeout)
    finally:
        engine.dispose()
    for line in result:
        print(line)
    print(f"{args.action}: {len(result)} relation(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
