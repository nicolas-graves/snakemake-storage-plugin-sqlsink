"""Runtime read-only grants on published relations (PostgreSQL).

A rename-swap publish replaces a table with a new object that carries none
of the old one's privileges, so grants must be re-applied after a publish
and `role_has_select` is what tells whether that is needed.
"""

from __future__ import annotations

from sqlalchemy import text

from .fingerprint import published_marker


def _split(relation: str, default_schema: str = "public") -> tuple[str, str]:
    schema, _, name = relation.rpartition(".")
    return (schema or default_schema), name


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _require_postgres(engine) -> None:
    if engine.dialect.name != "postgresql":
        raise NotImplementedError(f"role grants need PostgreSQL, not {engine.dialect.name!r}")


def grant_runtime(engine, role: str, relations) -> list[str]:
    """GRANT USAGE on the schemas and SELECT on `relations` to `role`.
    Returns the sorted relations granted."""
    _require_postgres(engine)
    relations = sorted(set(relations))
    with engine.begin() as conn:
        for schema in sorted({_split(r)[0] for r in relations}):
            conn.execute(text(f"GRANT USAGE ON SCHEMA {_quote(schema)} TO {_quote(role)}"))
        for relation in relations:
            schema, name = _split(relation)
            conn.execute(text(f"GRANT SELECT ON {_quote(schema)}.{_quote(name)} TO {_quote(role)}"))
    return relations


def role_has_select(engine, role: str, relations) -> bool:
    """Read-only: does `role` currently hold SELECT on every relation?"""
    _require_postgres(engine)
    with engine.connect() as conn:
        for relation in relations:
            schema, name = _split(relation)
            granted = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.role_table_grants WHERE grantee = :role "
                    "AND table_schema = :schema AND table_name = :name AND privilege_type = 'SELECT'"
                ),
                {"role": role, "schema": schema, "name": name},
            ).first()
            if granted is None:
                return False
    return True


def grants_receipt(engine, role: str, relations) -> dict:
    """Receipt content: the sorted relations and each one's marker (without
    `published_at`), so it changes only when a relation's content was
    republished or the set changed, not when a publish merely refreshed the
    marker timestamps."""
    relations = sorted(set(relations))
    published = {}
    for relation in relations:
        published[relation] = published_marker(engine, _split(relation)[1], timestamps=False)
    return {"role": role, "relations": relations, "markers": published}
