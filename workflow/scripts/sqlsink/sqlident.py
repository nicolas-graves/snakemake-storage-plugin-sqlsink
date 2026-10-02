"""SQL identifier and string-literal quoting shared by every module.

A leaf module (no package imports) so that anything can use it. Identifiers
are *always* double-quoted, with embedded `"` doubled: this is valid on
PostgreSQL and DuckDB alike, keeps the rendered SQL byte-identical to the
`"{name}"` form used before for every name without a `"` (rendered SQL feeds
some fingerprints), and is safe for names that come from data -- Parquet
column names, catalog names -- rather than from a validated manifest.
"""

from __future__ import annotations

import re

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def quote_ident(identifier: str) -> str:
    """`identifier` as a quoted SQL identifier."""
    return '"' + str(identifier).replace('"', '""') + '"'


def qualified(name: str, schema: str | None = None) -> str:
    """`"schema"."name"`, or `"name"` when `schema` is None or empty."""
    return f"{quote_ident(schema)}.{quote_ident(name)}" if schema else quote_ident(name)


def quote_literal(value: str) -> str:
    """`value` as a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def is_plain_ident(value: object) -> bool:
    """True if `value` is a plain identifier (`[A-Za-z_][A-Za-z0-9_]*`, the
    whole string: `re.match` with `$` would accept a trailing newline)."""
    return isinstance(value, str) and IDENT_RE.fullmatch(value) is not None
