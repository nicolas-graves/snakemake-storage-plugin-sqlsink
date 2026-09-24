"""Dialect-independent description of "compact facts JOIN contours".

The join is declared once, structurally (`JoinSpec`), and rendered to SQL
text for a target: a PostgreSQL/DuckDB compatibility view over physical
tables, or DuckDB reading component Parquet files. Nothing here touches a
database; identifier quoting comes from SQLAlchemy's per-dialect preparer.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.engine.url import URL

from . import metadata as meta_mod
from .manifest import DatasetMaterialization


@dataclass(frozen=True)
class Relation:
    """One side of the join: either a physical table (`name`, optionally
    schema-qualified) or a raw SQL relation expression (`sql`), e.g.
    `read_parquet('...')`."""

    name: str | None = None
    schema: str | None = None
    sql: str | None = None

    def __post_init__(self):
        if (self.name is None) == (self.sql is None):
            raise ValueError("Relation needs exactly one of `name` or `sql`")


@dataclass(frozen=True)
class JoinSpec:
    fact: Relation
    contour: Relation
    # (alias, column) per output column, in output order; alias is "f" or "c".
    select: tuple[tuple[str, str], ...]
    # (fact column, contour column) pairs, ANDed.
    on: tuple[tuple[str, str], ...]


def join_spec(
    manifest: DatasetMaterialization, fact: Relation, contour: Relation
) -> JoinSpec:
    return JoinSpec(
        fact=fact,
        contour=contour,
        select=tuple(
            ("c" if col == manifest.geometry_column else "f", col)
            for col in manifest.output_columns
        ),
        on=tuple(zip(manifest.fact_join_columns, manifest.contour_join_columns)),
    )


def physical_join_spec(manifest: DatasetMaterialization, dialect_name: str) -> JoinSpec:
    """Join over the published compact + contour tables of `dialect_name`."""
    compact_name, compact_schema = meta_mod.physical_name_and_schema(
        dialect_name, meta_mod.compact_table_name(manifest.name), manifest.compact_schema
    )
    contour_name, contour_schema = meta_mod.physical_name_and_schema(
        dialect_name, manifest.contour_table, manifest.compact_schema
    )
    return join_spec(
        manifest,
        Relation(name=compact_name, schema=compact_schema),
        Relation(name=contour_name, schema=contour_schema),
    )


def _preparer(dialect_name: str):
    # DuckDB has no SQLAlchemy dialect here; its identifier quoting is the
    # ANSI/PostgreSQL one.
    if dialect_name == "duckdb":
        dialect_name = "postgresql"
    return URL.create(dialect_name).get_dialect()().identifier_preparer


def render_join_sql(spec: JoinSpec, dialect_name: str) -> str:
    q = _preparer(dialect_name).quote_identifier

    def ref(rel: Relation) -> str:
        if rel.sql is not None:
            return rel.sql
        return f"{q(rel.schema)}.{q(rel.name)}" if rel.schema else q(rel.name)

    select_list = ", ".join(f"{alias}.{q(col)}" for alias, col in spec.select)
    on = " AND ".join(f"f.{q(fc)} = c.{q(cc)}" for fc, cc in spec.on)
    return (
        f"SELECT {select_list} "
        f"FROM {ref(spec.fact)} f "
        f"JOIN {ref(spec.contour)} c ON {on}"
    )
