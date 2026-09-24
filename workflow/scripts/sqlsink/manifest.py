"""Declarative dataset/materialization manifest.

A `DatasetMaterialization` describes one *logical dataset* that has two
physical shapes: a Parquet export (handled entirely upstream, outside this
package) and a normalized PostgreSQL representation -- a compact fact table
without the repeated geometry column, a shared contour/dimension table, and
a public compatibility view that joins them back together under the
original dataset name, column names and order.

This is intentionally data-agnostic: join columns, the geometry column to
omit and the output column order are declared here explicitly, never
inferred from names, per the migration plan's correctness constraint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

MANIFEST_VERSION = 1
PART_COLUMN = "part_no"


@dataclass(frozen=True)
class DatasetMaterialization:
    name: str
    geometry_column: str
    contour_table: str
    fact_join_columns: tuple[str, ...]
    contour_join_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    fact_source: str | None = None
    contour_source: str = "zone_emploi_contours"
    compact_schema: str = "analytics_storage"
    # Relational form: the contour becomes `zone` (PK: join columns) plus
    # `zone_part` (PK: join columns + part_no, FK -> zone), and the compact
    # facts get a foreign key to `zone`. Same view output either way.
    keyed: bool = False

    def __post_init__(self):
        if len(self.fact_join_columns) != len(self.contour_join_columns):
            raise ValueError(
                f"dataset {self.name!r}: fact_join_columns and contour_join_columns "
                f"must declare the same number of columns, in corresponding order "
                f"({self.fact_join_columns!r} vs {self.contour_join_columns!r})"
            )
        if self.geometry_column not in self.output_columns:
            raise ValueError(
                f"dataset {self.name!r}: geometry_column {self.geometry_column!r} "
                f"must appear in output_columns (it is reconstructed via the join, "
                f"not stored in the compact fact table)"
            )
        for col in self.fact_join_columns:
            if col not in self.output_columns:
                raise ValueError(
                    f"dataset {self.name!r}: fact join column {col!r} is not in output_columns"
                )

    def fact_parquet_key(self) -> str:
        return self.fact_source or self.name

    def compact_columns(self) -> tuple[str, ...]:
        """Compact fact columns: the declared output, minus the geometry
        column that the compatibility view reconstructs via the join."""
        return tuple(c for c in self.output_columns if c != self.geometry_column)

    @property
    def zone_table(self) -> str:
        return f"{self.contour_table}_zone"

    def relation_graph(self):
        """The declared relations of the keyed form: fact -> zone <- part."""
        from .relations import Edge, Entity, JoinGraph

        return JoinGraph(
            (
                Entity(self.name, ()),
                Entity(self.zone_table, self.contour_join_columns),
                Entity(self.contour_table, (*self.contour_join_columns, PART_COLUMN)),
            ),
            (
                Edge(self.name, self.zone_table, self.fact_join_columns),
                Edge(self.contour_table, self.zone_table, self.contour_join_columns),
            ),
        )

    def canonical_dict(self) -> dict:
        d = self._base_canonical_dict()
        if self.keyed:  # absent when False, so pre-existing markers stay valid
            d["keyed"] = True
        return d

    def _base_canonical_dict(self) -> dict:
        return {
            "manifest_version": MANIFEST_VERSION,
            "name": self.name,
            "geometry_column": self.geometry_column,
            "contour_table": self.contour_table,
            "contour_source": self.contour_source,
            "fact_join_columns": list(self.fact_join_columns),
            "contour_join_columns": list(self.contour_join_columns),
            "output_columns": list(self.output_columns),
            "compact_schema": self.compact_schema,
        }

    def manifest_hash(self) -> str:
        """Canonical hash of everything that determines the compact table's
        layout and the view's join/select. Changing a join column, a cast,
        the geometry column or the output order must change this hash so
        that it invalidates every publication that depends on it."""
        blob = json.dumps(self.canonical_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def load_manifest(spec: dict) -> DatasetMaterialization:
    """Build a `DatasetMaterialization` from a plain dict (e.g. parsed from
    `config.yaml`), converting list fields to the tuples the dataclass
    expects."""
    kwargs = dict(spec)
    for key in ("fact_join_columns", "contour_join_columns", "output_columns"):
        if key in kwargs:
            kwargs[key] = tuple(kwargs[key])
    return DatasetMaterialization(**kwargs)
