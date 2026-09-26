"""Render the public relation of a v2 dataset from its components.

Pure: no database. A dataset's view is defined either by `joins` (rendered
here, validated against a `JoinGraph` when the manifest loads) or by a
`view_sql` template whose `{component}` references are replaced by a relation
expression: the qualified physical table of a sink, or `read_parquet(...)` for
a DuckDB run over source files.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import metadata as meta_mod
from .join import _preparer
from .manifest import TEMPLATE_REF, DatasetV2, ManifestError, SelectItem


def physical_refs(manifest: DatasetV2, dialect_name: str) -> dict[str, str]:
    """Quoted, schema-qualified physical table of each component."""
    q = _preparer(dialect_name).quote_identifier
    refs = {}
    for component in manifest.components:
        name, schema = meta_mod.physical_name_and_schema(dialect_name, component.name, manifest.schema_of(component))
        refs[component.name] = f"{q(schema)}.{q(name)}" if schema else q(name)
    return refs


def pick_template(manifest: DatasetV2, dialect_name: str) -> str:
    templates = dict(manifest.view_sql)
    if dialect_name in templates:
        return templates[dialect_name]
    if "default" in templates:
        return templates["default"]
    raise ManifestError(f"dataset {manifest.name!r}: no view_sql for dialect {dialect_name!r} and no default")


def render_template(template: str, refs: Mapping[str, str]) -> str:
    """Replace each `{component}` by its relation expression. Only declared
    names are touched, so other braces in the SQL survive; an unknown name is an error."""

    def replace(match) -> str:
        name = match.group(1)
        if name not in refs:
            raise ManifestError(f"view_sql references unknown component {{{name}}} (known: {sorted(refs)})")
        return refs[name]

    return TEMPLATE_REF.sub(replace, template)


def _resolve_select(
    manifest: DatasetV2, columns: Mapping[str, Sequence[str]], aliases: Mapping[str, str]
) -> list[tuple[str, str, str]]:
    """(alias, column, output name) per output column."""
    base = manifest.base
    assert base is not None
    items = manifest.select
    if not items:
        seen: set[str] = set()
        items_list: list[SelectItem] = []
        for name in aliases:  # base first, then joins in order
            for col in columns[name]:
                if col not in seen:
                    seen.add(col)
                    items_list.append(SelectItem(col, name))
        items = tuple(items_list)
    resolved = []
    outputs: set[str] = set()
    for item in items:
        if item.component is not None:
            owner = item.component
            if item.column not in columns[owner]:
                raise ManifestError(f"dataset {manifest.name!r}: component {owner!r} has no column {item.column!r}")
        else:
            owners = [n for n in aliases if item.column in columns[n]]
            if not owners:
                raise ManifestError(f"dataset {manifest.name!r}: no component has column {item.column!r}")
            # A join key is present on both sides: the base's copy wins; any
            # other duplicate is ambiguous and must name its component.
            owner = base if base in owners else owners[0]
            if len(owners) > 1 and base not in owners:
                raise ManifestError(
                    f"dataset {manifest.name!r}: column {item.column!r} is in {owners}; name one with `from`"
                )
        if item.output in outputs:
            raise ManifestError(f"dataset {manifest.name!r}: output column {item.output!r} selected twice")
        outputs.add(item.output)
        resolved.append((aliases[owner], item.column, item.output))
    return resolved


def render_view_sql(
    manifest: DatasetV2,
    dialect_name: str,
    refs: Mapping[str, str],
    columns: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """The SELECT defining the dataset's public relation.

    `refs` maps each component to a relation expression. `columns` (component
    -> column names) is needed for `joins` datasets whose `select` names
    columns without their component or is omitted."""
    if manifest.view_sql:
        return render_template(pick_template(manifest, dialect_name), refs)

    q = _preparer(dialect_name).quote_identifier
    assert manifest.base is not None
    aliases = {manifest.base: "f"}
    for index, join in enumerate(manifest.joins, start=1):
        aliases[join.component] = f"j{index}"
    if columns is None:
        columns = {}
    needs_columns = not manifest.select or any(i.component is None for i in manifest.select)
    if needs_columns and not all(name in columns for name in aliases):
        raise ManifestError(
            f"dataset {manifest.name!r}: column lists of {sorted(aliases)} are needed to resolve its select list"
        )
    if needs_columns:
        selected = _resolve_select(manifest, columns, aliases)
    else:
        selected = [(aliases[i.component], i.column, i.output) for i in manifest.select if i.component]
    select_list = ", ".join(f"{a}.{q(c)} AS {q(out)}" for a, c, out in selected)
    parts = [f"SELECT {select_list} FROM {refs[manifest.base]} f"]
    for join in manifest.joins:
        left = aliases[join.left or manifest.base]
        alias = aliases[join.component]
        on = " AND ".join(f"{left}.{q(l)} = {alias}.{q(r)}" for l, r in join.on)
        kind = "LEFT JOIN" if join.type == "left" else "JOIN"
        parts.append(f"{kind} {refs[join.component]} {alias} ON {on}")
    return " ".join(parts)
