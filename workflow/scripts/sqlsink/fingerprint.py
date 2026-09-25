"""Content-derived identity for a published table.

Pure functions, no DB or Snakemake dependency, so they can be unit-tested
in isolation and reused by the one-off seeding script. `fingerprint` at the
end is the exception: it reads a live database (lazy imports).
"""

from __future__ import annotations

import hashlib
import json

LOADER_VERSION = 1
TYPE_MAP_VERSION = 2  # 2: DOUBLE -> sa.Double (was FLOAT: float32 on DuckDB), unmapped types raise


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrow(table) -> str:
    """Content hash of an in-memory Arrow table (schema + rows, chunk-layout
    independent): the IPC stream of the table, hashed as it is written."""
    import pyarrow as pa

    class _Hasher:
        def __init__(self):
            self.h = hashlib.sha256()
            self.pos = 0

        def write(self, data):
            self.h.update(data)
            self.pos += len(data)
            return len(data)

        def flush(self):
            pass

        def tell(self):
            return self.pos

        def close(self):
            pass

        closed = False

        def writable(self):
            return True

        def seekable(self):
            return False

        def readable(self):
            return False

    sink = _Hasher()
    combined = table.combine_chunks()
    with pa.ipc.new_stream(pa.PythonFile(sink, mode="w"), combined.schema) as writer:
        writer.write_table(combined)
    return sink.h.hexdigest()


def sha256_source(source) -> str:
    """Fingerprint of a source: a Parquet file path, or an `ArrowSource`."""
    from .queries import ArrowSource

    if isinstance(source, ArrowSource):
        return sha256_arrow(source.table)
    return sha256_file(source)


def compute_update_id(
    table: str,
    parquet_sha256: str,
    extra_config: dict | None = None,
    loader_version: int = LOADER_VERSION,
    type_map_version: int = TYPE_MAP_VERSION,
) -> str:
    """Identity of "table T, built from this Parquet, with this loader"."""
    payload = {
        "table": table,
        "parquet_sha256": parquet_sha256,
        "loader_version": loader_version,
        "type_map_version": type_map_version,
        "extra_config": extra_config or {},
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def compute_update_id_for_file(
    table: str,
    parquet_path: str,
    extra_config: dict | None = None,
    loader_version: int = LOADER_VERSION,
    type_map_version: int = TYPE_MAP_VERSION,
) -> tuple[str, str]:
    """Convenience wrapper: hashes the file, then computes the update_id.

    Returns (update_id, parquet_sha256) since both are worth keeping in a
    receipt/marker row.
    """
    parquet_sha256 = sha256_file(parquet_path)
    update_id = compute_update_id(
        table,
        parquet_sha256,
        extra_config=extra_config,
        loader_version=loader_version,
        type_map_version=type_map_version,
    )
    return update_id, parquet_sha256


def compute_dataset_update_id(
    manifest_hash: str,
    fact_sha256: str,
    contour_sha256: str,
    loader_version: int = LOADER_VERSION,
    type_map_version: int = TYPE_MAP_VERSION,
) -> str:
    """Identity of "this logical dataset, built from this fact Parquet and
    this contour Parquet, under this materialization manifest, with this
    loader". A geometry (manifest) change or a contour change must
    invalidate the publication even when the fact Parquet itself did not
    change, so both feed the hash independently of the fact fingerprint.
    """
    payload = {
        "kind": "dataset",
        "manifest_hash": manifest_hash,
        "fact_sha256": fact_sha256,
        "contour_sha256": contour_sha256,
        "loader_version": loader_version,
        "type_map_version": type_map_version,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def compute_dataset_update_id_for_files(
    manifest,
    fact_parquet_path: str,
    contour_parquet_path: str,
    loader_version: int = LOADER_VERSION,
    type_map_version: int = TYPE_MAP_VERSION,
) -> tuple[str, str, str]:
    """Convenience wrapper mirroring `compute_update_id_for_file`.

    Returns (update_id, fact_sha256, contour_sha256).
    """
    fact_sha256 = sha256_file(fact_parquet_path)
    contour_sha256 = sha256_file(contour_parquet_path)
    update_id = compute_dataset_update_id(
        manifest.manifest_hash(),
        fact_sha256,
        contour_sha256,
        loader_version=loader_version,
        type_map_version=type_map_version,
    )
    return update_id, fact_sha256, contour_sha256


def _split_relation(relation: str) -> tuple[str | None, str]:
    schema, _, name = relation.rpartition(".")
    return (schema or None), name


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def published_marker(engine, name: str, kind: str | None = None) -> dict | None:
    """The publish marker row of relation `name` (a table, else a dataset,
    else a contour table; only `kind` if given), reduced to JSON-safe change-token fields."""
    from sqlalchemy import select

    from . import metadata as meta_mod

    for k, table, key in (
        ("table", meta_mod.analytics_table_updates, "table_name"),
        ("dataset", meta_mod.analytics_dataset_updates, "dataset_name"),
        ("contour", meta_mod.contour_updates, "contour_table"),
    ):
        if kind not in (None, k):
            continue
        with engine.connect() as conn:
            row = conn.execute(select(table).where(table.c[key] == name)).mappings().first()
        if row is not None:
            out = {"kind": k}
            for col, value in row.items():
                if col != key:
                    out[col] = _iso(value) if col == "published_at" else value
            return out
    return None


def fingerprint(engine, relations, *, count_rows=True) -> dict:
    """Deterministic state fingerprint of `relations` ("name" or "schema.name").

    Neither PostgreSQL nor DuckDB exposes a cheap per-table mtime or content
    checksum, so the authoritative change token is the plugin's marker row
    (source checksum, row count, `published_at`). It is combined with cheap
    probes: a hash of the column names/types, optionally `count(*)` (a scan
    on PostgreSQL: pass `count_rows=False`, or a set of relation names to
    count, for large tables) and, on PostgreSQL, `pg_relation_size` and
    `pg_class.relfilenode` (which changes on TRUNCATE/VACUUM FULL/CLUSTER).
    No wall-clock value other than the marker's `published_at`, so the same
    database state always yields the same `fingerprint_json` bytes. Accepted
    limit: an out-of-band UPDATE that keeps row count, size and relfilenode
    is not detected until the table is restaged.
    """
    from sqlalchemy import inspect, text

    dialect = engine.dialect.name
    inspector = inspect(engine)
    out: dict = {}
    for relation in sorted(set(relations)):
        schema, name = _split_relation(relation)
        entry: dict = {"marker": published_marker(engine, name)}
        exists = inspector.has_table(name, schema=schema) or name in inspector.get_view_names(schema=schema)
        entry["exists"] = exists
        if exists:
            cols = [[c["name"], str(c["type"])] for c in inspector.get_columns(name, schema=schema)]
            entry["columns_sha256"] = hashlib.sha256(json.dumps(cols).encode()).hexdigest()
            ref = f'"{schema}"."{name}"' if schema else f'"{name}"'
            if count_rows is True or (count_rows and relation in count_rows):
                with engine.connect() as conn:
                    entry["row_count"] = conn.execute(text(f"SELECT count(*) FROM {ref}")).scalar_one()
            if dialect == "postgresql":
                with engine.connect() as conn:
                    row = conn.execute(
                        text("SELECT pg_relation_size(c.oid), c.relfilenode FROM pg_class c WHERE c.oid = to_regclass(:ref)"),
                        {"ref": ref},
                    ).first()
                if row is not None:
                    entry["relation_size"], entry["relfilenode"] = int(row[0]), int(row[1])
        out[relation] = entry
    return out


def fingerprint_json(engine, relations, **kwargs) -> str:
    """Byte-stable serialization of `fingerprint`."""
    return json.dumps(fingerprint(engine, relations, **kwargs), sort_keys=True, indent=2, default=str) + "\n"
