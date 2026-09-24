"""Content-derived identity for a published table.

Pure functions, no DB or Snakemake dependency, so they can be unit-tested
in isolation and reused by the one-off seeding script.
"""

from __future__ import annotations

import hashlib
import json

LOADER_VERSION = 1
TYPE_MAP_VERSION = 1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
