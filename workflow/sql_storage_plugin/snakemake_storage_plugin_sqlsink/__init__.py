"""Private, repo-local Snakemake storage plugin.

Represents one analytics table as a Snakemake storage object, so that
Snakemake's own scheduler can decide -- via `exists()`/`mtime()` -- whether
a table needs to be re-staged, instead of always running `stage_table` and
having it write a no-op "current" receipt.

Queries select the kind of object:

    {table}             staged table (exists = current parquet is staged/published)
    published/{table}   published table (exists = publish marker matches the staged one)
    dataset/{name}      published dataset (needs the `manifests` setting)
    grants/{role}       runtime SELECT grants (needs the `grants_file` setting)

This does NOT replace `sqlsink.publish.publish_tables`: the storage
plugin interface has no notion of a transaction spanning several storage
objects, so atomic multi-table publication stays a separate final rule, as
before, whose outputs are the `published/{table}` objects. See
`workflow/rules/db_publish.smk`.

The "local materialization" of a table is a small JSON manifest (the same
shape as `sqlsink.stage.StageReceipt.to_dict()`), never the table's
actual rows -- retrieving/storing it is therefore cheap regardless of table
size.
"""


import datetime as dt
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from snakemake_interface_storage_plugins.io import IOCacheStorageInterface, Mtime
from snakemake_interface_storage_plugins.storage_object import (
    StorageObjectRead,
    StorageObjectWrite,
)
from snakemake_interface_storage_plugins.storage_provider import (
    ExampleQuery,
    Operation,
    QueryType,
    StorageProviderBase,
    StorageQueryValidationResult,
)
from snakemake_interface_storage_plugins.settings import StorageProviderSettingsBase

from sqlsink import grants as grants_mod
from sqlsink.engine import make_engine
from sqlsink.fingerprint import compute_dataset_update_id_for_files, published_marker
from sqlsink.metadata import create_all
from sqlsink.stage import fetch_marker, fetch_staged_marker, is_current, stage_table

TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
KINDS = ("published", "dataset", "grants")
ON_UNREACHABLE = ("error", "treat-missing")


def split_query(query: str) -> tuple[str, str]:
    """`"published/t"` -> `("published", "t")`; a bare name is a staged table."""
    kind, sep, name = query.partition("/")
    return (kind, name) if sep and kind in KINDS else ("staged", query)


@dataclass
class StorageProviderSettings(StorageProviderSettingsBase):
    dsn: Optional[str] = field(
        default=None,
        metadata={"help": "SQLAlchemy DSN of the database holding the marker table and the tables themselves. Stored in Snakemake's metadata: prefer dsn_file or dsn_env."},
    )
    dsn_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path of a file holding the DSN (whitespace stripped); keeps the secret out of .snakemake metadata."},
    )
    dsn_env: Optional[str] = field(
        default=None,
        metadata={"help": "Name of an environment variable holding the DSN."},
    )
    manifests: Optional[str] = field(
        default=None,
        metadata={"help": "YAML file of dataset manifests (a list, or a mapping with a 'datasets' list), for 'dataset/{name}' objects."},
    )
    grants_file: Optional[str] = field(
        default=None,
        metadata={"help": "YAML mapping role -> list of relations ('name' or 'schema.name'), for 'grants/{role}' objects."},
    )
    on_unreachable: str = field(
        default="error",
        metadata={"help": "'error' (default) or 'treat-missing': when the database is unreachable at startup, report every object as missing so dry runs work. Can hide a real outage."},
    )
    parquet_dir: str = field(
        default="results/parquet",
        metadata={"help": "Directory containing '{table}.parquet' upstream outputs."},
    )


def resolve_dsn(settings: StorageProviderSettings) -> str:
    """The DSN from `dsn`, else `dsn_file`, else `dsn_env`."""
    if settings.dsn:
        return settings.dsn
    if settings.dsn_file:
        return Path(settings.dsn_file).read_text().strip()
    if settings.dsn_env:
        value = os.environ.get(settings.dsn_env)
        if not value:
            raise ValueError(f"environment variable {settings.dsn_env!r} (dsn_env) is not set")
        return value
    raise ValueError("no database configured: set one of dsn, dsn_file, dsn_env")


class StorageProvider(StorageProviderBase):
    def __post_init__(self):
        if self.settings.on_unreachable not in ON_UNREACHABLE:
            raise ValueError(f"on_unreachable must be one of {ON_UNREACHABLE}, not {self.settings.on_unreachable!r}")
        self.reachable = True
        # Snakemake queries every storage object's exists()/mtime() while
        # building the DAG, before any rule (including a would-be
        # "setup_pg_meta" rule) has run. So the marker table has to be
        # guaranteed to exist right here, at provider construction time,
        # rather than via a separate Snakemake rule with its own ordering.
        self.engine = make_engine(resolve_dsn(self.settings))
        try:
            create_all(self.engine)
        except SQLAlchemyError:
            if self.settings.on_unreachable == "error":
                raise
            self.reachable = False
            self.logger.warning("sqlsink: database unreachable, treating every storage object as missing")
        self._manifests = None
        self._grants = None

    def manifests(self) -> dict:
        if self._manifests is None:
            import yaml
            from sqlsink.manifest import load_manifest

            if not self.settings.manifests:
                raise ValueError("'dataset/{name}' objects need the `manifests` provider setting")
            data = yaml.safe_load(Path(self.settings.manifests).read_text())
            specs = data.get("datasets", []) if isinstance(data, dict) else data
            self._manifests = {m.name: m for m in map(load_manifest, specs)}
        return self._manifests

    def grant_relations(self, role: str) -> list:
        if self._grants is None:
            import yaml

            if not self.settings.grants_file:
                raise ValueError("'grants/{role}' objects need the `grants_file` provider setting")
            self._grants = yaml.safe_load(Path(self.settings.grants_file).read_text()) or {}
        if role not in self._grants:
            raise ValueError(f"role {role!r} is not declared in {self.settings.grants_file}")
        return sorted(self._grants[role])

    @classmethod
    def example_queries(cls) -> List[ExampleQuery]:
        return [
            ExampleQuery(
                query="some_table",
                type=QueryType.OUTPUT,
                description="Name of an analytics table staged by sqlsink.",
            ),
            ExampleQuery(query="published/some_table", type=QueryType.INPUT, description="A published table."),
            ExampleQuery(query="dataset/some_dataset", type=QueryType.INPUT, description="A published dataset."),
            ExampleQuery(query="grants/some_role", type=QueryType.INPUT, description="Runtime SELECT grants of a role."),
        ]

    def rate_limiter_key(self, query: str, operation: Operation) -> Any:
        return None

    def default_max_requests_per_second(self) -> float:
        return 1000.0

    def use_rate_limiter(self) -> bool:
        return False

    @classmethod
    def is_valid_query(cls, query: str) -> StorageQueryValidationResult:
        if "{" in query:
            # Contains an unresolved wildcard (e.g. "{table}"); wildcards
            # are resolved before the storage object is actually used, so
            # accept it here rather than validating the literal string.
            return StorageQueryValidationResult(query=query, valid=True)
        if not TABLE_NAME_RE.match(split_query(query)[1]):
            return StorageQueryValidationResult(
                query=query,
                valid=False,
                reason="query must be a bare name ([a-zA-Z_][a-zA-Z0-9_]*), optionally prefixed by "
                "published/, dataset/ or grants/; not a path or URL",
            )
        return StorageQueryValidationResult(query=query, valid=True)


class StorageObject(StorageObjectRead, StorageObjectWrite):
    def __post_init__(self):
        self.kind, self.table_name = split_query(self.query)

    @property
    def _engine(self):
        return self.provider.engine

    def _parquet_path(self, key: Optional[str] = None) -> str:
        return str(Path(self.provider.settings.parquet_dir) / f"{key or self.table_name}.parquet")

    def local_suffix(self) -> str:
        return self.query

    async def inventory(self, cache: IOCacheStorageInterface):
        key = self.cache_key()
        if key in cache.exists_in_storage:
            return
        exists = self.exists()
        cache.exists_in_storage[key] = exists
        if exists:
            cache.mtime[key] = Mtime(storage=self.mtime())
            cache.size[key] = self.size()

    def get_inventory_parent(self) -> Optional[str]:
        return None

    def cleanup(self):
        pass

    # -- per-kind state -------------------------------------------------

    def _manifest(self):
        return self.provider.manifests()[self.table_name]

    def _dataset_marker(self):
        return published_marker(self._engine, self.table_name, "dataset")

    def _dataset_current(self) -> bool:
        from sqlsink.sink_postgres import SqlSink

        manifest = self._manifest()
        update_id, _, _ = compute_dataset_update_id_for_files(
            manifest, self._parquet_path(manifest.fact_parquet_key()), self._parquet_path(manifest.contour_source)
        )
        marker = self._dataset_marker()
        return (
            marker is not None
            and marker["update_id"] == update_id
            and SqlSink(self._engine).published_intact(manifest)
        )

    def _published_current(self) -> bool:
        # The publish marker must be the staged one: a restage with new
        # content leaves a staged marker the publish marker does not match.
        marker = fetch_marker(self._engine, self.table_name)
        if marker is None or not inspect(self._engine).has_table(self.table_name):
            return False
        staged = fetch_staged_marker(self._engine, self.table_name)
        return staged is None or staged["update_id"] == marker["update_id"]

    def _grants_current(self) -> bool:
        return grants_mod.role_has_select(self._engine, self.table_name, self.provider.grant_relations(self.table_name))

    def exists(self) -> bool:
        if not self.provider.reachable:
            return False
        if self.kind == "published":
            return self._published_current()
        if self.kind == "dataset":
            return self._dataset_current()
        if self.kind == "grants":
            return self._grants_current()
        # Read-only freshness check: no DuckDB read, no writes. This is the
        # actual gain of this plugin -- Snakemake can skip the whole
        # `stage_table` rule for a table that's already current, rather
        # than running it just to have it write a no-op receipt.
        return is_current(self._engine, self.table_name, self._parquet_path())

    def mtime(self) -> float:
        # Must always be a finite, valid epoch timestamp (Snakemake uses it
        # for os.utime() on the local proxy file after storing).
        if not self.provider.reachable:
            return 0.0
        if self.kind == "grants":
            stamps = [
                m["published_at"]
                for m in (published_marker(self._engine, r.rpartition(".")[2]) for r in self.provider.grant_relations(self.table_name))
                if m
            ]
            return dt.datetime.fromisoformat(max(stamps)).timestamp() if stamps else 0.0
        if self.kind == "dataset":
            marker = self._dataset_marker()
            return dt.datetime.fromisoformat(marker["published_at"]).timestamp() if marker else 0.0
        # A table that was just staged but not yet published (publish_tables
        # hasn't run yet this session) has no publish marker row -- fall
        # back to the staging marker's timestamp in that case.
        marker = fetch_marker(self._engine, self.table_name)
        if marker is not None:
            return marker["published_at"].timestamp()
        if self.kind == "published":
            return 0.0
        staged = fetch_staged_marker(self._engine, self.table_name)
        if staged is not None:
            return staged["staged_at"].timestamp()
        return 0.0

    def size(self) -> int:
        return self.local_path().stat().st_size if self.local_path().exists() else 0

    def local_footprint(self) -> int:
        # The local materialization is a small JSON manifest, never the
        # table's actual rows.
        return self.size()

    def _receipt(self) -> dict:
        """Content of the local receipt. For the published kinds it is built
        from marker rows only, so it changes exactly when a relation was
        republished and is byte-stable otherwise."""
        if self.kind == "published":
            return {"table": self.table_name, "marker": published_marker(self._engine, self.table_name, "table")}
        if self.kind == "dataset":
            manifest = self._manifest()
            return {
                "dataset": self.table_name,
                "marker": self._dataset_marker(),
                "contour": published_marker(self._engine, manifest.contour_table, "contour"),
            }
        if self.kind == "grants":
            return grants_mod.grants_receipt(
                self._engine, self.table_name, self.provider.grant_relations(self.table_name)
            )
        # Called both when a downstream rule needs this table as an input
        # (table already current, `stage_table` never ran this session --
        # there is no local file to fall back on) and, defensively, in any
        # other case: `stage_table` itself is cheap when the table already
        # matches the marker (it returns before touching DuckDB or the
        # staging table), so calling it here is always safe.
        return stage_table(self._engine, self.table_name, self._parquet_path()).to_dict()

    def retrieve_object(self):
        """Materialize the receipt for this object locally."""
        receipt = self._receipt()
        self.local_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_path(), "w") as f:
            json.dump(receipt, f, indent=2, sort_keys=True, default=str)

    def store_object(self):
        # No-op: the rule's script (`stage_table.py`, `publish_tables.py`,
        # ...) already performed the real side effect and wrote the receipt
        # to local_path() itself. There is no separate "storage backend
        # location" to push it to -- the database rows are the storage.
        pass

    def remove(self):
        # A published/staged table is never removed via the storage
        # plugin's own lifecycle; deletion is a deliberate, separate
        # operation on the database, not a workflow side effect.
        pass
