"""Private, repo-local Snakemake storage plugin.

Represents one analytics table as a Snakemake storage object, so that
Snakemake's own scheduler can decide -- via `exists()`/`mtime()` -- whether
a table needs to be re-staged, instead of always running `stage_table` and
having it write a no-op "current" receipt.

This does NOT replace `sqlsink.publish.publish_tables`: the storage
plugin interface has no notion of a transaction spanning several storage
objects, so atomic multi-table publication stays a separate final rule, as
before. See `workflow/rules/postgres_publish.smk`.

The "local materialization" of a table is a small JSON manifest (the same
shape as `sqlsink.stage.StageReceipt.to_dict()`), never the table's
actual rows -- retrieving/storing it is therefore cheap regardless of table
size.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

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

from sqlsink.engine import make_engine
from sqlsink.metadata import create_all
from sqlsink.stage import fetch_marker, fetch_staged_marker, is_current, stage_table

TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass
class StorageProviderSettings(StorageProviderSettingsBase):
    dsn: Optional[str] = field(
        default=None,
        metadata={"help": "SQLAlchemy DSN of the database holding the marker table and the tables themselves."},
    )
    parquet_dir: str = field(
        default="results/parquet",
        metadata={"help": "Directory containing '{table}.parquet' upstream outputs."},
    )


class StorageProvider(StorageProviderBase):
    def __post_init__(self):
        # Snakemake queries every storage object's exists()/mtime() while
        # building the DAG, before any rule (including a would-be
        # "setup_pg_meta" rule) has run. So the marker table has to be
        # guaranteed to exist right here, at provider construction time,
        # rather than via a separate Snakemake rule with its own ordering.
        self.engine = make_engine(self.settings.dsn)
        create_all(self.engine)

    @classmethod
    def example_queries(cls) -> List[ExampleQuery]:
        return [
            ExampleQuery(
                query="some_table",
                type=QueryType.OUTPUT,
                description="Name of an analytics table published by sqlsink.",
            )
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
        if not TABLE_NAME_RE.match(query):
            return StorageQueryValidationResult(
                query=query,
                valid=False,
                reason="query must be a bare table name ([a-zA-Z_][a-zA-Z0-9_]*), not a path or URL",
            )
        return StorageQueryValidationResult(query=query, valid=True)


class StorageObject(StorageObjectRead, StorageObjectWrite):
    def __post_init__(self):
        self.table_name = self.query

    @property
    def _engine(self):
        return self.provider.engine

    def _parquet_path(self) -> str:
        return str(Path(self.provider.settings.parquet_dir) / f"{self.table_name}.parquet")

    def local_suffix(self) -> str:
        return self.table_name

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

    def exists(self) -> bool:
        # Read-only freshness check: no DuckDB read, no writes. This is the
        # actual gain of this plugin -- Snakemake can skip the whole
        # `stage_table` rule for a table that's already current, rather
        # than running it just to have it write a no-op receipt.
        return is_current(self._engine, self.table_name, self._parquet_path())

    def mtime(self) -> float:
        # Must always be a finite, valid epoch timestamp (Snakemake uses it
        # for os.utime() on the local proxy file after storing). A table
        # that was just staged but not yet published (publish_tables
        # hasn't run yet this session) has no publish marker row -- fall
        # back to the staging marker's timestamp in that case.
        marker = fetch_marker(self._engine, self.table_name)
        if marker is not None:
            return marker["published_at"].timestamp()
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

    def retrieve_object(self):
        """Materialize the manifest for this table locally.

        Called both when a downstream rule needs this table as an input
        (table already current, `stage_table` never ran this session --
        there is no local file to fall back on) and, defensively, in any
        other case: `stage_table` itself is cheap when the table already
        matches the marker (it returns before touching DuckDB or the
        staging table), so calling it here is always safe.
        """
        receipt = stage_table(self._engine, self.table_name, self._parquet_path())
        self.local_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self.local_path(), "w") as f:
            json.dump(receipt.to_dict(), f, indent=2)

    def store_object(self):
        # No-op: the rule's script (`stage_table.py`) already performed the
        # real side effect (staging into stg__{table}) and wrote the
        # manifest to local_path() itself. There is no separate "storage
        # backend location" to push it to -- the database rows are the
        # storage, and they were already updated in-process by the script.
        pass

    def remove(self):
        # A published/staged table is never removed via the storage
        # plugin's own lifecycle; deletion is a deliberate, separate
        # operation on the database, not a workflow side effect.
        pass
