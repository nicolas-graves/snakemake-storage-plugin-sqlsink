"""Snakemake `script:` entrypoint for rule `rollback_kept_old`: swap the
relations a `keep_old` publish kept aside back in, in one transaction."""

import json

from sqlsink.credentials import resolve_url
from sqlsink.engine import make_engine
from sqlsink.keep_old import rollback_kept_old

engine = make_engine(resolve_url(**snakemake.params.db))  # noqa: F821
restored = rollback_kept_old(engine, lock_timeout=snakemake.params.get("lock_timeout"))  # noqa: F821
with open(snakemake.output.report, "w") as f:  # noqa: F821
    json.dump({"restored": restored}, f, indent=2)
print(f"Restored {len(restored)} relation(s): {restored}")
