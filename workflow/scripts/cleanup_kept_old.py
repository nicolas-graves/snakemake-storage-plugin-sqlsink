"""Snakemake `script:` entrypoint for rule `cleanup_kept_old`: drop the
relations a `keep_old` publish kept aside (`__old__*`)."""

import json

from sqlsink.engine import make_engine
from sqlsink.keep_old import cleanup_kept_old

engine = make_engine(snakemake.params.dsn)  # noqa: F821
dropped = cleanup_kept_old(engine, lock_timeout=snakemake.params.get("lock_timeout"))  # noqa: F821
with open(snakemake.output.report, "w") as f:  # noqa: F821
    json.dump({"dropped": dropped}, f, indent=2)
print(f"Dropped {len(dropped)} kept relation(s): {dropped}")
