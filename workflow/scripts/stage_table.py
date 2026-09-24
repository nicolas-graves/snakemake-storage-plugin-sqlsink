"""Snakemake `script:` entrypoint for rule `stage_table`."""

import json

from sql_incremental.engine import make_engine
from sql_incremental.stage import stage_table

engine = make_engine(snakemake.params.dsn)  # noqa: F821
receipt = stage_table(engine, snakemake.params.table, snakemake.input.parquet)  # noqa: F821

with open(snakemake.output.receipt, "w") as f:  # noqa: F821
    json.dump(receipt.to_dict(), f, indent=2)
