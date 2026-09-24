"""Snakemake `script:` entrypoint for rule `publish_tables`."""

import json

from sqlsink.engine import make_engine
from sqlsink.publish import publish_tables

engine = make_engine(snakemake.params.dsn)  # noqa: F821

receipts = []
for path in snakemake.input.receipts:  # noqa: F821
    with open(path) as f:
        receipts.append(json.load(f))

published = publish_tables(engine, receipts)
print(f"Published {len(published)} table(s): {published}")
