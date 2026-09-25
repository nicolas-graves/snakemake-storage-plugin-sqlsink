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
# Write each `published/{table}` receipt (the marker row, now current) to its
# output path; the storage object's store_object() is a no-op.
from sqlsink.fingerprint import published_marker  # noqa: E402

for path in snakemake.output.published:  # noqa: F821
    table = path.rsplit("/", 1)[-1]
    with open(path, "w") as f:
        json.dump({"table": table, "marker": published_marker(engine, table, "table")}, f, indent=2, sort_keys=True, default=str)

print(f"Published {len(published)} table(s): {published}")
