"""Snakemake `script:` entrypoint for rule `publish_datasets` (any sink)."""

import json

from sql_incremental.manifest import load_manifest
from sql_incremental.sink import make_sink

sink = make_sink(snakemake.params.sink)  # noqa: F821
manifests = [load_manifest(spec) for spec in snakemake.params.manifests]  # noqa: F821


def _load(paths):
    receipts = []
    for path in paths:
        with open(path) as f:
            receipts.append(json.load(f))
    return receipts


published = sink.publish(
    manifests,
    _load(snakemake.input.dataset_receipts),  # noqa: F821
    _load(snakemake.input.contour_receipts),  # noqa: F821
)
print(f"Published {len(published)} dataset(s) to the {sink.name} sink: {published}")
