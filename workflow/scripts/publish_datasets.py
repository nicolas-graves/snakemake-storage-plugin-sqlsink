"""Snakemake `script:` entrypoint for rule `publish_datasets` (any sink)."""

import json

from sqlsink.manifest import load_manifests
from sqlsink.publish import merge_component_receipts
from sqlsink.sink import make_sink

sink = make_sink(snakemake.params.sink)  # noqa: F821
manifests = load_manifests(
    {"datasets": snakemake.params.manifests, "components": snakemake.params.get("shared", [])}  # noqa: F821
)


def _load(paths):
    receipts = []
    for path in paths:
        with open(path) as f:
            loaded = json.load(f)
        receipts.extend(loaded if isinstance(loaded, list) else [loaded])
    return receipts


contours, component_list = [], []
for receipt in _load(snakemake.input.contour_receipts):  # noqa: F821
    (component_list if receipt.get("kind") == "component" else contours).append(receipt)
components = merge_component_receipts(component_list)  # a shared component appears once per dataset

published = sink.publish(
    manifests,
    _load(snakemake.input.dataset_receipts),  # noqa: F821
    contours,
    component_receipts=list(components.values()),
)
print(f"Published {len(published)} dataset(s) to the {sink.name} sink: {published}")
