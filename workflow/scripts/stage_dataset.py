"""Snakemake `script:` entrypoint for rule `stage_dataset` (any sink).

v1 datasets take their two Parquets as `input[0]` (facts) and `input[1]`
(contours); the receipts are the dataset's and the contour's. v2 datasets read
each component's Parquet from `params.parquet_dir`; `output.contour_receipt`
then holds the list of component receipts (the rule keeps one shape for both).
"""

import json

from sqlsink.manifest import DatasetV2, load_manifest, load_shared_components
from sqlsink.sink import make_sink, normalize, normalize_v2, sources_from_dir, stage, stage_v2

sink = make_sink(snakemake.params.sink)  # noqa: F821
manifest = load_manifest(
    snakemake.params.manifest, load_shared_components(snakemake.params.get("shared"))  # noqa: F821
)

if isinstance(manifest, DatasetV2):
    dataset = normalize_v2(manifest, sources_from_dir(manifest, snakemake.params.parquet_dir), sink=sink)  # noqa: F821
    result = stage_v2(dataset, sink, threads=snakemake.threads)  # noqa: F821
    side_receipt = [r.to_dict() for r in result.components]
else:
    dataset = normalize(manifest, snakemake.input[0], snakemake.input[1])  # noqa: F821
    result = stage(dataset, sink, threads=snakemake.threads)  # noqa: F821
    side_receipt = result.contour.to_dict()

with open(snakemake.output.contour_receipt, "w") as f:  # noqa: F821
    json.dump(side_receipt, f, indent=2)

with open(snakemake.output.dataset_receipt, "w") as f:  # noqa: F821
    json.dump(result.dataset.to_dict(), f, indent=2)
