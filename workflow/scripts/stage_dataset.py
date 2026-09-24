"""Snakemake `script:` entrypoint for rule `stage_dataset` (any sink)."""

import json

from sql_incremental.manifest import load_manifest
from sql_incremental.sink import make_sink, normalize, stage

sink = make_sink(snakemake.params.sink)  # noqa: F821
manifest = load_manifest(snakemake.params.manifest)  # noqa: F821
dataset = normalize(
    manifest, snakemake.input.fact_parquet, snakemake.input.contour_parquet  # noqa: F821
)

result = stage(dataset, sink, threads=snakemake.threads)  # noqa: F821

with open(snakemake.output.contour_receipt, "w") as f:  # noqa: F821
    json.dump(result.contour.to_dict(), f, indent=2)

with open(snakemake.output.dataset_receipt, "w") as f:  # noqa: F821
    json.dump(result.dataset.to_dict(), f, indent=2)
