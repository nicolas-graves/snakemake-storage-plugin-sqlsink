"""Snakemake `script:` entrypoint for rule `export_dataset`: write the
published version of one dataset to Parquet."""

from sqlsink.export import export_parquet
from sqlsink.manifest import load_manifest, load_shared_components
from sqlsink.sink import make_sink

sink = make_sink(snakemake.params.sink)  # noqa: F821
manifest = load_manifest(snakemake.params.manifest, load_shared_components(snakemake.params.get("shared")))  # noqa: F821
rows = export_parquet(sink, manifest, snakemake.output.parquet, threads=snakemake.threads)  # noqa: F821
print(f"Exported {rows} row(s) of {manifest.name} to {snakemake.output.parquet}")  # noqa: F821
