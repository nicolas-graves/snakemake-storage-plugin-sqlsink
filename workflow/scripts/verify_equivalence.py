"""Check that each published dataset view equals its original Parquet as a multiset.

Two SQL sinks, one proof. `--target postgres` (default) compares the
published compatibility view with the Parquet. `--target duckdb` first
materializes each joined Parquet into a DuckDB file (--duckdb-path), then
compares its published view with the original (no server needed).

Usage (from the repository root):

    python workflow/scripts/verify_equivalence.py --parquet-dir results/parquet
    python workflow/scripts/verify_equivalence.py --dataset formation_initiale_fap_maps \\
        --parquet-dir /path/to/out --dsn postgresql+psycopg://user:pass@host/db

Exit status is 1 if any dataset differs. Memory is bounded (see
`sql_incremental.verify`); lower --threads/--max-rss-mb on small machines.
"""

import argparse
import sys
from pathlib import Path

import yaml

from sql_incremental.manifest import load_manifest
from sql_incremental.sink import make_sink, materialize
from sql_incremental.verify import verify_sink


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dataset", action="append", help="dataset name (repeatable); default: all")
    parser.add_argument("--parquet-dir", default="results/parquet")
    parser.add_argument("--target", choices=("postgres", "duckdb"), default="postgres")
    parser.add_argument("--duckdb-path", default="results/sink.duckdb", help="DuckDB sink file")
    parser.add_argument("--dsn", help="SQLAlchemy DSN; default: db.dsn from the config")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="2500MB")
    parser.add_argument("--max-rss-mb", type=int, default=4500)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    dsn = args.dsn or config.get("db", {}).get("dsn")
    if args.target == "postgres" and not dsn:
        parser.error("--target postgres needs --dsn or db.dsn in the config")
    manifests = [load_manifest(spec) for spec in config.get("datasets", [])]
    if args.dataset:
        unknown = set(args.dataset) - {m.name for m in manifests}
        if unknown:
            parser.error(f"unknown dataset(s): {sorted(unknown)}")
        manifests = [m for m in manifests if m.name in args.dataset]

    sink = make_sink(
        {"type": "duckdb", "path": args.duckdb_path}
        if args.target == "duckdb"
        else {"type": "postgres", "dsn": dsn}
    )
    failed = False
    for manifest in manifests:
        parquet = Path(args.parquet_dir) / f"{manifest.fact_parquet_key()}.parquet"
        options = dict(
            threads=args.threads,
            memory_limit=args.memory_limit,
            batch_size=args.batch_size,
            max_rss_mb=args.max_rss_mb,
        )
        if args.target == "duckdb":
            contour = Path(args.parquet_dir) / f"{manifest.contour_source}.parquet"
            materialize(
                manifest, str(parquet), str(contour), sink,
                threads=args.threads, memory_limit=args.memory_limit,
            )
        result = verify_sink(manifest, str(parquet), sink, **options)
        status = "EQUIVALENT" if result.equivalent else "MISMATCH"
        print(
            f"{manifest.name}: {status} original={result.original[0]} rows "
            f"view={result.view[0]} rows batches={result.batches} peak_rss={result.peak_rss_mb}MB"
        )
        if result.null_key_original != result.null_key_view:
            print(f"  NULL-key rows differ: original={result.null_key_original} view={result.null_key_view}")
        for diff in result.schema_mismatches[:5]:
            print(f"  schema differs at column {diff[0]}: original={diff[1:3]} published={diff[3:5]}")
        for keys, original, view in result.mismatched_batches[:5]:
            print(f"  differing batch {keys}: original={original} view={view}")
        failed = failed or not result.equivalent
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
