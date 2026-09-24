"""One-time bootstrap: seed marker rows for tables already known-good.

Run manually once, NOT wired into the Snakemake DAG (it must never be
replayed automatically). For each table in SEEDED_TABLES, this computes
the update_id from the table's *current* Parquet file and inserts/updates
the corresponding marker row directly, without touching the public table
or going through staging.

This exists to avoid redundantly reloading all 37 tables on the first run
of the new incremental system: 32 of them already passed the equivalence
audit against their currently-published state, so we tell the new system
"trust what's already published" for those, and only stage+publish the
genuinely changed ones.

Before running against production, verify each seeded table's live row
count matches expectations (see `--verify-only`) — seeding a marker for a
table whose public state is subtly wrong would make the system believe
it's current forever, until the Parquet changes again.

Usage:
    python seed_markers.py --dsn postgresql://... --table t1 --table t2 ...
    python seed_markers.py --dsn postgresql://... --tables-file seeded_tables.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sql_incremental.engine import make_engine  # noqa: E402
from sql_incremental.fingerprint import compute_update_id_for_file  # noqa: E402
from sql_incremental.metadata import analytics_table_updates, create_all  # noqa: E402
from sqlalchemy import inspect, select, text  # noqa: E402


def seed_table(engine, table_name: str, parquet_path: str, verify_only: bool = False) -> None:
    update_id, parquet_sha256 = compute_update_id_for_file(table_name, parquet_path)

    with engine.begin() as conn:
        inspector = inspect(conn)
        if not inspector.has_table(table_name):
            raise RuntimeError(
                f"refusing to seed {table_name!r}: public table does not exist"
            )
        row_count = conn.execute(text(f'SELECT count(*) FROM "{table_name}"')).scalar_one()

        print(f"{table_name}: public row_count={row_count}, update_id={update_id}")
        if verify_only:
            return

        existing = conn.execute(
            select(analytics_table_updates.c.table_name).where(
                analytics_table_updates.c.table_name == table_name
            )
        ).scalar_one_or_none()
        values = dict(
            table_name=table_name,
            update_id=update_id,
            parquet_sha256=parquet_sha256,
            loader_version=1,
            type_map_version=1,
            row_count=row_count,
            published_by="seed_markers",
        )
        if existing is None:
            conn.execute(analytics_table_updates.insert().values(**values))
        else:
            conn.execute(
                analytics_table_updates.update()
                .where(analytics_table_updates.c.table_name == table_name)
                .values(**values)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--parquet-dir", default="results/parquet")
    parser.add_argument("--table", action="append", default=[], help="repeatable")
    parser.add_argument("--tables-file", help="one table name per line")
    parser.add_argument(
        "--verify-only", action="store_true", help="print row counts, write nothing"
    )
    args = parser.parse_args()

    tables = list(args.table)
    if args.tables_file:
        tables += [line.strip() for line in Path(args.tables_file).read_text().splitlines() if line.strip()]
    if not tables:
        parser.error("provide at least one --table or --tables-file")

    engine = make_engine(args.dsn)
    create_all(engine)

    for table_name in tables:
        parquet_path = str(Path(args.parquet_dir) / f"{table_name}.parquet")
        seed_table(engine, table_name, parquet_path, verify_only=args.verify_only)


if __name__ == "__main__":
    main()
