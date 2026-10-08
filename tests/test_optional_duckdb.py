"""Planning and package imports work when DuckDB is not installed."""

import os
import subprocess
import sys
from pathlib import Path


def test_imports_without_duckdb():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "workflow" / "scripts"), str(root / "workflow" / "sql_storage_plugin"), env.get("PYTHONPATH", "")]
    )
    code = """
import builtins
original_import = builtins.__import__
def without_duckdb(name, *args, **kwargs):
    if name == 'duckdb' or name.startswith('duckdb.'):
        raise ModuleNotFoundError("No module named 'duckdb'", name='duckdb')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_duckdb
import sqlsink
import sqlsink.stage
import sqlsink.sink
import sqlsink.sink_postgres
import sqlsink.verify
import sqlsink.components
import snakemake_storage_plugin_sqlsink
try:
    sqlsink.stage._read_parquet_schema_and_stats('unused.parquet')
except ImportError as exc:
    assert 'sqlsink[duckdb]' in str(exc)
else:
    raise AssertionError('DuckDB operation unexpectedly succeeded')
try:
    sqlsink.sink.make_sink({'type': 'duckdb', 'path': 'unused.duckdb'})
except ImportError as exc:
    assert 'sqlsink[duckdb]' in str(exc)
else:
    raise AssertionError('DuckDB sink unexpectedly succeeded')
"""
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
