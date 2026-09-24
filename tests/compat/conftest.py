"""Fixtures for the capability suite, plus the capability matrix.

Every test carries `@capability(tool, id, kind)`. After the run, the matrix
of (tool, capability) -> parity / beyond / gap is printed; with
`--compat-scorecard` it is also written to `tests/compat/SCORECARD.md`.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from fixtures.make_fixtures import make_multipart_geometry_fixtures
from sql_incremental.sink_postgres import SqlSink

_RESULTS = pytest.StashKey[dict]()


def pytest_addoption(parser):
    parser.addoption("--compat-scorecard", action="store_true", help="write tests/compat/SCORECARD.md")


def pytest_configure(config):
    config.addinivalue_line("markers", "capability(tool, id, kind, note): map a test to a reference-tool advantage")
    config.stash[_RESULTS] = defaultdict(dict)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    marker = item.get_closest_marker("capability")
    if marker is None or report.when != "call":
        return
    kw = marker.kwargs
    if hasattr(report, "wasxfail"):
        status = "gap" if report.skipped else "FAIL"  # xpassed strict => failed report
    else:
        status = "ok" if report.passed else "FAIL"
    entry = item.config.stash[_RESULTS][(kw["tool"], kw["id"])]
    entry.setdefault("kind", kw["kind"])
    entry.setdefault("note", kw.get("note", ""))
    order = {"ok": 0, "gap": 1, "FAIL": 2}
    if order[status] >= order[entry.get("status", "ok")]:
        entry["status"] = status


def render_matrix(results) -> list[str]:
    by_tool = defaultdict(list)
    for (tool, cid), entry in sorted(results.items()):
        by_tool[tool].append((cid, entry))
    lines = []
    for tool, entries in by_tool.items():
        counts = defaultdict(int)
        for _, e in entries:
            counts[e["kind"] if e["status"] != "FAIL" else "FAIL"] += 1
        summary = ", ".join(f"{counts[k]} {k}" for k in ("parity", "beyond", "gap", "FAIL") if counts[k])
        lines.append(f"{tool}: {summary}")
        for cid, e in entries:
            mark = {"ok": "ok  ", "gap": "GAP ", "FAIL": "FAIL"}[e["status"]]
            lines.append(f"  [{mark}] {e['kind']:<6} {cid}" + (f"  -- {e['note']}" if e["note"] else ""))
    return lines


def pytest_terminal_summary(terminalreporter, config):
    results = config.stash[_RESULTS]
    if not results:
        return
    lines = render_matrix(results)
    terminalreporter.section("capability matrix")
    for line in lines:
        terminalreporter.write_line(line)
    if config.getoption("--compat-scorecard"):
        path = Path(__file__).parent / "SCORECARD.md"
        path.write_text("# Capability scorecard (generated)\n\n```\n" + "\n".join(lines) + "\n```\n")
        terminalreporter.write_line(f"scorecard written to {path}")


@pytest.fixture
def paths(tmp_path):
    return make_multipart_geometry_fixtures(tmp_path / "parquet")


@pytest.fixture
def sink(engine):
    """The SQL sink behind the one API (a DuckDB file stands in for PostgreSQL)."""
    return SqlSink(engine)
