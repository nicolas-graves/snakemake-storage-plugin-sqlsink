"""SQLMesh's advantage: the public name is a view over physical tables that
is repointed, so readers never see a half-built state."""

from __future__ import annotations

from sqlalchemy import inspect, text

from compat_support import MANIFEST, PARTS_V2, capability, rewrite_contours
from sqlsink.sink import materialize
from sqlsink.sink_postgres import SqlSink


@capability("sqlmesh", "virtual_layer_repoint")
def test_the_public_view_switches_between_versions_in_one_step(engine, paths):
    sink = SqlSink(engine)
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    with engine.connect() as conn:
        before = conn.execute(text('SELECT COUNT(*) FROM "zones"')).scalar()

    rewrite_contours(paths, PARTS_V2)
    materialize(MANIFEST, str(paths["facts"]), str(paths["contours"]), sink)
    with engine.connect() as conn:
        after = conn.execute(text('SELECT COUNT(*) FROM "zones"')).scalar()

    assert (before, after) == (3, 4)
    assert inspect(engine).get_view_names() == ["zones"]  # still a view, never a table
