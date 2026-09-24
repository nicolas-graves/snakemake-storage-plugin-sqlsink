"""PostgreSQL COPY payloads keep NULL and the empty string apart."""

import csv
import io

from sql_incremental.engine import _csv_rows


def test_null_is_an_unquoted_empty_field_and_empty_string_is_quoted():
    payload = _csv_rows([(None, "", 'a"b,c\nd', 1)])
    assert payload == ',"","a""b,c\nd","1"\n'


def test_the_payload_round_trips_through_a_csv_reader_except_for_null():
    rows = [("x", "", "y,z"), ("", "q", 'w"')]
    assert [tuple(r) for r in csv.reader(io.StringIO(_csv_rows(rows)))] == rows


def test_bulk_load_keeps_empty_strings_and_nulls_apart(engine):
    from sqlalchemy import Column, MetaData, Table, Text, select

    from sql_incremental.engine import bulk_load

    table = Table("copy_probe", MetaData(), Column("a", Text), Column("b", Text))
    with engine.begin() as conn:
        table.create(conn)
        bulk_load(conn, table, [{"a": "", "b": None}, {"a": None, "b": ""}])
        rows = sorted(map(tuple, conn.execute(select(table.c.a, table.c.b))), key=repr)
    assert rows == sorted([("", None), (None, "")], key=repr)
