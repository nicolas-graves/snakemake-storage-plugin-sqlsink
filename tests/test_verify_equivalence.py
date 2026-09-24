import duckdb

from sqlsink.verify import compare_batched

COLUMNS = ["zone_id", "metric", "polygon_coords"]


def _con_with(original_rows, view_rows):
    con = duckdb.connect()
    for table, rows in (("original", original_rows), ("view", view_rows)):
        con.execute(f"CREATE TABLE {table} (zone_id VARCHAR, metric INTEGER, polygon_coords VARCHAR)")
        con.executemany(f"INSERT INTO {table} VALUES (?, ?, ?)", rows)
    return con


ROWS = [("Z1", 10, "a"), ("Z1", 10, "a"), ("Z1", 10, "b"), ("Z2", 20, "c"), ("Z3", None, "d")]


def test_equal_multisets_are_equivalent_regardless_of_order_and_batching():
    con = _con_with(ROWS, list(reversed(ROWS)))
    for batch_size in (1, 2, 10):
        result = compare_batched(con, "original", "view", COLUMNS, "zone_id", batch_size=batch_size)
        assert result.equivalent
        assert result.original == result.view
        assert result.original[0] == len(ROWS)


def test_duplicate_multiplicity_difference_is_detected():
    view_rows = [r for i, r in enumerate(ROWS) if i != 1]  # drop one of the two identical Z1 rows
    con = _con_with(ROWS, view_rows)
    result = compare_batched(con, "original", "view", COLUMNS, "zone_id", batch_size=1)
    assert not result.equivalent
    assert [batch for batch, _, _ in result.mismatched_batches] == [["Z1"]]


def test_changed_value_is_detected():
    view_rows = [("Z2", 21, "c") if r[0] == "Z2" else r for r in ROWS]
    con = _con_with(ROWS, view_rows)
    result = compare_batched(con, "original", "view", COLUMNS, "zone_id")
    assert not result.equivalent


def test_rows_with_a_null_key_are_compared_not_ignored():
    original = ROWS + [(None, 1, "x")]
    con = _con_with(original, ROWS)  # the view lost the NULL-key row

    result = compare_batched(con, "original", "view", COLUMNS, "zone_id")

    assert result.original == result.view  # keyed batches are identical...
    assert result.null_key_original == (1, result.null_key_original[1])
    assert result.null_key_view[0] == 0
    assert not result.equivalent  # ...but the NULL-key group is not


def test_key_present_only_in_the_view_is_a_mismatched_batch():
    con = _con_with(ROWS, ROWS + [("Z9", 1, "extra")])

    result = compare_batched(con, "original", "view", COLUMNS, "zone_id")

    assert not result.equivalent
    assert any(batch == ["Z9"] for batch, _, _ in result.mismatched_batches)
