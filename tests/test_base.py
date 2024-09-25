"""Test base classes."""

import pandas as pd
import pandera as pa
import pytest

from predictables.base import PredicTable, Schema


def test_default():
    """Default test case."""

    class TestSchema(Schema):

        column_a: str = "Column A"
        column_b: str = "Column B"

        def schema(self) -> pa.DataFrameSchema:
            """Return schema."""
            return pa.DataFrameSchema(
                {
                    self.column_a: pa.Column(str),
                    self.column_b: pa.Column(int),
                }
            )

    class TestTable(PredicTable[TestSchema]):
        """Test table."""

    test_data = pd.DataFrame({"Column A": ["a", "b", "c"], "Column B": [1, 2, 3]})
    test_table = TestTable(test_data, TestSchema())
    assert test_data.equals(test_table.data)


def test_wrong_schema_default():
    """Default test case."""

    class TestSchema(Schema):

        column_a: str = "Column A"
        column_b: str = "Column B"

        def schema(self) -> pa.DataFrameSchema:
            """Return schema."""
            return pa.DataFrameSchema(
                {
                    self.column_a: pa.Column(
                        str, pa.Check.str_matches(r"^[a-z0-9-]+$")
                    ),
                    self.column_b: pa.Column(int, pa.Check.less_than(100)),
                }
            )

    class TestTable(PredicTable[TestSchema]):
        """Test table."""

    test_data = pd.DataFrame({"Column A": ["a", "b", "!"], "Column B": [1, 2, 1001]})
    with pytest.raises(pa.errors.SchemaError):
        TestTable(test_data, TestSchema())
