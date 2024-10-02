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

    test_data = pd.DataFrame(
        {"Column A": ["a", "b", "c"], "Column B": [1, 2, 3]}
    )
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

    test_data = pd.DataFrame(
        {"Column A": ["a", "b", "!"], "Column B": [1, 2, 1001]}
    )
    with pytest.raises(pa.errors.SchemaError):
        TestTable(test_data, TestSchema())


def test_replace_generic_in_child():
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

    class NextTestSchema(TestSchema):

        column_c: str = "Column C"

        def schema(self) -> pa.DataFrameSchema:
            """Return schema."""
            parent = super().schema()
            return parent.add_columns(
                {
                    self.column_c: pa.Column(
                        int, pa.Check.greater_than_or_equal_to(100)
                    )
                }
            )

    class NextTestTable(TestTable, PredicTable[NextTestSchema]):
        """Test table."""

    test_data = pd.DataFrame(
        {
            "Column A": ["a", "b", "c"],
            "Column B": [1, 2, 3],
            "Column C": [100, 200, 300],
        }
    )
    test_table = NextTestTable(test_data, NextTestSchema())
    assert test_data.equals(test_table.data)

    test_data = pd.DataFrame(
        {
            "Column A": ["a", "b", "c"],
            "Column B": [1, 2, 3],
            "Column C": [1, 2, 3],
        }
    )
    with pytest.raises(pa.errors.SchemaError):
        test_table = NextTestTable(test_data, NextTestSchema())
