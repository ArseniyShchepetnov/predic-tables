A way to fix [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) data schema with [pandera](https://pandera.readthedocs.io/en/stable/) and use autocomplete for column names.

```python
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
```
