"""Main class of predictable table."""

from typing import Generic, TypeVar

import pandas as pd
import pandera as pa

from predictables import Schema, TransformBase

SchemaType = TypeVar("SchemaType", bound=Schema)


class PredicTable(Generic[SchemaType]):
    """Data with fixed schema."""

    def __init__(self, data: pd.DataFrame, schema: SchemaType) -> None:
        data = schema.get_schema().validate(data)
        self._data = data
        self._schema = schema

    @property
    def data(self) -> pd.DataFrame:
        """Get dataframe."""
        return self._data

    @property
    def schema(self) -> pa.DataFrameSchema:
        """Get schema."""
        return self._schema.get_schema()

    @property
    def c(self) -> SchemaType:
        """Get columns."""
        return self._schema

    @property
    def d(self) -> pd.DataFrame:
        """Get columns."""
        return self._data

    def transform(self, pipeline: TransformBase) -> "PredicTable":
        """Transform data."""
        return self.__class__(pipeline(self.data), self.c)
