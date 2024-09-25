"""Main classes."""

import abc
from typing import Generic, Protocol, TypeVar

import pandas as pd
import pandera as pa


class TransformBase(Protocol):
    """Base transformation class for data frames."""

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dataframe transformation callable."""


class IdentityTransform(TransformBase):
    """Simple identity transformation."""

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Do nothing."""
        return data


class Schema(metaclass=abc.ABCMeta):
    """Columns and schema."""

    @abc.abstractmethod
    def schema(self) -> pa.DataFrameSchema:
        """Return schema."""


SchemaType = TypeVar("SchemaType", bound=Schema)


class PredicTable(Generic[SchemaType]):
    """Data with fixed schema."""

    def __init__(self, data: pd.DataFrame, schema: SchemaType) -> None:
        data = schema.schema().validate(data)
        self._data = data
        self._schema = schema

    @property
    def data(self) -> pd.DataFrame:
        """Get dataframe."""
        return self._data

    @property
    def schema(self) -> pa.DataFrameSchema:
        """Get schema."""
        return self._schema.schema()

    @property
    def c(self) -> SchemaType:
        """Get columns."""
        return self._schema

    def transform(self, pipeline: TransformBase) -> "PredicTable":
        """Transform data."""
        return self.__class__(pipeline(self.data), self.c)
