"""Main classes."""

import abc
from typing import Protocol

import pandas as pd
import pandera as pa
from pydantic import BaseModel


class TransformBase(Protocol):
    """Base transformation class for data frames."""

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Dataframe transformation callable."""


class IdentityTransform(TransformBase):
    """Simple identity transformation."""

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Do nothing."""
        return data


class Pipeline(TransformBase):
    """Pipelines implementation."""

    def __init__(self, *transforms: TransformBase):
        self.transforms = transforms

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transforms."""
        for transform in self.transforms:
            data = transform(data)
        return data


class Schema(BaseModel, metaclass=abc.ABCMeta):
    """Columns and schema."""

    def schema_member_map(self) -> dict[str, str]:
        """Get schema literals map to column names."""
        columns = self.get_schema().columns
        return {
            lit: col
            for lit, col in self.model_dump().items()
            if col in columns
        }

    def get_schema_dtypes(self) -> dict[str, pa.DataType]:
        """Get data types for each member."""
        member_map = {v: k for k, v in self.schema_member_map().items()}
        return {member_map[k]: v for k, v in self.get_schema().dtypes.items()}

    @abc.abstractmethod
    def get_schema(self) -> pa.DataFrameSchema:
        """Return schema."""


class TransformWithSchema(TransformBase):
    """Transform with schema."""

    def __init__(self, schema: Schema):
        self._schema = schema
