"""Parquet file writing utilities.

This module provides a simple writer for incrementally writing batches of
log entries to Parquet part files.

Classes:
    - ParquetWriter: Writes batches of entries to numbered Parquet files.

.. autoclass:: ParquetWriter
    :members:
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Any, Type


class ParquetWriter:
    """Writer for incremental Parquet file output.

    Writes batches of log entries as numbered part files (part-00000.parquet,
    part-00001.parquet, etc.) in a specified folder.

    Attributes:
        folder (str): Output folder path.
        schema (Type): Dataclass type for entries (used for type reference).
        counter (int): Current part file number.
    """

    def __init__(self, folder: str, schema: Type) -> None:
        """Initialize the Parquet writer.

        Args:
            folder (str): Output folder path. Created if it doesn't exist.
            schema (Type): Dataclass type for entries being written.
        """
        self.folder = folder
        self.schema = schema
        os.makedirs(folder, exist_ok=True)
        self.counter = 0

    def write_batch(self, entries: List[Any]) -> None:
        """Write a batch of entries to a new Parquet part file.

        Args:
            entries (List[Any]): List of dataclass instances with to_dict() method.

        Notes:
            Does nothing if entries list is empty.
        """
        if not entries:
            return
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        df = pd.DataFrame([entry.to_dict() for entry in entries])
        file_path = os.path.join(self.folder, f"part-{self.counter:05d}.parquet")
        self.counter += 1

        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression='snappy')