import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List

class ParquetWriter:
    def __init__(self, folder: str, schema):
        self.folder = folder
        self.schema = schema
        os.makedirs(folder, exist_ok=True)
        self.counter = 0

    def write_batch(self, entries: List):
        if not entries:
            return
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        df = pd.DataFrame([entry.to_dict() for entry in entries])
        file_path = os.path.join(self.folder, f"part-{self.counter:05d}.parquet")
        self.counter += 1

        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, compression='brotli')