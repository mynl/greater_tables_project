"""
Make fake dataframes for testing.

GPT from SJMM design.
"""

# from pathlib import Path
# from dataclasses import dataclass, field
# from typing import Optional, Union
# from datetime import datetime, timedelta
# import hashlib
# import re


# import numpy as np
# import pandas as pd
# from faker import Faker


# @dataclass
# class TestDataFrameFactory:
#     """
#     Factory for generating small synthetic pandas DataFrames for testing.

#     Attributes:
#         colname_words: Optional list of strings to use for column names.
#         default_word_count: Max number of words for string columns (default 3).
#         seed: Optional random seed. If None, one is generated.
#     """
#     colname_words: Optional[list[str]] = None
#     default_word_count: int = 3
#     seed: Optional[int] = None
#     _last_args: dict = field(default_factory=dict, init=False)

#     def __post_init__(self):
#         self.faker = Faker()
#         self.seed = int(self.seed if self.seed is not None else np.random.SeedSequence().entropy)
#         self.rng = np.random.default_rng(self.seed)

#     def make(self, rows: int, columns: Union[int, str], index: Union[int, str] = 0,
#              col_index: Union[int, str] = 0, missing: float = 0.0) -> pd.DataFrame:
#         """
#         Generate a test DataFrame with the given specification.

#         Args:
#             rows: Number of rows.
#             columns: Column type spec (int for all float cols, or string type codes).
#             index: Index level types (int for RangeIndex or string like 'ti').
#             col_index: Column index levels (same format as `index`).
#             missing: Proportion of missing data in each column.

#         Returns:
#             DataFrame
#         """
#         self._last_args = dict(rows=rows, columns=columns, index=index, col_index=col_index, missing=missing)
#         return self._generate(**self._last_args)

#     def another(self, new_seed: bool = True) -> pd.DataFrame:
#         """
#         Generate another DataFrame with the last parameters.

#         Args:
#             new_seed: If True, re-randomize the generator seed.

#         Returns:
#             DataFrame
#         """
#         if new_seed:
#             self.seed = int(np.random.SeedSequence().entropy)
#             self.rng = np.random.default_rng(self.seed)
#         return self._generate(**self._last_args)

#     def random(self, index_levels: int = 1, column_levels: int = 1) -> pd.DataFrame:
#         """
#         Generate a DataFrame with randomly chosen settings.

#         Args:
#             index_levels: Number of index levels to use.
#             column_levels: Number of column MultiIndex levels.

#         Returns:
#             DataFrame
#         """
#         rows = self.rng.integers(10, 50)
#         col_types = self.rng.choice(['d', 'f', 'i', 's1', 's3', 's7', 'h', 't', 'p'], size=self.rng.integers(3, 7))
#         missing = round(float(self.rng.uniform(0, 0.15)), 2)
#         index = ''.join(self.rng.choice(['t', 'd', 'i', 's2'], size=index_levels))
#         col_index = ''.join(self.rng.choice(['s', 'i', 'd'], size=column_levels))
#         return self.make(rows=rows, columns=''.join(col_types), index=index, col_index=col_index, missing=missing)

#     def _parse_colspec(self, spec: str) -> list[str]:
#         return re.findall(r's\d+|[a-z]', spec)


#     def _generate(self, rows: int, columns: Union[int, str], index: Union[int, str],
#                   col_index: Union[int, str], missing: float) -> pd.DataFrame:
#         if isinstance(columns, int):
#             col_types = ['s3'] * columns
#         else:
#             col_types = self._parse_colspec(columns)

#         colnames = self._make_column_names(len(col_types))
#         data = {
#             name: self._generate_column(dt, rows) for name, dt in zip(colnames, col_types)
#         }
#         df = pd.DataFrame(data)
#         df.index = self._make_index(index, rows, "i")
#         df.columns = self._make_index(col_index, len(df.columns), "c") if isinstance(col_index, str) else df.columns
#         df = self._insert_missing(df, missing)
#         return df

#     def _make_column_names(self, n: int) -> list[str]:
#         if self.colname_words:
#             pool = self.colname_words
#         else:
#             pool = [self.faker.word() for _ in range(n * 2)]
#         names = []
#         used = set()
#         for word in pool:
#             if len(names) >= n:
#                 break
#             if word not in used:
#                 names.append(word)
#                 used.add(word)
#         while len(names) < n:
#             names.append(f"col_{len(names)}")
#         return names

#     def _generate_column(self, dtype: str, n: int) -> pd.Series:
#         if dtype.startswith('s'):
#             max_words = int(dtype[1:]) if len(dtype) > 1 else self.default_word_count
#             return pd.Series([" ".join(self.faker.words(self.rng.integers(max_words // 2 + 1, max_words + 1))) for _ in range(n)])
#         if dtype == 'f':
#             return pd.Series(self.rng.normal(loc=100, scale=25, size=n))
#         if dtype == 'i':
#             return pd.Series(self.rng.integers(1e9, 1e12, size=n), dtype='int64')
#         if dtype == 'd':
#             start_date = self.faker.date_between(start_date='-10y', end_date='today')
#             return pd.Series(pd.date_range(start=start_date, periods=n, freq='D'))
#         if dtype == 't':
#             start_dt = datetime.now() - timedelta(days=365 * 2)
#             return pd.Series([start_dt + timedelta(minutes=int(self.rng.integers(0, 2 * 365 * 24 * 60))) for _ in range(n)])
#         if dtype == 'h':
#             return pd.Series([
#                 hashlib.blake2b(f"val{i}".encode(), digest_size=32).hexdigest()
#                 for i in range(n)
#             ])
#         if dtype == 'p':
#             return pd.Series([str(Path(f"/data/{self.faker.word()}/{i}.dat")) for i in range(n)])
#         raise ValueError(f"Unknown dtype: {dtype}")

#     def _make_index(self, desc: Union[int, str], n: int, label_prefix: str) -> pd.Index:
#         if isinstance(desc, int):
#             return pd.RangeIndex(n, name=f"{label_prefix}0")
#         levels = []
#         names = []
#         for j, dt in enumerate(desc):
#             s = self._generate_column(dt, n)
#             levels.append(s)
#             names.append(f"{label_prefix}{j}")
#         return pd.MultiIndex.from_arrays(levels, names=names)

#     def _insert_missing(self, df: pd.DataFrame, prop: float) -> pd.DataFrame:
#         if prop <= 0:
#             return df
#         n_rows = df.shape[0]
#         for col in df.columns:
#             n_missing = max(1, int(np.floor(prop * n_rows)))
#             missing_indices = self.rng.choice(n_rows, size=n_missing, replace=False)
#             df.iloc[missing_indices, df.columns.get_loc(col)] = np.nan
#         return df


# Reimport necessary modules after kernel reset
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union
from datetime import datetime, timedelta
import hashlib
import numpy as np
import pandas as pd
from faker import Faker
import re

@dataclass
class TestDataFrameFactory:
    colname_words: Optional[list[str]] = None
    default_word_count: int = 3
    seed: Optional[int] = None
    _last_args: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.faker = Faker()
        self.seed = int(self.seed if self.seed is not None else np.random.SeedSequence().entropy)
        self.rng = np.random.default_rng(self.seed)

    def make(self, rows: int, columns: Union[int, str], index: Union[int, str] = 0,
             col_index: Union[int, str] = 0, missing: float = 0.0) -> pd.DataFrame:
        self._last_args = dict(rows=rows, columns=columns, index=index, col_index=col_index, missing=missing)
        return self._generate(**self._last_args).sort_index()

    def another(self, new_seed: bool = True) -> pd.DataFrame:
        if new_seed:
            self.seed = int(np.random.SeedSequence().entropy)
            self.rng = np.random.default_rng(self.seed)
        return self._generate(**self._last_args).sort_index()

    def random(self, index_levels: int = 1, column_levels: int = 1) -> pd.DataFrame:
        rows = self.rng.integers(10, 50)
        col_types = self.rng.choice(['d', 'f', 'i', 's3', 'h', 't', 'p'], size=self.rng.integers(3, 7))
        missing = round(float(self.rng.uniform(0, 0.15)), 2)
        index = ''.join(self.rng.choice(['t', 'd', 'i', 's2'], size=index_levels))
        col_index = ''.join(self.rng.choice(['s', 'i', 'd'], size=column_levels))
        return self.make(rows=rows, columns=''.join(col_types), index=index, col_index=col_index, missing=missing)

    def _generate(self, rows: int, columns: Union[int, str], index: Union[int, str],
                  col_index: Union[int, str], missing: float) -> pd.DataFrame:
        col_types = ['f'] * columns if isinstance(columns, int) else self._parse_colspec(columns)
        colnames = self._make_column_names(len(col_types))
        data = {
            name: self._generate_column(dt, rows) for name, dt in zip(colnames, col_types)
        }
        df = pd.DataFrame(data)
        df.index = self._make_index(index, rows, "i")
        df.columns = self._make_index(col_index, len(df.columns), "c") if isinstance(col_index, str) else df.columns
        df = self._insert_missing(df, missing)
        return df

    def _parse_colspec(self, spec: str) -> list[str]:
        return re.findall(r's\d+|[a-z]', spec)

    def _make_column_names(self, n: int) -> list[str]:
        if self.colname_words:
            pool = self.colname_words
        else:
            pool = [self.faker.word() for _ in range(n * 2)]
        names, used = [], set()
        for word in pool:
            if len(names) >= n:
                break
            if word not in used:
                names.append(word)
                used.add(word)
        while len(names) < n:
            names.append(f"col_{len(names)}")
        return names

    def _generate_column(self, dtype: str, n: int) -> pd.Series:
        if dtype.startswith('s'):
            max_words = int(dtype[1:]) if len(dtype) > 1 else self.default_word_count
            return pd.Series([" ".join(self.faker.words(self.rng.integers(1, max_words + 1))) for _ in range(n)])
        if dtype == 'f':
            return pd.Series(self.rng.normal(loc=100, scale=25, size=n))
        if dtype == 'i':
            return pd.Series(self.rng.integers(1e9, 1e12, size=n), dtype='int64')
        if dtype == 'd':
            start_date = self.faker.date_between(start_date='-10y', end_date='today')
            return pd.Series(pd.date_range(start=start_date, periods=n, freq='D'))
        if dtype == 't':
            start_dt = datetime.now() - timedelta(days=365 * 2)
            return pd.Series([
                start_dt + timedelta(minutes=int(self.rng.integers(0, 2 * 365 * 24 * 60)))
                for _ in range(n)
            ])
        if dtype == 'h':
            return pd.Series([
                hashlib.blake2b(f"val{i}".encode(), digest_size=32).hexdigest()
                for i in range(n)
            ])
        if dtype == 'p':
            return pd.Series([str(Path(f"/data/{self.faker.word()}/{i}.dat")) for i in range(n)])
        raise ValueError(f"Unknown dtype: {dtype}")

    def _make_index(self, desc: Union[int, str], n: int, label_prefix: str) -> pd.Index:
        if isinstance(desc, int):
            return pd.RangeIndex(n, name=f"{label_prefix}0")
        if len(desc) == 1:
            s = self._generate_column(desc[0], n)
            return pd.Index(s, name=f"{label_prefix}0")
        return self._make_hierarchical_index(desc, n, label_prefix)

    def _make_hierarchical_index(self, desc: str, n: int, label_prefix: str) -> pd.MultiIndex:
        """
        Generate a nested hierarchical index of length `n` with `len(desc)` levels.
        Levels are naturally nested, i.e., upper levels have fewer unique values.
        """
        levels = []

        # generate lower-level (more detailed) values with full cardinality
        detailed = self._generate_column(desc[-1], n)
        levels.insert(0, detailed)

        # generate higher levels with fewer unique values
        for i, dt in enumerate(desc[:-1]):
            u = 2 if i == 0 else 3
            unique_vals = self._generate_column(dt, u).unique()
            repeated = self.rng.choice(unique_vals, size=n, replace=True)
            levels.insert(0, repeated)

        names = [f"{label_prefix}{j}" for j in range(len(desc))]
        return pd.MultiIndex.from_arrays(levels, names=names)


    def _insert_missing(self, df: pd.DataFrame, prop: float) -> pd.DataFrame:
        if prop <= 0:
            return df
        n_rows = df.shape[0]
        for col in df.columns:
            n_missing = max(1, int(np.floor(prop * n_rows)))
            missing_indices = self.rng.choice(n_rows, size=n_missing, replace=False)
            df.iloc[missing_indices, df.columns.get_loc(col)] = np.nan
        return df
