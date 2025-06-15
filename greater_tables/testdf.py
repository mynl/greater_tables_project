"""
Make fake dataframes for testing.

GPT from SJMM design.
"""

from collections import deque
from datetime import datetime, timedelta
from importlib.resources import files
from itertools import cycle, chain
from math import prod
from pathlib import Path
from typing import Optional, Union
import hashlib
import random
import re

import numpy as np
import pandas as pd


name_word_list = [
    "account",
    "address",
    "amount",
    "balance",
    "category",
    "client",
    "combined ratio",
    "comment",
    "currency",
    "description",
    "duration",
    "email",
    "entry",
    "estimate",
    "extension",
    "failure",
    "filename",
    "identifier",
    "location",
    "loss ratio",
    "note",
    "operation",
    "premium",
    "processing",
    "project",
    "reference",
    "remark",
    "status",
    "supplier",
    "timestamp",
    "transaction",
    "type",
    "user",
    'expense ratio',
    'loss date'
]


class TestDataFrameFactory:
    """
    Create super-dooper test dataframes.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Factory for generating small synthetic pandas DataFrames for testing.

        Attributes:
            seed: Optional random seed. If None, one is generated.
        """
        self._last_args = {}
        self.seed = int(
            seed if seed is not None else np.random.SeedSequence().entropy)

        # rng
        self.rng = np.random.default_rng(self.seed)

        # word list for names of index levels
        nwl = name_word_list[:]
        random.shuffle(nwl)
        self._index_namer = cycle(nwl)

        # read words and create cycler
        data_path = files('greater_tables').joinpath('data', 'words-12.md')
        with data_path.open('r', encoding='utf-8') as f:
            txt = f.read()
        word_list = txt.split('\n')
        temp = word_list[:]
        random.shuffle(temp)
        self._word_gen = cycle(temp)

        # read tex expressions and create cycler
        data_path = files('greater_tables').joinpath('data', 'tex_list.csv')
        with data_path.open('r', encoding='utf-8') as f:
            tex_list = pd.read_csv(f, index_col=0)['expr'].to_list()
        # trim down slightly
        tex_list = [i for i in tex_list if len(i) < 50]
        random.shuffle(tex_list)
        self._tex_gen = cycle(tex_list)

        self.simple_namer = {
            'd': 'date',
            'f': 'float',
            'h': 'hash',
            'i': 'integer',
            'l': 'large_float',
            'm': 'yr-mo',
            'p': 'path',
            'r': 'ratio',
            's': 'string',
            't': 'time',
            'v': 'extreme_float',
            'x': 'tex',
            'y': 'year',
        }

        # lengths of index (word count) sampled from:
        self.index_value_lengths = [1]*10 + [2] * 4 + [3]
        self.cache = deque(maxlen=10)

    # def cache(self, n=0):
    #     """Get nth item ago from cache, default = 0, latest."""
    #     if n < len(self._cache):
    #         return self._cache[n]
    #     else:
    #         print(f'Cache only contains {len(self._cache)} < {n} items.')

    def make(self, rows: int, columns: Union[int, str], index: Union[int, str] = 0,
             col_index: Union[int, str] = 0, missing: float = 0.0) -> pd.DataFrame:
        """
        Generate a test DataFrame with the given specification.

        Data types

            d   date
            f   float
            h   hash
            i   integer
            l   log float (greater range than float)
            m   year - month
            p   path (filename)
            r   ratio (smaller floats, for percents)
            sx  string length x
            t   time
            v   very large range float
            x   tex text - an equation
            y   year

            Args:
            rows: Number of rows.
            columns: Column type spec (int for all float cols, or string type codes).
            index: Index level types (int for RangeIndex or string like 'ti').
            col_index: Column index levels (same format as `index`).
            missing: Proportion of missing data in each column.

        Returns:
            DataFrame
        """
        self._last_args = dict(rows=rows, columns=columns,
                               index=index, col_index=col_index, missing=missing)
        return self._generate(**self._last_args)

    def another(self, new_seed: bool = True) -> pd.DataFrame:
        """
        Generate another DataFrame with the last parameters.

        Args:
            new_seed: If True, re-randomize the generator seed.

        Returns:
            DataFrame
        """
        if new_seed:
            self.seed = int(np.random.SeedSequence().entropy)
            self.rng = np.random.default_rng(self.seed)
        return self._generate(**self._last_args)

    def random(self, index_levels: int = 0, column_levels: int = 0, omit: str = 'p') -> pd.DataFrame:
        """
        Generate a DataFrame with randomly chosen settings.


        Args:
            index_levels: Number of index levels to use.
            column_levels: Number of column MultiIndex levels.
            omit: omit column datatypes in omit
        Returns:
            DataFrame
        """
        if index_levels == 0:
            index_levels = random.choice([1, 1, 1, 1, 1, 2, 2, 3])
        if column_levels == 0:
            column_levels = random.choice([1, 1, 1, 1, 1, 2, 2, 3])
        rows = self.rng.integers(5 * index_levels, 10 * index_levels)
        valid_types = [i for i in ['d', 'f', 'i', 's3', 'l', 'h', 't', 'p', 'x', 'r', 'y']
        if i not in omit]
        col_types = self.rng.choice(
            valid_types, size=self.rng.integers(3, 7))
        missing = round(float(self.rng.uniform(0, 0.15)), 2)
        index = ''.join(self.rng.choice(
            ['t', 'd', 'y', 'i', 's2'], size=index_levels))
        col_index = ''.join(self.rng.choice(
            ['s', 's2', 's2', 's3'], size=column_levels))
        return self.make(rows=rows, columns=''.join(col_types), index=index, col_index=col_index, missing=missing)

    def _generate(self, rows: int, columns: Union[int, str], index: Union[int, str],
                  col_index: Union[int, str], missing: float) -> pd.DataFrame:
        # if columns is an int then make up types
        if isinstance(columns, int):
            col_types = self.rng.choice(
                ['d', 't', 'f', 'l', 'i', 's1', 's3', 's9', 'h', 'p', 'x'], size=columns)
        else:
            col_types = self._parse_colspec(columns)
        # if col_index is an int then use all strings of that depth
        if col_index == 'simple':
            col_idx = map(self.simple_namer.get, [i[0] for i in col_types])
            col_idx = pd.Index(col_idx, name='simple')
        else:
            if isinstance(col_index, int):
                col_index_types = ['s'] * col_index
            else:
                col_index_types = self._parse_colspec(col_index)
            col_idx = self._make_index(col_index_types, len(col_types))
        if isinstance(index, int):
            index = ['s'] * index
        else:
            index = self._parse_colspec(index)
            # print(index)
        # col names are a transposed index.
        df = pd.DataFrame(index=range(rows))
        for dt, c in zip(col_types, range(len(col_idx))):
            df[c] = self._generate_column(dt, rows)
        df.columns = col_idx
        df.index = self._make_index(index, rows)
        df = self._insert_missing(df, missing)
        self.cache.appendleft(df)
        return df

    def _parse_colspec(self, spec: str) -> list[str]:
        return re.findall(r's\d+|[a-z]', spec)

    def _generate_column(self, dtype: str, n: int) -> pd.Series:
        if dtype.startswith('s'):
            max_words = int(dtype[1:]) if len(dtype) > 1 else 1
            return pd.Series([" ".join(self.word() for i in range(max_words)) for j in range(n)])
        if dtype == 'f':
            return pd.Series(self.rng.normal(loc=100000, scale=250000, size=n))
        if dtype == 'r':
            return pd.Series(self.rng.normal(loc=0.5, scale=0.35, size=n))
        if dtype == 'l':
            # log float (greater range)
            return pd.Series(np.exp(self.rng.normal(loc=-4 / 2 + 4, scale=4, size=n)))
        if dtype == 'v':
            # log float (greater range)
            sc = 5
            return pd.Series(np.exp(self.rng.normal(loc=-sc**2 / 2 + 10, scale=sc, size=n)))
        if dtype == 'i':
            return pd.Series(self.rng.integers(-1e4, 1e6, size=n), dtype='int64')
        if dtype == 'd':
            start_date = TestDataFrameFactory.random_date_within_last_n_years(
                10)
            return pd.Series(pd.date_range(start=start_date, periods=n, freq='D'))
        if dtype == 'y':
            return pd.Series(random.sample(range(1990, 2031), n))
        if dtype == 't':
            start_dt = datetime.now() - timedelta(days=365 * 2)
            return pd.Series([
                start_dt +
                timedelta(minutes=int(self.rng.integers(0, 2 * 365 * 24 * 60)))
                for _ in range(n)
            ])
        if dtype == 'h':
            return pd.Series([
                hashlib.blake2b(f"val{i}".encode(), digest_size=32).hexdigest()
                for i in range(n)
            ])
        if dtype == 'p':
            return pd.Series([str(Path(f"/data/{self.word()}/{i}.dat")) for i in range(n)])
        if dtype == 'x':
            # tex
            return pd.Series([self.tex() for i in range(n)])
        raise ValueError(f"Unknown dtype: {dtype}")

    def _make_index(self, desc: Union[int, str, list[str]], n: int) -> pd.Index:
        if isinstance(desc, int):
            return pd.RangeIndex(n, name=self.index_name())
        if isinstance(desc, str):
            desc = self._parse_colspec(desc)
        if len(desc) == 1:
            if desc[0] == 'i':
                return pd.RangeIndex(n, name=self.index_name())
            elif desc[0] in ('d', 't', 'x', 'y'):
                vals = self._generate_column(desc[0], n)
                return pd.Index(vals, name=self.index_name())
            elif not all(i[0] == 's' for i in desc):
                raise ValueError(
                    f'Inadmissible index spec: only string, int, and date types allowed, not {desc}.')
        level_value_lengths = [1 if len(i) == 1 else int(i[1:]) for i in desc]
        return self.make_index(rows=n, levels=len(desc), level_value_lengths=level_value_lengths,
                               p0=1, padding=2)

    def index_name(self):
        """Return a one-word index name."""
        return next(self._index_namer)

    def word(self):
        """Return a random word (cycles eventually)."""
        return next(self._word_gen)

    def tex(self):
        """Return a blob of TeX."""
        return next(self._tex_gen)

    @staticmethod
    def random_date_within_last_n_years(n: int) -> pd.Timestamp:
        today = datetime.today()
        days = random.randint(0, n * 365)
        return pd.Timestamp(today - timedelta(days=days))

    def _insert_missing(self, df: pd.DataFrame, prop: float) -> pd.DataFrame:
        """Insert missing values into dataframe."""
        if prop <= 0:
            return df
        n_rows = df.shape[0]
        for col in df.columns:
            n_missing = max(1, int(np.floor(prop * n_rows)))
            missing_indices = self.rng.choice(
                n_rows, size=n_missing, replace=False)
            df.iloc[missing_indices, df.columns.get_loc(col)] = np.nan
        return df

    @staticmethod
    def _is_prime(p: int) -> bool:
        if p < 2:
            return False
        if p == 2:
            return True
        if p % 2 == 0:
            return False
        for i in range(3, int(p**0.5) + 1, 2):
            if p % i == 0:
                return False
        return True

    @staticmethod
    def _next_prime(p: int) -> int:
        if p < 2:
            return 2
        p += 1 if p % 2 == 0 else 2  # ensure odd start > p
        while True:
            if TestDataFrameFactory._is_prime(p):
                return p
            p += 2

    @staticmethod
    def primes_for_product(n: int, v: int, p0: int) -> list[int]:
        """Return a list of distinct primes all >= p0 whose product is >= n."""
        primes = []
        p = TestDataFrameFactory._next_prime(max(p0 - 1, 1))
        while len(primes) < v:
            primes.append(p)
            p = TestDataFrameFactory._next_prime(p)

        while prod(primes := sorted(primes)) < n:
            # increase one level until product is high enough
            p = TestDataFrameFactory._next_prime(primes[-1])
            primes[-1] = p
        # shuffle order
        random.shuffle(primes)
        return primes

    def make_index(self, rows: int, levels: int,
                   level_value_lengths: Union[list[int], None] = None,
                   p0: int = 1,
                   padding: int = 2):
        """
        Make an Index with unique values, rows x len(level_value_lengths) cols.

        level_velue_lengths shows how many words long each value should be.
        padding = over-sample by padding and select sample.
        """
        if level_value_lengths is None:
            level_value_lengths = random.sample(
                self.index_value_lengths, levels)
        else:
            assert levels == len(
                level_value_lengths), 'levels must equal len(level_value_lengths)'
        level_choices = self.primes_for_product(rows * padding, levels, p0=p0)
        r = [cycle([' '.join([self.word() for _ in range(w)]) for _ in range(k)])
             for w, k in zip(level_value_lengths, level_choices)]
        x = [[next(j) for j in r] for i in range(rows)]
        names = random.sample(name_word_list, levels)
        if levels == 1:
            idx = pd.Index(
                list(chain.from_iterable(random.sample(x, rows))), name=names[0]).sort_values()
        else:
            idx = pd.MultiIndex.from_tuples(
                random.sample(x, rows), names=names).sort_values()
        assert idx.is_unique
        return idx
