"""
Fabricate dataframes for testing.
"""

from collections import deque
from datetime import datetime, timedelta
from importlib.resources import files
from itertools import cycle, chain, count, zip_longest, product, islice
from math import prod
from pathlib import Path
from typing import Optional, Union
import hashlib
import random
import re

import numpy as np
import pandas as pd

from IPython.display import display


class Fabricator:
    """
    Fabricate dataframes.
    """

    metric_roots = ['absorption', 'acceleration', 'account', 'activation', 'adjustment', 'allocation', 'amplitude', 'approval', 'asset', 'atom', 'attrition', 'balance', 'band', 'binding', 'cancellation', 'capacitance', 'capital', 'cashflow', 'category', 'cell', 'charge', 'claim', 'commission', 'compound', 'concentration', 'conductivity', 'constraint', 'consumption', 'conversion', 'correlation', 'cost', 'count', 'coverage', 'credit', 'current', 'debt', 'decay', 'decibel', 'deductible', 'deficit', 'deflator', 'demand', 'density', 'development', 'diffusion', 'discount', 'distribution', 'dividend', 'dose', 'duration', 'earnings', 'efficiency', 'elasticity', 'employment', 'energy', 'entropy', 'enzyme', 'estimate', 'excess', 'exhaustion', 'expense', 'exposure', 'failure', 'field', 'flux', 'force', 'frequency', 'funding', 'gdp', 'gene', 'gradient', 'growth', 'half_life', 'incidence', 'income', 'index', 'indicator', 'inequality', 'inflation', 'inhibition', 'input', 'intensity', 'investment', 'kurtosis', 'lapse', 'layer', 'leverage', 'liability', 'limit', 'loss', 'luminosity', 'margin', 'mass', 'molecule', 'momentum', 'mortality', 'neutron', 'noise', 'operating', 'output', 'penalty', 'photon', 'policy', 'portfolio', 'potential', 'power', 'preference', 'premium', 'pressure', 'price', 'productivity', 'profit', 'protein', 'proton', 'provision', 'radiation', 'rate', 'ratio', 'reaction', 'recovery', 'reflection', 'refraction', 'renewal', 'reserve', 'residual', 'resistance', 'return', 'revenue', 'risk', 'sample', 'savings', 'scenario', 'score', 'sector', 'settlement', 'severity', 'shock', 'shortfall', 'signal', 'skewness', 'spread', 'strain', 'stress', 'subsidy', 'supply', 'tail', 'tariff', 'tax', 'temperature', 'tension', 'term', 'threshold', 'trade', 'trend', 'turbulence', 'unemployment', 'uptake', 'utility', 'utilization', 'valuation', 'variance', 'velocity', 'viscosity', 'volatility', 'voltage', 'volume', 'wage', 'wavelength', 'wealth', 'weight', 'yield']

    metric_suffix = ["", "rate", "score", "amount", "index", "ratio", "factor", "value"]

    def __init__(self, seed: Optional[int] = None):
        """
        Fabricate small synthetic pandas DataFrames for testing.

        Attributes:
            seed: Optional random seed. If None, one is generated.
        """
        self._last_args = {}
        self.seed = int(
            seed if seed is not None else np.random.SeedSequence().entropy)

        # rng
        self.rng = np.random.default_rng(self.seed)

        # word list for names of index levels
        nwl = self.metric_roots[:]
        self.rng.shuffle(nwl)
        self._metric_namer = cycle(nwl)

        # read words and create cycler
        data_path = files('greater_tables').joinpath('data', 'words-12.md')
        with data_path.open('r', encoding='utf-8') as f:
            txt = f.read()
        word_list = txt.split('\n')
        temp = word_list[:]
        self.rng.shuffle(temp)
        self._word_gen = cycle(temp)

        # read tex expressions and create cycler
        data_path = files('greater_tables').joinpath('data', 'tex_list.csv')
        with data_path.open('r', encoding='utf-8') as f:
            tex_list = pd.read_csv(f, index_col=0)['expr'].to_list()
        # trim down slightly
        pat = re.compile(r'(?<!\\)\b[a-z]{4,}\b')
        tex_list = [i for i in tex_list if not pat.search(i) and len(i)<=50]
        self.rng.shuffle(tex_list)
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

    @staticmethod
    def roll_columns(df, levels=-1):
        """"Roll" the column MultiIndex round by levels, default makes top bottom, rest move up."""
        idx = df.columns
        idx = idx.reorder_levels(np.roll(range(df.columns.nlevels), levels))
        df.columns = idx
        df = df.sort_index(axis=1)
        return df

    def uber(self, rows, data_spec, *, index_levels=1, index_names=None, column_groups=1, column_levels=1, column_names=None, decorate=False, simplify=True, oversample=1):
        """
        Fabricate a dataframe.

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


        metrics
        total num cols = metrics x column_groups
        """
        # validate args
        assert column_levels <= column_groups, 'Column levels must be <= groups'
        assert index_names is None or len(index_names) == index_levels, 'Index names must have length index_levels'
        assert column_names is None or len(column_names) == column_levels, 'Column names must have length column_levels'

        # figure data_spec and hence (important) number of metrics
        data_spec = self._parse_colspec(data_spec)
        metrics = len(data_spec)
        if oversample > 1:
            df = self.uber(oversample * rows, metrics, data_spec, index_levels=index_levels,

                index_names=index_names, column_groups=column_groups, column_levels=column_levels,
                column_names=column_names, decorate=decorate, oversample=1)
            df = df.iloc[:rows, :]
            return df

        inames = index_names or [f'i_{i}' for i in range(index_levels)]
        index = pd.MultiIndex.from_tuples(islice(product(*(self._generate_column('s', v) for v in self.primes_for_product(rows, index_levels))), rows), names=inames)

        # create with col groups and drop later if needed
        cnames = (column_names or [f'c_{i}' for i in range(column_levels)]) + ['metric']
        columns_pfp = self.primes_for_product(column_groups, column_levels)
        cgroup_product = product(*(self._generate_column('s', v) for v in columns_pfp))
        # take first column_groups entries - islice works without creating the full iterable
        cgroup_product = islice(cgroup_product, column_groups)
        # add metrics
        metric_names = [self.metric_name for _ in range(metrics)]
        cgroup_product = product(cgroup_product, metric_names)
        # flatten
        cgroup_product = [(*x, y) for x, y in cgroup_product]
        columns = pd.MultiIndex.from_tuples(cgroup_product, names=cnames)

        # create empty df
        df = pd.DataFrame(index=index, columns=columns)

        if df.shape[1] != prod(columns_pfp):
            print("Incomplete column...won't unstack")
            print(df.shape[1], prod(columns_pfp ))

        # fill in the data, data_spec x column_groups groups
        for c, dt in zip(df.columns, data_spec * column_groups):
            df[c] =self._generate_column(dt, rows).values

        if simplify:
            df = self.drop_singleton_levels(df)

        return df

    @staticmethod
    def drop_singleton_levels(df):
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel([i for i, lvl in enumerate(df.index.levels)
                               if len(lvl) == 1])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel([i for i, lvl in enumerate(df.columns.levels)
                                               if len(lvl) == 1])
        return df

    # def make(self, rows: int, columns: Union[int, str], index: Union[int, str] = 0,
    #          col_index: Union[int, str] = 0, missing: float = 0.0) -> pd.DataFrame:
    #     """
    #         Generate a test DataFrame with the given specification.


    #         Args:
    #         rows: Number of rows.
    #         columns: Column type spec (int for all float cols, or string type codes).
    #         index: Index level types (int for RangeIndex or string like 'ti').
    #         col_index: Column index levels (same format as `index`).
    #         missing: Proportion of missing data in each column.

    #     Returns:
    #         DataFrame
    #     """
    #     self._last_args = dict(rows=rows, columns=columns,
    #                            index=index, col_index=col_index, missing=missing)
    #     return self._generate(**self._last_args)

    # def another(self) -> pd.DataFrame:
    #     """
    #     Generate another DataFrame with the last parameters.

    #     Returns:
    #         DataFrame
    #     """
    #     return self._generate(**self._last_args)

    # def random(self, index_levels: int = 0, column_levels: int = 0, omit: str = 'p') -> pd.DataFrame:
    #     """
    #     Generate a DataFrame with randomly chosen settings.

    #     Args:
    #         index_levels: Number of index levels to use.
    #         column_levels: Number of column MultiIndex levels.
    #         omit: omit column datatypes in omit
    #     Returns:
    #         DataFrame
    #     """
    #     if index_levels == 0:
    #         index_levels = int(self.choice([1, 2, 3], p=(.6, .3, .1)))
    #     if column_levels == 0:
    #         column_levels =  int(self.choice([1, 2, 3], p=(.5, .3, .2)))
    #     rows = self.rng.integers(5 * index_levels, 10 * index_levels)
    #     valid_types = [i for i in ['d', 'f', 'i', 's3', 'l', 'h', 't', 'p', 'x', 'r', 'y'] if i not in omit]
    #     col_types = self.rng.choice(
    #         valid_types, size=self.rng.integers(3, 7))
    #     missing = round(float(self.rng.uniform(0, 0.15)), 2)
    #     index = ''.join(self.rng.choice(
    #         ['t', 'd', 'y', 'i', 's2'], size=index_levels))
    #     col_index = ''.join(self.rng.choice(
    #         ['s', 's2', 's2', 's3'], size=column_levels))
    #     return self.make(rows=rows, columns=''.join(col_types), index=index, col_index=col_index, missing=missing)

    def _parse_colspec(self, spec: str) -> list[str]:
        return re.findall(r's\d+|[a-z]', spec)

    # def _generate_column_ex(self, dtype: str, n: int, d: int = 0, r: int = 0) -> pd.Series:
    #     """
    #     Generate a sample of n values from d distinct values of dtype repeated r times

    #     If d == 0 generate unique values.
    #     """
    #     if d == 0:
    #         return self._generate_column(dtype, n)
    #     assert d > 0
    #     base = self._generate_column(dtype, d)
    #     return pd.Series(np.repeat(base.values, r)[:n])

    def _generate_column(self, dtype: str, n: int) -> pd.Series:
        """Generate a sample of n distinct values of dtype."""
        if dtype.startswith('s'):
            max_words = int(dtype[1:]) if len(dtype) > 1 else 1
            return pd.Series([" ".join(self.word() for i in range(max_words)) for j in range(n)])
        if dtype == 'f':
            return pd.Series(self.rng.normal(loc=100000, scale=250000, size=n))
        if dtype == 'r':
            return pd.Series(self.rng.normal(loc=0.5, scale=0.35, size=n))
        if dtype == 'l':
            # log float (greater range)
            scale = 10.
            return pd.Series(np.exp(self.rng.normal(loc=-scale**2 / 2 + 15, scale=scale, size=n)))
        if dtype == 'v':
            # log float (greater range)
            sc = 5
            return pd.Series(np.exp(self.rng.normal(loc=-sc**2 / 2 + 10, scale=sc, size=n)))
        if dtype == 'i':
            return pd.Series(self.rng.integers(-1e4, 1e6, size=n), dtype='int64')
        if dtype == 'd':
            start_date = Fabricator.random_date_within_last_n_years(
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

    # def _make_index(self, desc: Union[int, str, list[str]], n: int) -> pd.Index:
    #     if isinstance(desc, int):
    #         return pd.RangeIndex(n, name=self.index_name())
    #     if isinstance(desc, str):
    #         desc = self._parse_colspec(desc)
    #     if len(desc) == 1:
    #         if desc[0] == 'i':
    #             return pd.RangeIndex(n, name=self.index_name())
    #         elif desc[0] in ('d', 't', 'x', 'y'):
    #             vals = self._generate_column(desc[0], n)
    #             return pd.Index(vals, name=self.index_name())
    #         elif not all(i[0] == 's' for i in desc):
    #             raise ValueError(
    #                 f'Inadmissible index spec: only string, int, and date types allowed, not {desc}.')
    #     level_value_lengths = [1 if len(i) == 1 else int(i[1:]) for i in desc]
    #     return self.make_index(rows=n, levels=len(desc), level_value_lengths=level_value_lengths,
    #                            p0=1, padding=2)

    @property
    def metric_name(self):
        """Return a one-word metric name."""
        return next(self._metric_namer)

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
            if Fabricator._is_prime(p):
                return p
            p += 2

    def primes_for_product(self, n: int, v: int, shuffle: bool = False) -> list[int]:
        """Return a list of v distinct primes whose product is >= n."""
        # starting prime is next after p0
        if n == 1:
            # still want it to work for n = 1
            return [1]
        p0 = max(1, int(n ** (1 / (v))))
        primes = []
        p = Fabricator._next_prime(max(p0 - 1, 1))
        while len(primes) < v:
            primes.append(p)
            p = Fabricator._next_prime(p)

        while prod(primes := sorted(primes)) < n:
            # increase one level until product is high enough
            p = Fabricator._next_prime(primes[-1])
            primes[-1] = p
        # shuffle order? really hierarchical order will go smallest to largest...
        # but for rules other orders may be of interest?
        if shuffle:
            self.rng.shuffle(primes)
        return primes

    # def make_index(self, rows: int, levels: int,
    #                level_value_lengths: Union[list[int], None] = None,
    #                p0: int = 1,
    #                padding: int = 2):
    #     """
    #     Make an Index with unique values, rows x len(level_value_lengths) cols.

    #     level_velue_lengths shows how many words long each value should be.
    #     padding = over-sample by padding and select sample.
    #     """
    #     if level_value_lengths is None:
    #         level_value_lengths = random.sample(
    #             self.index_value_lengths, levels)
    #     else:
    #         assert levels == len(
    #             level_value_lengths), 'levels must equal len(level_value_lengths)'
    #     level_choices = self.primes_for_product(rows * padding, levels, p0=p0)
    #     r = [cycle([' '.join([self.word() for _ in range(w)]) for _ in range(k)])
    #          for w, k in zip(level_value_lengths, level_choices)]
    #     x = [[next(j) for j in r] for i in range(rows)]
    #     names = random.sample(name_word_list, levels)
    #     if levels == 1:
    #         idx = pd.Index(
    #             list(chain.from_iterable(random.sample(x, rows))), name=names[0]).sort_values()
    #     else:
    #         idx = pd.MultiIndex.from_tuples(
    #             random.sample(x, rows), names=names).sort_values()
    #     assert idx.is_unique
    #     return idx
