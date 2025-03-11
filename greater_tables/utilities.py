import datetime as dt
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys
from IPython.display import HTML, display

from . greater_tables import GT

# GPT recommended approach
logger = logging.getLogger(__name__)
# Disable log propagation to prevent duplicates
logger.propagate = False
if logger.hasHandlers():
    # Clear existing handlers
    logger.handlers.clear()
# SET DEGBUUGER LEVEL
LEVEL = logging.INFO    # DEBUG or INFO, WARNING, ERROR, CRITICAL
logger.setLevel(LEVEL)
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(LEVEL)
formatter = logging.Formatter('%(asctime)s | %(levelname)s |  %(funcName)-15s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Logger Setup; module recompiled.')


def write_all_tables(out_path='\\s\\telos\\pmir_studynote\\quarto_scratch\\tables.qmd'):
    """Write a tester for all tables to a qmd file."""
    header = '''---
title: {title}
format:
  html:
    html-table-processing: none
  pdf:
    include-in-header: prefobnicate.tex
---

# Set up code 

```{{python}}
#| echo: true 
#| label: setup
%run prefobnicate.py
import proformas as pf

import greater_tables as gter
import greater_tables.utilities as gtu
gter.logger.setLevel(gter.logging.WARNING)
from IPython.display import display

```

...code build completed. 

# Greater_tables Output 
    
```{{python}}
#| echo: true 
#| label: greater-tables-test
test_gen = gtu.TestDFGenerator()
ans = test_gen.test_suite()    
```
    
'''
    template = '''

## Test Table {k}
    
```{{python}}
#| echo: fold
#| label: tbl-greater-tables-test-{i}
#| tbl-cap: Output for test table {k}
hrw = {hrw}
f = gter.GT(ans['{k}'], "{title}", ratio_cols='z', aligners={{'w': 'l'}},
        hrule_widths=hrw)
h = f._repr_html_()
print(f.df.dtypes)
h
```

Comments go here. 

'''
    tdf = TestDFGenerator()
    ans = tdf.test_suite()
    out = [header.format(title='All Tables Test - New TestDFGenerator test_suite')]
    for i, (k, v) in enumerate(ans.items()):
        if v.index.nlevels > 1:
            hrw = (1.5, 1.0, 0.5)
        else:
            hrw = (0,0,0)
        out.append(template.format(i=i, k=k, hrw=hrw, title=k.title()))

    p = Path(out_path)
    p.write_text('\n'.join(out), encoding='utf-8')


# ==================================================
# SUPER DOOPER test df generator with help from GPT
class TestDFGenerator:
    """Make excellent test DataFrames."""
    # Load a list of words
    _word_list_path = 'C:\\s\\Websites\\new_mynl\\word_lists\\match 12.md'
    _word_list = None

    def __init__(self, nan_proportion=0.05, missing_proportion=0,
                title=False, sep='_', file_path=None):
        """Initialise the generator."""
        self.nan_proportion = nan_proportion
        self.missing_proportion = missing_proportion
        self.title = title  # whether to apply title to col names
        self.sep = sep      # separator for column names
        file_path = file_path or TestDFGenerator._word_list_path
        if TestDFGenerator._word_list is None:
            TestDFGenerator._word_list = TestDFGenerator.load_words(file_path)
        # control datatypes
        self.data_types = ["int", "float", "str", "year", "date", 'datetime']
        # types:
        self.index_probs = np.array([20, 1, 20, 45, 12, 5], dtype=float)
        self.index_probs /= self.index_probs.sum()
        # control datatypes, types as above
        self.data_type_probs = np.array([1, 2, 0.5, 0.5, 0.5, 0.5], dtype=float)
        self.data_type_probs /= self.data_type_probs.sum()

    def __repr__(self):
        """Return a string representation."""
        return f"TestDFGenerator({len(self.words):,d} words)"

    @staticmethod
    def load_words(file_path):
        """Load a list of words from a file."""
        p = Path(file_path)
        txt = p.read_text(encoding='utf-8')
        wl = txt.split('\n')
        logger.info(f"Loaded wordlist.")  # Debug print
        return wl

    @property
    def words(self):
        """Return the word list."""
        random.shuffle(self._word_list)
        return self._word_list

    def make_column_names(self, n, g):
        """Make n column names each g words long."""
        if self.title:
            return [self.sep.join(x).title() for x in zip(*[iter(self.words[:n * g])] * g)]
        else:
            return [self.sep.join(x) for x in zip(*[iter(self.words[:n * g])] * g)]

    def make_index_data(self, dtype, size):
        """Generate index values with natural nesting."""
        if dtype == "int":
            values = np.random.randint(0, 100000, size=size)
        elif dtype == "float":
            values = np.random.uniform(-1e6, 1e6, size=size).round(2)
        elif dtype == "str":
            values = np.random.choice(self.words, size=size)
        elif dtype == 'year':
            values = np.random.choice(np.arange(1990, 2030, dtype=int), size=size, replace=False)
        elif dtype == "date":
            start_date = datetime(2020, 1, 1)
            values = [start_date + timedelta(days=random.randint(-5000, 5000)) for _ in range(size)]
        elif dtype == "datetime":
            start_date = datetime(2020, 1, 1)
            values = [start_date + timedelta(days=random.randint(-5000, 5000),
                                                  hours=random.randint(0, 23),
                                                  minutes=random.randint(0, 59),
                                                  seconds=random.randint(0, 59),
                                                  microseconds=random.randint(0, 999999))
                           for _ in range(size)]
        return values   # noqa

    def make_multi_index(self, dtypes, levels, size):
        """Generate a MultiIndex with natural nesting."""
        # lowest level of index
        detailed_index = self.make_index_data(dtypes[-1], size)
        # now make the higher levels, here we want far fewer unique values to make repeats
        higher_levels = []
        for i in range(levels - 1):
            # at level i have i + 2 types?? no just go with 3
            sample = self.make_index_data(dtypes[i], 2 if i==0 else 3)
            higher_levels.append(np.random.choice(sample, size=size))
        index_names = np.random.choice(self.words, levels, replace=False)
        return pd.MultiIndex.from_arrays([*higher_levels, detailed_index], names=index_names)

    def make_column_data(self, dtype, size):
        """Generate column data based on type."""
        if dtype == "int":
            picker = np.random.rand()
            if picker < 0.5:
                return np.random.randint(-10000, 10000, size=size)
            else:
                return np.random.randint(0, 10**9, size=size)
        elif dtype == "float":
            picker = np.random.rand()
            if picker < 0.4:
                return 10. ** np.random.uniform(-9, 1, size=size)
            elif picker < 0.8:
                return 10. ** np.random.uniform(-1, 10, size=size)
            else:
                signs = np.random.choice([-1, 1], size=size)
                return np.pi ** np.random.uniform(-75, 75, size=size) * signs
        elif dtype == "str":
            return np.random.choice(self.words, size=size)
        elif dtype == 'year':
            return np.random.choice(range(1990, 2030), size=size)
        elif dtype == "date":
            start_date = datetime(2020, 1, 1)
            dates = [start_date + timedelta(days=random.randint(-5000, 5000)) for _ in range(size)]
            return pd.to_datetime(np.random.choice([d.strftime("%Y-%m-%d") for d in dates], size=size))
        elif dtype == "datetime":
            start_date = datetime(2020, 1, 1)
            dates = [start_date + timedelta(days=random.randint(-5000, 5000),
                                            hours=random.randint(0, 23),
                                            minutes=random.randint(0, 59),
                                            seconds=random.randint(0, 59),
                                            microseconds=random.randint(0, 999999))
                     for _ in range(size)]
            return pd.to_datetime(np.random.choice([d.strftime("%Y-%m-%d %H:%M:%S.%f") for d in dates], size=size))

    def make_test_dataframe(self,
        num_rows=10,
        num_columns=5,
        num_index_levels=1,
        num_column_levels=1,
        column_name_length=3,
        dtype_label=True,
        index_types=None,
        title=False,
        sep='_'
        ):
        """
        Generate a random pandas DataFrame with diverse structures for testing.

        Parameters:
        - num_rows (int): Number of rows.
        - num_columns (int): Number of columns.
        - num_index_levels (int): Levels in the index (1+).
        - num_column_levels (int): Levels in the columns (1+).
        - column_name_length (int): Words per column name.
        - dtype_label (bool): Whether to tag columns with their type.
        - index_types (list): List of index data types for each level.
        - words (list): List of words for generating column names.

        Returns:
        - pd.DataFrame: A test DataFrame with diverse structures.
        """
        # update
        self.title = title
        self.sep = sep
        # Generate column names
        col_names = self.make_column_names(num_columns,
                                           max(1, column_name_length - (1 if dtype_label else 0)))

        # Randomly select index data types for each level
        if index_types is None:
            index_types = np.random.choice(self.data_types, num_index_levels, p=self.index_probs, replace=True)
        if not isinstance(index_types, (tuple, list)):
            index_types = [index_types]
        if len(index_types) < num_index_levels:
            # well...
            index_types = (index_types * 10)[:num_index_levels]

        # Generate hierarchical MultiIndex with natural grouping
        if num_index_levels > 1:
            index = self.make_multi_index(index_types, num_index_levels, num_rows)
        else:
            name = np.random.choice(self.words, 1)[0]
            index = pd.Index(self.make_index_data(index_types[0], num_rows), name=name)

        # Data types
        dtype_choices = np.random.choice(self.data_types, num_columns, p=self.data_type_probs, replace=True)

        # Generate column structure
        if num_column_levels > 1:
            columns = self.make_multi_index(['str'] * num_column_levels, num_column_levels, num_columns)
            # don't want the index names
            columns.names = [''] * num_column_levels
        else:
            columns = pd.Index([f"{col} {dtype}" if dtype_label else col
                                for col, dtype in zip(col_names, dtype_choices)],
                               name="Column")

        # Generate data
        data = {col: self.make_column_data(dtype, num_rows) for col, dtype in zip(columns, dtype_choices)}
        df = pd.DataFrame(data, index=index, columns=columns)

        # Convert date columns to datetime dtype
        for col, dtype in zip(columns, dtype_choices):
            if dtype == "date":
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Introduce NaNs
        num_nans = int(self.nan_proportion * num_rows * num_columns)
        for _ in range(num_nans):
            df.iat[random.randint(0, num_rows - 1), random.randint(0, num_columns - 1)] = np.nan

        # Introduce radical None values
        if self.missing_proportion:
            num_missing = int(self.missing_proportion * num_rows * num_columns)
            for _ in range(num_missing):
                df.iat[random.randint(0, num_rows - 1), random.randint(0, num_columns - 1)] = None

        df = df.sort_index().sort_index(axis=1)
        return df

    __call__ = make_test_dataframe

    def test_suite(self):
        """Make a dict of test dataframes with different characteristics."""
        ans = {}

        ans['basic'] = self.make_test_dataframe(num_rows=10, num_columns=8,
                                 num_index_levels=1, num_column_levels=1,
                                 column_name_length=1,
                                 index_types=['int'])

        ans['timeseries'] = self.make_test_dataframe(num_rows=20, num_columns=3,
                                 num_index_levels=1, num_column_levels=1,
                                 column_name_length=4, title=True, sep=' ',
                                 index_types=['datetime'])

        ans['multiindex'] = self.make_test_dataframe(num_rows=10, num_columns=5,
                                    num_index_levels=3, num_column_levels=1,
                                    column_name_length=4, title=True, sep=' ',
                                    index_types=['int', 'str'])

        ans['multicolumns'] = self.make_test_dataframe(num_rows=10, num_columns=5,
                                    num_index_levels=1, num_column_levels=3,
                                    column_name_length=4, title=True, sep=' ',
                                    index_types=['int', 'str'])

        ans['complex'] = self.make_test_dataframe(num_rows=20, num_columns=10,
                                    num_index_levels=3, num_column_levels=3,
                                    column_name_length=4,
                                    index_types=['int', 'str'])

        return ans
