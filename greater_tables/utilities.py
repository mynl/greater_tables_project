import datetime as dt
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import random
import sys
from IPython.display import HTML, display
from docs.conf import html_theme

from . greater_tables import GT

# Load a list of words
p = Path('C:\\s\\Websites\\new_mynl\\word_lists\\match 12.md')
word_list = p.read_text().split('\n')


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


# __gt_global = GT()


# def qhtml(df, **kwargs):
#     """Generic "quick display" function."""
#     return HTML(__gt_global(df, **kwargs))


# def qd(df, **kwargs):
#     """Generic "quick display" function."""
#     if isinstance(df, pd.Series):
#         if df.name is None:
#             df.name = 'value'
#         df = df.to_frame()
#     return display(HTML(__gt_global(df, **kwargs)))


def create_three_level_multiindex(df):
    """
    Adds two random levels to a DataFrame's column MultiIndex.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a three-level MultiIndex on the columns.
    """
    n_columns = len(df.columns)
    level_1 = np.random.choice(["A", "B", "C"], size=n_columns)
    level_2 = np.random.choice(["X", "Y", "Z"], size=n_columns)

    # Create the MultiIndex
    multi_index = pd.MultiIndex.from_tuples(
        [(l1, l2, col) for l1, l2, col in zip(level_1, level_2, df.columns)],
        names=["Level 1", "Level 2", df.columns.name]
    )

    # Apply the new MultiIndex to the DataFrame
    df.columns = multi_index
    return df


def test_df(date=False, mi_columns=True):
    """Make a test dataframe nr rows with multi index."""
    nr = 10
    words = 'Parliament organised a year-long programme of events called Parliament in the Making to celebrate the 800th anniversary of the sealing of Magna Carta on 15 June and the 750th anniversary of the first representative parliament on 20 January Events were coordinated with Parliament Week'
    words = list(set(words.split(' ')))
    w1 = ['Abel', 'Cain', 'Issac', 'Fred', 'George', 'Harry', 'Ivan', 'John', 'Karl', 'Lenny', 'Moe', 'Ned', 'Otto', 'Paul', 'Quinn', 'Ralph', 'Steve', 'Tom', 'Ulysses', 'Victor', 'Walter', 'Xavier', 'Yuri', 'Zach']
    w2 = ['South', 'East', 'West', 'North', "North West", "North East", "South West", "South East", "Central", "Outer", "Inner", "Mid", "Upper", "Lower", "Far", "Near", "Middle", "Farthest", "Nearest"]
    w4 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO', 'PP', 'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ']
    rints = [0, -10, 10, 100, -100, 1000, -1000, 10000, -10000, 100000, -100000, 1000000, -1000000]
    df = pd.DataFrame({'idx1': np.random.choice(w1, nr),
                       'idx2': np.random.choice(w2, nr),
                       'idx3': np.random.poisson(2, nr),
                       'floats': np.random.rand(nr) * 3000.,
                       'smaller': np.random.rand(nr) * 10 ** np.linspace(-3, 4, nr),
                       'larger': np.random.choice([-1., 0, 1.], nr) * np.random.rand(nr) * 10 ** np.linspace(3, 12, nr),
                       'ints': np.random.poisson(20, nr),
                       'powers': np.pi * 10. ** np.arange(-20, 26, 5),
                       'ratios': np.random.rand(nr) * 3. - 1.,
                       'string': [' '.join(np.random.choice(words, 4, replace=False)) for i in range(nr)],
                       # 'object': [np.random.poisson(2, nr) for i in range(nr)]
                       })

    if date:
        df['date'] = [dt.datetime.fromordinal(np.random.randint(dt.date(2020, 1, 1).toordinal(),
                                                                dt.date(2030, 1, 1).toordinal())) + dt.timedelta(seconds=np.random.randint(86400)) for _ in range(nr)]
        df['date'] = pd.to_datetime(df['date'])
    df.columns.name = 'Col name'
    df = df.set_index(['idx1', 'idx2', 'idx3'])
    if mi_columns:
        df = create_three_level_multiindex(df)

    # check unique and sort
    df = df.loc[df.index[~df.index.duplicated()]]
    assert np.all(~df.columns.duplicated()), 'Columns not all unique'
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def make_test_dfs():
    """Make a dict of test dataframes with different characteristics."""
    ans = {}
    df = pd.DataFrame({'x': [int(i) for i in 3. ** np.arange(10)], 'y': np.arange(10, dtype=float)})
    df['w'] = df.x ** 0.35
    df['z'] = df.y ** .25

    ans['basic'] = df.copy()
    df1 = df.copy()
    df1.index.name = 'idx name'
    ans['basic w idx name'] = df1.copy()

    df1 = df.copy()
    df1.columns.name = 'col name'
    ans['basic w col name'] = df1.copy()

    df1 = df.copy()
    df1.index.name = 'idx name'
    df1.columns.name = 'col name'
    ans['basic w both names'] = df1.copy()

    df1['date'] = [dt.datetime.fromordinal(np.random.randint(dt.date(2020, 1, 1).toordinal(),
                   dt.date(2030, 1, 1).toordinal())) + dt.timedelta(seconds=np.random.randint(86400))
                    for _ in range(len(df1))]
    df2 = df1.set_index('date')
    ans['time series'] = df2.copy()

    df2 = df1.set_index('date', append=True)
    ans['time series and range'] = df2.copy()

    valid = False
    while not valid:
        a = ans['realistic'] = test_df(date=False, mi_columns=False)
        valid = a.index.is_unique and a.columns.is_unique

    valid = False
    while not valid:
        a = ans['realistic w date'] = test_df(date=True, mi_columns=False).droplevel(2, axis=0)
        valid = a.index.is_unique and a.columns.is_unique

    valid = False
    while not valid:
        a = ans['realistic mi'] = test_df(date=False, mi_columns=True).droplevel(2, axis=1)
        valid = a.index.is_unique and a.columns.is_unique

    valid = False
    while not valid:
        a = ans['realistic mi w date'] = test_df(date=True, mi_columns=True).droplevel(2, axis=0).droplevel(2, axis=1)
        valid = a.index.is_unique and a.columns.is_unique

    return ans



header = '''
---
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
gter.logger.setLevel(gter.logging.WARNING)
from IPython.display import display

```

...code build completed. 

# Greater_tables

```{{python}}
#| echo: true 
#| label: greater-tables-test
ans = gter.make_test_dfs()
```

'''


def write_all_tables():
    """Write a tester for all tables to a qmd file."""
    global header

    template = '''
```{{python}}
#| echo: fold
#| label: tbl-greater-tables-test-{i}
#| tbl-cap: Output for test table {k}
hrw = {hrw}
gter.GT(ans['{k}'], "{title}", ratio_cols='z', aligners={{'w': 'l'}},
        hrule_widths=hrw)
```

SPACER

'''
    ans = make_test_dfs()
    out = [header.format(title='All Tables Test')]
    for i, (k, v) in enumerate(ans.items()):
        if v.index.nlevels > 1:
            hrw = (1.5, 0.5, 0)
        else:
            hrw = (0,0,0)
        out.append(template.format(i=i, k=k, hrw=hrw, title=k.title()))

    p = Path('\\s\\telos\\pmir_studynote\\quarto_scratch\\tables.qmd')
    p.write_text('\n'.join(out), encoding='utf-8')


def apply_css(f, idx):
    """Apply subset of css to the GT object f"""
    style = f.df_style
    split_style = style.split(f'#{f.df_id}')
    print(f'style has {len(split_style)} separate entries')
    n = 2000
    if idx == 0:
        print('RAW')
        display(HTML(f.df.to_html(formatters=f.df_formatters)))
        print('='*80)
    if not isinstance(idx, (tuple, list)):
        idx = [idx]
    newbit = []
    for i in idx:
        code = f'T{n+i}'
        h = f.df_html.replace(f.df_id, code)
        bit = split_style[i].strip()
        if i == 1:
            newbit.append(f'#{code} {split_style[1]}\n')
        else:
            newbit.append(f'#{code} {bit}')
    newbit = '<style>\n' + '\n'.join(newbit) + '\n</style>\n'
    print(newbit)
    newcode = f'{newbit}{h}'
    display(HTML(newcode))


def incremental_qmd(nm, css_bits=22):
    """Write a qmd file with incremental tables."""
    global header

    bit = '''

```{{python}}
#| echo: fold 
#| label: tbl-greater-tables-test-{i}
gter.apply_css(f, {i})
```
'''
    step1 = f'''

## GT Format 

```{{python}}
#| echo: fold
#| label: setup-01
df = ans['{nm}']
f = gter.GT(df, "{nm}", ratio_cols='z', aligners={{'w': 'l'}},
hrule_widths=(1.5, 1, 0.5), vrule_widths=(1.5, 1, 0.5))
f
```

# Incremental Test Suite
 
'''
    out = [header.format(title='Incremental Build Test'), step1]

    for i in range(css_bits):
        out.append(bit.format(i=i))
    p = Path('\\s\\telos\\pmir_studynote\\quarto_scratch\\seq_build.qmd')
    p.write_text('\n'.join(out), encoding='utf-8')


# SUPER DOOPER test df generator with help from GPT
def make_column_names(n, g, words):
    """Make n column names each g words long."""
    return [' '.join(x).title() for x in zip(*[iter(words[:n*g])] * g)]


def generate_test_dataframe(
    num_rows=10,
    num_columns=5,
    num_index_levels=1,
    num_column_levels=1,
    column_name_length=3,
    dtype_label=True,
    nan_proportion=0.05,
    missing_proportion=0.05,
    index_types=None,
    words=None
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
    - nan_proportion (float): Proportion of NaNs.
    - missing_proportion (float): Proportion of None values.
    - index_types (list): List of index data types for each level.
    - words (list): List of words for generating column names.

    Returns:
    - pd.DataFrame: A test DataFrame with diverse structures.
    """
    global word_list  # Ensure access to global words

    # Use default word list if not provided
    if words is None or len(words) < num_columns:
        words = word_list
    random.shuffle(words)

    # Generate column names
    col_names = make_column_names(num_columns, max(1, column_name_length - (1 if dtype_label else 0)), words)

    # Randomly select index data types for each level
    index_dtypes = ["int", "float", "str", "date", "datetime"]
    index_probs = np.array([20, 1, 20, 5, 5], dtype=float)
    index_probs /= index_probs.sum()
    if index_types is None:
        index_types = np.random.choice(index_dtypes, num_index_levels, p=index_probs, replace=True)
    if len(index_types) < num_index_levels:
        # well...
        index_types = (index_types * 10)[:num_index_levels]

    def generate_index_data(dtype, size):
        """Generate index values with natural nesting."""
        if dtype == "int":
            values = np.random.randint(0, 100000, size=size)
        elif dtype == "float":
            values = np.random.uniform(-1e6, 1e6, size=size).round(2)
        elif dtype == "str":
            values = np.random.choice(words, size=size)
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
        return values

    def generate_multi_index(dtypes, levels, num_rows):
        """Generate a MultiIndex with natural nesting."""
        # lowest level of index
        detailed_index = generate_index_data(dtypes[-1], num_rows)
        # now make the higher levels, here we want far fewer unique values to generate repeats
        higher_levels = []
        for i in range(levels - 1):
            # at level i have i + 2 types?? no just go with 3
            sample = generate_index_data(dtypes[i], 2 if i==0 else 3)
            higher_levels.append(np.random.choice(sample, size=num_rows))
        index_names = np.random.choice(words, levels, replace=False)
        return pd.MultiIndex.from_arrays([*higher_levels, detailed_index], names=index_names)

    # Generate hierarchical MultiIndex with natural grouping
    if num_index_levels > 1:
        index = generate_multi_index(index_types, num_index_levels, num_rows)
    else:
        name = np.random.choice(words, 1)[0]
        index = pd.Index(generate_index_data(index_types[0], num_rows), name=name)

    # Data types
    data_types = ["int", "float", "str", "date", 'datetime']
    p = np.array([1, 2, 0.5, 0.5, 0.5], dtype=float)
    p /= p.sum()
    dtype_choices = np.random.choice(data_types, num_columns, p=p, replace=True)

    # Generate column structure
    if num_column_levels > 1:
        columns = generate_multi_index(['str'] * num_column_levels, num_column_levels, num_columns)
        # don't want the index names
        columns.names = [''] * num_column_levels
    else:
        columns = pd.Index([f"{col} {dtype}" if dtype_label else col
                            for col, dtype in zip(col_names, dtype_choices)],
                           name="Column")

    def generate_column_data(dtype):
        """Generate column data based on type."""
        if dtype == "int":
            picker = np.random.rand()
            if picker < 0.5:
                return np.random.randint(-10000, 10000, size=num_rows)
            else:
                return np.random.randint(0, 10**9, size=num_rows)
        elif dtype == "float":
            picker = np.random.rand()
            if picker < 0.4:
                return 10. ** np.random.uniform(-9, 1, size=num_rows)
            elif picker < 0.8:
                return 10. ** np.random.uniform(-1, 10, size=num_rows)
            else:
                signs = np.random.choice([-1, 1], size=num_rows)
                return np.pi ** np.random.uniform(-75, 75, size=num_rows) * signs
        elif dtype == "str":
            return np.random.choice(words, size=num_rows)
        elif dtype == "date":
            start_date = datetime(2020, 1, 1)
            dates = [start_date + timedelta(days=random.randint(-5000, 5000)) for _ in range(num_rows)]
            return np.random.choice([d.strftime("%Y-%m-%d") for d in dates], size=num_rows)
        elif dtype == "datetime":
            start_date = datetime(2020, 1, 1)
            dates = [start_date + timedelta(days=random.randint(-5000, 5000),
                                            hours=random.randint(0, 23),
                                            minutes=random.randint(0, 59),
                                            seconds=random.randint(0, 59),
                                            microseconds=random.randint(0, 999999))
                     for _ in range(num_rows)]
            return np.random.choice([d.strftime("%Y-%m-%d %H:%M:%S.%f") for d in dates], size=num_rows)

    # Generate data
    data = {col: generate_column_data(dtype) for col, dtype in zip(columns, dtype_choices)}
    df = pd.DataFrame(data, index=index, columns=columns)

    # Convert date columns to datetime dtype
    for col, dtype in zip(columns, dtype_choices):
        if dtype == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Introduce NaNs
    num_nans = int(nan_proportion * num_rows * num_columns)
    for _ in range(num_nans):
        df.iat[random.randint(0, num_rows - 1), random.randint(0, num_columns - 1)] = np.nan

    # Introduce radical None values
    num_missing = int(missing_proportion * num_rows * num_columns)
    for _ in range(num_missing):
        df.iat[random.randint(0, num_rows - 1), random.randint(0, num_columns - 1)] = None
    df = df.sort_index().sort_index(axis=1)
    return df


######################################
### roll my own
def make_style(self, spacing='medium', debug=False):
    if debug:
        head_tb = '#0ff'
        body_b = '#f0f'
        h0 = '#f00'
        h1 = '#b00'
        h2 = '#900'
        bh0 = '#f00'
        bh1 = '#b00'
        v0 = '#0f0'
        v1 = '#0a0'
        v2 = '#090'
        padt, padr, padb, padl = 2, 10, 2, 10
    else:
        head_tb = '#000'
        body_b = '#000'
        h0 = '#000'
        h1 = '#000'
        h2 = '#000'
        bh0 = '#000'
        bh1 = '#000'
        v0 = '#000'
        v1 = '#000'
        v2 = '#000'
    table_hrule = 2.5
    if spacing == 'tight':
        padt, padr, padb, padl = 0, 5, 0, 5
    elif spacing == 'medium':
        padt, padr, padb, padl = 2, 10, 2, 10
    elif spacing == 'loose':
        padt, padr, padb, padl = 4, 15, 4, 15
    else:
        raise ValueError('spacing must be tight, medium or loose')

    style = f'''
    <style>
        #{self.df_id}  {{
        border-collapse: collapse;
        font-family: "Roboto", "Open Sans Condensed", "Arial", 'Segoe UI', sans-serif;
        font-size: {self.font_size}em;
        width: auto;
        border: none;
        overflow: auto;    }}
        /* tag formats */
        #{self.df_id} thead {{
            /* top and bottom of header */
            border-top: {table_hrule}px solid {  head_tb};
            border-bottom: {table_hrule}px solid {head_tb};
            }}
        #{self.df_id} tbody {{
            /* bottom of body */ 
            border-bottom: {table_hrule}px solid {body_b};
            }}
        #{self.df_id} tbody th  {{
            vertical-align: top;
        }}
        #{self.df_id} caption {{
            padding-top: 10px;
            padding-bottom: 4px;
            font-size: 1.1em;
            text-align: left;
            font-weight: bold;
            caption-side: top;
        #{self.df_id} td, th {{
            /* top, right, bottom left cell padding */
            padding: {padt}px {padr}px {padb}2px {padl}10px;
            vertical-align: top;
        }}
        }}
        /* class overrides */
        #{self.df_id} .grt-hrule-0 {{
            border-top: {self.hrule_widths[0]}px solid {h0};
        }}
        #{self.df_id} .grt-hrule-1 {{
            border-top: {self.hrule_widths[1]}px solid {h1};
        }}
        #{self.df_id} .grt-hrule-2 {{
            border-top: {self.hrule_widths[2]}px solid {h2};
        }}
        #{self.df_id} .grt-bhrule-0 {{
            border-bottom: {self.hrule_widths[0]}px solid {bh0};
        }}
        #{self.df_id} .grt-bhrule-1 {{
            border-bottom: {self.hrule_widths[1]}px solid {bh1};
        }}
        #{self.df_id} .grt-vrule-0 {{
            border-left: {self.vrule_widths[0]}px solid {v0};
        }}
        #{self.df_id} .grt-vrule-1 {{
            border-left: {self.vrule_widths[1]}px solid {v1};
        }}
        #{self.df_id} .grt-vrule-2 {{
            border-left: {self.vrule_widths[2]}px solid {v2};
        }}        
        #{self.df_id} .grt-left {{
            text-align: left;
        }}
        #{self.df_id} .grt-center {{
            text-align: center;
        }}
        #{self.df_id} .grt-right {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        #{self.df_id} .grt-head {{
            font-family: "Times New Roman", 'Courier New'; 
            font-size: {self.font_size}em;
        }}
    </style>
    '''
    return style


def df_to_html(self, spacing='medium', debug=False):
    """Convert a pandas DataFrame to an HTML table with sparsification."""
    index_name_to_level = dict(zip(self.raw_df.index.names, range(self.nindex)))
    index_change_level = self.index_change_level.map(index_name_to_level)
    # this is easier and computed in the init
    column_change_level = self.column_change_level

    # Start table
    html = [f'<table id="{self.df_id}">']

    # Process header
    bit = self.df.T.reset_index(drop=False)
    idx_header = bit.iloc[:self.nindex, :self.ncolumns]
    columns = bit.iloc[self.nindex:, :self.ncolumns]

    # this is TRANSPOSED!!
    html.append("<thead>")
    for i in range(self.ncolumns):
        # one per row of columns m index, usually only 1
        # TODO Add header aligners
        html.append("<tr>")
        for j, r in enumerate(idx_header.iloc[:, i]):
            # columns one per level of index
            html.append(f'<th class="grt-left">{r}</th>')
        for j, r in enumerate(columns.iloc[:, i]):
            # one per column of dataframe
            # figure how high up mindex the vrules go
            # all headings get hrules, it's the vrules that are tricky
            hrule = f'grt-bhrule-{i}' if i < self.ncolumns - 1 else ''
            if 0 < j < self.ncols and i >= column_change_level[j]:
                vrule = f'grt-vrule-{column_change_level[j]}'
            elif j == 0:
                # start with the first column come what may
                vrule = f'grt-vrule-{column_change_level[0]}'
            else:
                vrule = ''
            html.append(f'<th class="grt-center {hrule} {vrule}">{r}</th>')
        html.append("</tr>")
    html.append("</thead>")

    html.append("<tbody>")
    for i, (n, r) in enumerate(self.df.iterrows()):
        # one per row of dataframe
        html.append("<tr>")
        hrule = ''
        for j, c in enumerate(r.iloc[:self.nindex]):
            # dx = data in index
            # if this is the level that changes for this row
            # will use a top rule  hence omit i = 0 which already has an hrule
            if i > 0 and hrule == '' and j == index_change_level[i]:
                hrule = f'grt-hrule-{j}'
            html.append(f'<td class="grt-dx-r-{i} grt-dx-c-{j} {self.df_aligners[j]} {hrule}">{c}</td>')
        for j, c in enumerate(r.iloc[self.nindex:]):
            # first col left handled by index/body divider
            if 0 < j < self.ncols:
                vrule = f'grt-vrule-{column_change_level[j]}'
            elif j == 0:
                # start with the first column come what may
                vrule = f'grt-vrule-{column_change_level[0]}'
            html.append(f'<td class="grt-data-r-{i} grt-data-c-{j} {self.df_aligners[j+self.nindex]} {hrule} {vrule}">{c}</td>')
        html.append("</tr>")
    html.append("</tbody>")
    return '\n'.join(html)


def to_html(self, spacing='medium', debug=False):
    """Full monty, raw string."""
    html = df_to_html(self, spacing=spacing, debug=debug)
    style = make_style(self, spacing=spacing, debug=debug)
    return style + html


def process(self, spacing='medium', debug=False):
    """Full monty."""
    return HTML(to_html(self, spacing=spacing, debug=debug))

